/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "fpu/softfloat.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/* ============================================================
 * Helper: initialize softfloat status from lane fcsr
 * ============================================================ */
static void lane_init_fp_status(GPGPULane *lane)
{
    set_float_rounding_mode(float_round_nearest_even, &lane->fp_status);
    set_float_exception_flags(0, &lane->fp_status);
}

/* ============================================================
 * Warp initialization
 * ============================================================ */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));

    warp->warp_id = warp_id;
    warp->thread_id_base = thread_id_base;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    warp->active_mask = 0;
    for (uint32_t i = 0; i < num_threads && i < GPGPU_WARP_SIZE; i++) {
        GPGPULane *lane = &warp->lanes[i];
        memset(lane, 0, sizeof(*lane));
        lane->pc = pc;
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id,
                                       thread_id_base + i);
        lane->active = true;
        lane_init_fp_status(lane);
        warp->active_mask |= (1u << i);
    }
}

/* ============================================================
 * VRAM memory access helpers (GPU core address space)
 * 0x00000000..vram_size-1  -> VRAM
 * 0x80000000..             -> CTRL registers (per-lane context)
 * ============================================================ */
static uint32_t vram_load_word(GPGPUState *s, uint32_t addr)
{
    if (addr + 4 > s->vram_size) {
        return 0;
    }
    uint32_t val;
    memcpy(&val, s->vram_ptr + addr, 4);
    return val;
}


static uint32_t mem_load(GPGPUState *s, GPGPULane *lane,
                         uint32_t addr, unsigned bytes)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        uint32_t off = addr - GPGPU_CORE_CTRL_BASE;
        switch (off) {
        case 0x00: return lane->mhartid & 0x1F;      /* thread_id_x */
        case 0x04: return 0;
        case 0x08: return 0;
        case 0x10: return (lane->mhartid >> 13) & 0x7FFFF;  /* block derived */
        default:   return 0;
        }
    }
    if (addr + bytes > s->vram_size) {
        return 0;
    }
    uint32_t val = 0;
    memcpy(&val, s->vram_ptr + addr, bytes);
    return val;
}

static void mem_store(GPGPUState *s, uint32_t addr,
                      uint32_t val, unsigned bytes)
{
    if (addr >= GPGPU_CORE_CTRL_BASE) {
        return;
    }
    if (addr + bytes > s->vram_size) {
        return;
    }
    memcpy(s->vram_ptr + addr, &val, bytes);
}

/* ============================================================
 * Low-precision float conversions (BF16, E4M3, E5M2, E2M1)
 * ============================================================ */

/* fcvt.bf16.s  fd = bf16(f32(fs1))  funct7=0x22, rs2=1 */
/* fcvt.s.bf16  fd = f32(bf16(fs1))  funct7=0x22, rs2=0 */

/* E4M3: sign(1)+exp(4,bias=7)+mant(3)=8bit, max=448, no Inf */
static float32 fp8_e4m3_to_f32(float8_e4m3 fp8, float_status *st)
{
    bfloat16 bf = float8_e4m3_to_bfloat16(fp8, st);
    return bfloat16_to_float32(bf, st);
}

static float8_e4m3 f32_to_fp8_e4m3_sat(float32 f, float_status *st)
{
    return float32_to_float8_e4m3(f, true, st);
}

/* E5M2: sign(1)+exp(5,bias=15)+mant(2)=8bit */
static float32 fp8_e5m2_to_f32(float8_e5m2 fp8, float_status *st)
{
    bfloat16 bf = float8_e5m2_to_bfloat16(fp8, st);
    return bfloat16_to_float32(bf, st);
}

static float8_e5m2 f32_to_fp8_e5m2_sat(float32 f, float_status *st)
{
    return float32_to_float8_e5m2(f, true, st);
}

/*
 * E2M1: sign(1)+exp(2,bias=1)+mant(1) = 4 bits
 * Representable positive values:
 *   bits 000->0.0, 001->0.5, 010->1.0, 011->1.5,
 *   100->2.0, 101->3.0, 110->4.0, 111->6.0 (max)
 * No Inf/NaN.  Saturation to +-6.0 for overflow/Inf/NaN.
 */
static float32 fp4_e2m1_to_f32(float4_e2m1 fp4)
{
    static const float lut[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    uint8_t sign = (fp4 >> 3) & 1;
    uint8_t mag  = fp4 & 0x7;
    float v = lut[mag];
    if (sign) {
        v = -v;
    }
    uint32_t bits;
    memcpy(&bits, &v, 4);
    return bits;
}

static float4_e2m1 f32_to_fp4_e2m1(float32 f32bits)
{
    /* Threshold table: pick E2M1 value by comparing abs(f32) to midpoints.
     * midpoints between consecutive values (round-to-nearest-even):
     *   0.0 <-> 0.5: mid = 0.25
     *   0.5 <-> 1.0: mid = 0.75
     *   1.0 <-> 1.5: mid = 1.25
     *   1.5 <-> 2.0: mid = 1.75
     *   2.0 <-> 3.0: mid = 2.5
     *   3.0 <-> 4.0: mid = 3.5
     *   4.0 <-> 6.0: mid = 5.0
     *   >6.0       : saturate to 6.0
     */
    uint32_t sign_bit = f32bits & 0x80000000u;
    uint32_t absval   = f32bits & 0x7FFFFFFFu;

    /* NaN or Inf -> saturate to 6.0 */
    if (absval >= 0x7F800000u) {
        return sign_bit ? 0xF : 0x7;   /* -6.0 or +6.0 */
    }

    float fabs_v;
    memcpy(&fabs_v, &absval, 4);

    uint8_t mag;
    if (fabs_v >= 5.0f) {
        mag = 7; /* 6.0 */
    } else if (fabs_v >= 3.5f) {
        mag = 6; /* 4.0 */
    } else if (fabs_v >= 2.5f) {
        mag = 5; /* 3.0 */
    } else if (fabs_v >= 1.75f) {
        mag = 4; /* 2.0 */
    } else if (fabs_v >= 1.25f) {
        mag = 3; /* 1.5 */
    } else if (fabs_v >= 0.75f) {
        mag = 2; /* 1.0 */
    } else if (fabs_v >= 0.25f) {
        mag = 1; /* 0.5 */
    } else {
        mag = 0; /* 0.0 */
    }

    uint8_t s = sign_bit ? 1 : 0;
    return (float4_e2m1)((s << 3) | mag);
}

/* ============================================================
 * Apply rm (rounding mode) field to fp_status
 * ============================================================ */
static void apply_rm(unsigned rm, float_status *st)
{
    switch (rm) {
    case 0: set_float_rounding_mode(float_round_nearest_even, st); break;
    case 1: set_float_rounding_mode(float_round_to_zero, st); break;
    case 2: set_float_rounding_mode(float_round_down, st); break;
    case 3: set_float_rounding_mode(float_round_up, st); break;
    case 4: set_float_rounding_mode(float_round_ties_away, st); break;
    case 7: /* dynamic – use fcsr.frm (already set) */ break;
    default: set_float_rounding_mode(float_round_nearest_even, st); break;
    }
}

/* ============================================================
 * Execute one instruction for all active lanes in a warp
 * Returns:
 *   0  continue
 *   1  all lanes done (active_mask == 0)
 *  -1  fatal error
 * ============================================================ */

/* Instruction decode fields (RV32) */
#define OPCODE(i)    ((i) & 0x7F)
#define RD(i)        (((i) >> 7) & 0x1F)
#define FUNCT3(i)    (((i) >> 12) & 0x7)
#define RS1(i)       (((i) >> 15) & 0x1F)
#define RS2(i)       (((i) >> 20) & 0x1F)
#define FUNCT7(i)    (((i) >> 25) & 0x7F)
#define IMM_I(i)     ((int32_t)(i) >> 20)
#define IMM_S(i)     ((int32_t)(((i) & 0xFE000000) | (((i) >> 7) & 0x1F) << 7) >> 20)
#define IMM_B(i)     ((int32_t)((((i) >> 31) << 12) | ((((i) >> 7) & 1) << 11) | \
                      ((((i) >> 25) & 0x3F) << 5) | ((((i) >> 8) & 0xF) << 1)))
#define IMM_U(i)     ((int32_t)((i) & 0xFFFFF000))
#define IMM_J(i)     ((int32_t)((((i) >> 31) << 20) | ((((i) >> 12) & 0xFF) << 12) | \
                      ((((i) >> 20) & 1) << 11) | ((((i) >> 21) & 0x3FF) << 1)))
#define CSR_ADDR(i)  (((i) >> 20) & 0xFFF)

/* Opcodes */
#define OP_LUI    0x37
#define OP_AUIPC  0x17
#define OP_JAL    0x6F
#define OP_JALR   0x67
#define OP_BRANCH 0x63
#define OP_LOAD   0x03
#define OP_STORE  0x23
#define OP_ALUI   0x13  /* integer immediate */
#define OP_ALU    0x33  /* integer register */
#define OP_SYSTEM 0x73
#define OP_FP_LOAD  0x07
#define OP_FP_STORE 0x27
#define OP_FP     0x53  /* floating-point ops */
#define OP_FMADD  0x43
#define OP_FMSUB  0x47
#define OP_FNMSUB 0x4B
#define OP_FNMADD 0x4F

/* funct7 codes for OP_FP */
#define F7_FADD    0x00
#define F7_FSUB    0x04
#define F7_FMUL    0x08
#define F7_FDIV    0x0C
#define F7_FSQRT   0x2C
#define F7_FSGNJ   0x10
#define F7_FMINMAX 0x14
#define F7_FCVT_W_S  0x60  /* float->int: rs2 selects signed/unsigned */
#define F7_FCVT_S_W  0x68  /* int->float */
#define F7_FMV_W_X   0x78  /* fmv.w.x: move int->fp  (rs2=0) */
#define F7_FMV_X_W   0x70  /* fmv.x.w: move fp->int  (rs2=0) */
#define F7_FCMP      0x50  /* feq/flt/fle */
#define F7_FCLASS    0x70  /* fclass (rs2=0, funct3=1) - shares with FMV_X_W */

/* Low-precision float funct7 */
#define F7_FCVT_BF16  0x22   /* BF16: rs2=0 S->BF16, rs2=1 BF16->S */
#define F7_FCVT_FP8   0x24   /* FP8:  rs2=0 S->E4M3, rs2=1 E4M3->S, 2=S->E5M2, 3=E5M2->S */
#define F7_FCVT_FP4   0x26   /* FP4:  rs2=0 S->E2M1, rs2=1 E2M1->S */

static int exec_one_inst(GPGPUState *s, GPGPUWarp *warp)
{
    if (warp->active_mask == 0) {
        return 1;
    }

    /* All lanes share the same PC - fetch instruction from VRAM */
    uint32_t pc = warp->lanes[0].pc;
    /* Find the first active lane to get PC */
    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
        if (warp->lanes[i].active) {
            pc = warp->lanes[i].pc;
            break;
        }
    }

    uint32_t inst = vram_load_word(s, pc);
    uint32_t opcode = OPCODE(inst);
    uint32_t rd     = RD(inst);
    uint32_t rs1    = RS1(inst);
    uint32_t rs2    = RS2(inst);
    uint32_t funct3 = FUNCT3(inst);
    uint32_t funct7 = FUNCT7(inst);

    for (int li = 0; li < GPGPU_WARP_SIZE; li++) {
        GPGPULane *lane = &warp->lanes[li];
        if (!lane->active) {
            continue;
        }

        uint32_t next_pc = pc + 4;
        bool lane_done = false;

        switch (opcode) {

        /* ---- LUI ---- */
        case OP_LUI:
            if (rd) lane->gpr[rd] = (uint32_t)IMM_U(inst);
            break;

        /* ---- AUIPC ---- */
        case OP_AUIPC:
            if (rd) lane->gpr[rd] = pc + (uint32_t)IMM_U(inst);
            break;

        /* ---- JAL ---- */
        case OP_JAL: {
            int32_t off = IMM_J(inst);
            if (rd) lane->gpr[rd] = pc + 4;
            next_pc = (uint32_t)((int32_t)pc + off);
            break;
        }

        /* ---- JALR ---- */
        case OP_JALR: {
            uint32_t target = (lane->gpr[rs1] + (uint32_t)IMM_I(inst)) & ~1u;
            if (rd) lane->gpr[rd] = pc + 4;
            /* ret (jalr x0, 0(ra)) with ra=0 means stop */
            if (rs1 == 1 && rd == 0 && lane->gpr[1] == 0) {
                lane_done = true;
            }
            next_pc = target;
            break;
        }

        /* ---- BRANCH ---- */
        case OP_BRANCH: {
            int32_t off = IMM_B(inst);
            uint32_t a = lane->gpr[rs1];
            uint32_t b = lane->gpr[rs2];
            bool taken = false;
            switch (funct3) {
            case 0: taken = (a == b); break;
            case 1: taken = (a != b); break;
            case 4: taken = ((int32_t)a < (int32_t)b); break;
            case 5: taken = ((int32_t)a >= (int32_t)b); break;
            case 6: taken = (a < b); break;
            case 7: taken = (a >= b); break;
            default: break;
            }
            if (taken) {
                next_pc = (uint32_t)((int32_t)pc + off);
            }
            break;
        }

        /* ---- LOAD ---- */
        case OP_LOAD: {
            uint32_t addr = lane->gpr[rs1] + (uint32_t)IMM_I(inst);
            uint32_t val;
            switch (funct3) {
            case 0: val = (int32_t)(int8_t)mem_load(s, lane, addr, 1); break;
            case 1: val = (int32_t)(int16_t)mem_load(s, lane, addr, 2); break;
            case 2: val = mem_load(s, lane, addr, 4); break;
            case 4: val = mem_load(s, lane, addr, 1); break;
            case 5: val = mem_load(s, lane, addr, 2); break;
            default: val = 0; break;
            }
            if (rd) lane->gpr[rd] = val;
            break;
        }

        /* ---- STORE ---- */
        case OP_STORE: {
            uint32_t addr = lane->gpr[rs1] + (uint32_t)IMM_S(inst);
            uint32_t val  = lane->gpr[rs2];
            switch (funct3) {
            case 0: mem_store(s, addr, val & 0xFF, 1); break;
            case 1: mem_store(s, addr, val & 0xFFFF, 2); break;
            case 2: mem_store(s, addr, val, 4); break;
            default: break;
            }
            break;
        }

        /* ---- Integer immediate (ALUI) ---- */
        case OP_ALUI: {
            int32_t imm = IMM_I(inst);
            uint32_t a  = lane->gpr[rs1];
            uint32_t res = 0;
            switch (funct3) {
            case 0: res = a + (uint32_t)imm; break;           /* ADDI */
            case 1: res = a << (imm & 0x1F); break;           /* SLLI */
            case 2: res = (int32_t)a < imm ? 1 : 0; break;   /* SLTI */
            case 3: res = a < (uint32_t)imm ? 1 : 0; break;  /* SLTIU */
            case 4: res = a ^ (uint32_t)imm; break;           /* XORI */
            case 5:
                if (funct7 & 0x20) {
                    res = (uint32_t)((int32_t)a >> (imm & 0x1F));  /* SRAI */
                } else {
                    res = a >> (imm & 0x1F);                       /* SRLI */
                }
                break;
            case 6: res = a | (uint32_t)imm; break;           /* ORI */
            case 7: res = a & (uint32_t)imm; break;           /* ANDI */
            default: break;
            }
            if (rd) lane->gpr[rd] = res;
            break;
        }

        /* ---- Integer register (ALU) ---- */
        case OP_ALU: {
            uint32_t a = lane->gpr[rs1];
            uint32_t b = lane->gpr[rs2];
            uint32_t res = 0;
            if (funct7 == 0x01) {
                /* MUL/DIV extension */
                switch (funct3) {
                case 0: res = a * b; break;                              /* MUL */
                case 1: res = (uint32_t)(((int64_t)(int32_t)a * (int64_t)(int32_t)b) >> 32); break;
                case 3: res = (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32); break;
                case 4: res = (b == 0) ? -1 : (uint32_t)((int32_t)a / (int32_t)b); break;
                case 5: res = (b == 0) ? 0xFFFFFFFF : a / b; break;
                case 6: res = (b == 0) ? a : (uint32_t)((int32_t)a % (int32_t)b); break;
                case 7: res = (b == 0) ? a : a % b; break;
                default: break;
                }
            } else {
                switch (funct3) {
                case 0:
                    res = (funct7 & 0x20) ? (a - b) : (a + b);  /* ADD/SUB */
                    break;
                case 1: res = a << (b & 0x1F); break;            /* SLL */
                case 2: res = (int32_t)a < (int32_t)b ? 1 : 0; break; /* SLT */
                case 3: res = a < b ? 1 : 0; break;              /* SLTU */
                case 4: res = a ^ b; break;                      /* XOR */
                case 5:
                    if (funct7 & 0x20) {
                        res = (uint32_t)((int32_t)a >> (b & 0x1F)); /* SRA */
                    } else {
                        res = a >> (b & 0x1F);                   /* SRL */
                    }
                    break;
                case 6: res = a | b; break;                      /* OR */
                case 7: res = a & b; break;                      /* AND */
                default: break;
                }
            }
            if (rd) lane->gpr[rd] = res;
            break;
        }

        /* ---- SYSTEM (CSR, EBREAK) ---- */
        case OP_SYSTEM: {
            if (funct3 == 0) {
                /* ECALL/EBREAK */
                uint32_t imm = (inst >> 20) & 0xFFF;
                if (imm == 1) {  /* EBREAK */
                    lane_done = true;
                }
                /* ECALL: treat as stop */
                if (imm == 0) {
                    lane_done = true;
                }
            } else {
                /* CSR instructions */
                uint32_t csr  = CSR_ADDR(inst);
                uint32_t old_csr = 0;
                uint32_t rs1_val = lane->gpr[rs1];

                switch (csr) {
                case CSR_MHARTID: old_csr = lane->mhartid; break;
                case CSR_FFLAGS:  old_csr = lane->fcsr & 0x1F; break;
                case CSR_FRM:     old_csr = (lane->fcsr >> 5) & 0x7; break;
                case CSR_FCSR:    old_csr = lane->fcsr; break;
                default: old_csr = 0; break;
                }

                if (rd) lane->gpr[rd] = old_csr;

                uint32_t new_csr = old_csr;
                switch (funct3) {
                case 1: new_csr = rs1_val; break;                  /* CSRRW */
                case 2: new_csr = old_csr | rs1_val; break;        /* CSRRS */
                case 3: new_csr = old_csr & ~rs1_val; break;       /* CSRRC */
                case 5: new_csr = rs1; break;                      /* CSRRWI */
                case 6: new_csr = old_csr | rs1; break;            /* CSRRSI */
                case 7: new_csr = old_csr & ~(uint32_t)rs1; break; /* CSRRCI */
                default: break;
                }

                if (funct3 != 1 || rs1 != 0) { /* only write if not CSRRS x0 */
                    switch (csr) {
                    case CSR_FFLAGS:
                        lane->fcsr = (lane->fcsr & ~0x1F) | (new_csr & 0x1F);
                        break;
                    case CSR_FRM:
                        lane->fcsr = (lane->fcsr & ~0xE0) | ((new_csr & 0x7) << 5);
                        apply_rm((lane->fcsr >> 5) & 0x7, &lane->fp_status);
                        break;
                    case CSR_FCSR:
                        lane->fcsr = new_csr & 0xFF;
                        apply_rm((lane->fcsr >> 5) & 0x7, &lane->fp_status);
                        break;
                    default: break;
                    }
                }
            }
            break;
        }

        /* ---- FP LOAD (flw) ---- */
        case OP_FP_LOAD: {
            if (funct3 == 2) { /* FLW */
                uint32_t addr = lane->gpr[rs1] + (uint32_t)IMM_I(inst);
                if (rd < GPGPU_NUM_FREGS) {
                    lane->fpr[rd] = mem_load(s, lane, addr, 4);
                }
            }
            break;
        }

        /* ---- FP STORE (fsw) ---- */
        case OP_FP_STORE: {
            if (funct3 == 2) { /* FSW */
                uint32_t addr = lane->gpr[rs1] + (uint32_t)IMM_S(inst);
                mem_store(s, addr, lane->fpr[rs2], 4);
            }
            break;
        }

        /* ---- Floating-point ops ---- */
        case OP_FP: {
            unsigned rm = funct3;
            apply_rm(rm, &lane->fp_status);

            float32 fa = lane->fpr[rs1];
            float32 fb = lane->fpr[rs2];

            switch (funct7) {

            case F7_FADD:
                if (rd < GPGPU_NUM_FREGS)
                    lane->fpr[rd] = float32_add(fa, fb, &lane->fp_status);
                break;

            case F7_FSUB:
                if (rd < GPGPU_NUM_FREGS)
                    lane->fpr[rd] = float32_sub(fa, fb, &lane->fp_status);
                break;

            case F7_FMUL:
                if (rd < GPGPU_NUM_FREGS)
                    lane->fpr[rd] = float32_mul(fa, fb, &lane->fp_status);
                break;

            case F7_FDIV:
                if (rd < GPGPU_NUM_FREGS)
                    lane->fpr[rd] = float32_div(fa, fb, &lane->fp_status);
                break;

            case F7_FSQRT:
                if (rd < GPGPU_NUM_FREGS)
                    lane->fpr[rd] = float32_sqrt(fa, &lane->fp_status);
                break;

            case F7_FSGNJ:
                if (rd < GPGPU_NUM_FREGS) {
                    switch (funct3) {
                    case 0: /* FSGNJ */
                        lane->fpr[rd] = (fa & 0x7FFFFFFF) | (fb & 0x80000000);
                        break;
                    case 1: /* FSGNJN */
                        lane->fpr[rd] = (fa & 0x7FFFFFFF) | (~fb & 0x80000000);
                        break;
                    case 2: /* FSGNJX */
                        lane->fpr[rd] = fa ^ (fb & 0x80000000);
                        break;
                    default: break;
                    }
                }
                break;

            case F7_FMINMAX:
                if (rd < GPGPU_NUM_FREGS) {
                    if (funct3 == 0)
                        lane->fpr[rd] = float32_min(fa, fb, &lane->fp_status);
                    else
                        lane->fpr[rd] = float32_max(fa, fb, &lane->fp_status);
                }
                break;

            case F7_FCMP:
                if (rd < GPGPU_NUM_REGS) {
                    FloatRelation rel = float32_compare(fa, fb, &lane->fp_status);
                    switch (funct3) {
                    case 2: lane->gpr[rd] = (rel == float_relation_equal) ? 1 : 0; break;
                    case 1: lane->gpr[rd] = (rel == float_relation_less) ? 1 : 0; break;
                    case 0: lane->gpr[rd] = (rel <= float_relation_equal) ? 1 : 0; break;
                    default: lane->gpr[rd] = 0; break;
                    }
                }
                break;

            case F7_FCVT_W_S:
                /* fcvt.w.s  rd, fs1   (rs2=0, signed)
                   fcvt.wu.s rd, fs1   (rs2=1, unsigned) */
                apply_rm(funct3, &lane->fp_status);
                if (rd < GPGPU_NUM_REGS) {
                    if (rs2 == 0) {
                        lane->gpr[rd] = (uint32_t)float32_to_int32(fa, &lane->fp_status);
                    } else {
                        lane->gpr[rd] = float32_to_uint32(fa, &lane->fp_status);
                    }
                }
                break;

            case F7_FCVT_S_W:
                /* fcvt.s.w  fd, rs1   (rs2=0, signed)
                   fcvt.s.wu fd, rs1   (rs2=1, unsigned) */
                if (rd < GPGPU_NUM_FREGS) {
                    if (rs2 == 0) {
                        lane->fpr[rd] = int32_to_float32((int32_t)lane->gpr[rs1],
                                                         &lane->fp_status);
                    } else {
                        lane->fpr[rd] = uint32_to_float32(lane->gpr[rs1],
                                                          &lane->fp_status);
                    }
                }
                break;

            case F7_FMV_W_X:
                /* fmv.w.x fd, rs1  (rs2=0) */
                if (rd < GPGPU_NUM_FREGS && rs2 == 0) {
                    lane->fpr[rd] = lane->gpr[rs1];
                }
                break;

            case F7_FMV_X_W:
                /* fmv.x.w rd, fs1  (rs2=0, funct3=0)
                   fclass   rd, fs1  (rs2=0, funct3=1) */
                if (rd < GPGPU_NUM_REGS && rs2 == 0 && funct3 == 0) {
                    lane->gpr[rd] = lane->fpr[rs1];
                }
                break;

            /* ---- Low-precision float: BF16 (funct7=0x22) ---- */
            case F7_FCVT_BF16: {
                if (rs2 == 0) {
                    /* fcvt.s.bf16  fd, fs1  (bf16 in fs1[15:0] -> f32) */
                    bfloat16 bf = (bfloat16)(fa & 0xFFFF);
                    if (rd < GPGPU_NUM_FREGS)
                        lane->fpr[rd] = bfloat16_to_float32(bf, &lane->fp_status);
                } else if (rs2 == 1) {
                    /* fcvt.bf16.s  fd, fs1  (f32 -> bf16 stored in f32 reg) */
                    bfloat16 bf = float32_to_bfloat16(fa, &lane->fp_status);
                    if (rd < GPGPU_NUM_FREGS)
                        lane->fpr[rd] = (uint32_t)bf;  /* zero-extend */
                }
                break;
            }

            /* ---- Low-precision float: FP8 (funct7=0x24) ---- */
            case F7_FCVT_FP8: {
                if (rs2 == 0) {
                    /* fcvt.s.e4m3  fd, fs1  (E4M3 -> f32) */
                    float8_e4m3 fp8 = (float8_e4m3)(fa & 0xFF);
                    if (rd < GPGPU_NUM_FREGS)
                        lane->fpr[rd] = fp8_e4m3_to_f32(fp8, &lane->fp_status);
                } else if (rs2 == 1) {
                    /* fcvt.e4m3.s  fd, fs1  (f32 -> E4M3) */
                    float8_e4m3 fp8 = f32_to_fp8_e4m3_sat(fa, &lane->fp_status);
                    if (rd < GPGPU_NUM_FREGS)
                        lane->fpr[rd] = (uint32_t)fp8;
                } else if (rs2 == 2) {
                    /* fcvt.s.e5m2  fd, fs1  (E5M2 -> f32) */
                    float8_e5m2 fp8 = (float8_e5m2)(fa & 0xFF);
                    if (rd < GPGPU_NUM_FREGS)
                        lane->fpr[rd] = fp8_e5m2_to_f32(fp8, &lane->fp_status);
                } else if (rs2 == 3) {
                    /* fcvt.e5m2.s  fd, fs1  (f32 -> E5M2) */
                    float8_e5m2 fp8 = f32_to_fp8_e5m2_sat(fa, &lane->fp_status);
                    if (rd < GPGPU_NUM_FREGS)
                        lane->fpr[rd] = (uint32_t)fp8;
                }
                break;
            }

            /* ---- Low-precision float: FP4 E2M1 (funct7=0x26) ---- */
            case F7_FCVT_FP4: {
                if (rs2 == 0) {
                    /* fcvt.e2m1.s  fd, fs1 */
                    float4_e2m1 fp4 = f32_to_fp4_e2m1(fa);
                    if (rd < GPGPU_NUM_FREGS)
                        lane->fpr[rd] = (uint32_t)fp4;
                } else if (rs2 == 1) {
                    /* fcvt.s.e2m1  fd, fs1 */
                    float4_e2m1 fp4 = (float4_e2m1)(fa & 0x0F);
                    if (rd < GPGPU_NUM_FREGS)
                        lane->fpr[rd] = fp4_e2m1_to_f32(fp4);
                }
                break;
            }

            default:
                qemu_log_mask(LOG_UNIMP,
                              "gpgpu: unimplemented FP funct7=0x%02x at pc=0x%x\n",
                              funct7, pc);
                break;
            }
            break; /* end OP_FP */
        }

        /* ---- FMADD/FMSUB/FNMSUB/FNMADD ---- */
        case OP_FMADD: {
            uint32_t rs3  = (inst >> 27) & 0x1F;
            float32 fma_a = lane->fpr[rs1];
            float32 fma_b = lane->fpr[rs2];
            float32 fma_c = lane->fpr[rs3];
            apply_rm(funct3, &lane->fp_status);
            if (rd < GPGPU_NUM_FREGS)
                lane->fpr[rd] = float32_muladd(fma_a, fma_b, fma_c,
                                               0, &lane->fp_status);
            break;
        }
        case OP_FMSUB: {
            uint32_t rs3  = (inst >> 27) & 0x1F;
            float32 fma_a = lane->fpr[rs1];
            float32 fma_b = lane->fpr[rs2];
            float32 fma_c = lane->fpr[rs3];
            apply_rm(funct3, &lane->fp_status);
            if (rd < GPGPU_NUM_FREGS)
                lane->fpr[rd] = float32_muladd(fma_a, fma_b, fma_c,
                                               float_muladd_negate_c,
                                               &lane->fp_status);
            break;
        }
        case OP_FNMSUB: {
            uint32_t rs3  = (inst >> 27) & 0x1F;
            float32 fma_a = lane->fpr[rs1];
            float32 fma_b = lane->fpr[rs2];
            float32 fma_c = lane->fpr[rs3];
            apply_rm(funct3, &lane->fp_status);
            if (rd < GPGPU_NUM_FREGS)
                lane->fpr[rd] = float32_muladd(fma_a, fma_b, fma_c,
                                               float_muladd_negate_product,
                                               &lane->fp_status);
            break;
        }
        case OP_FNMADD: {
            uint32_t rs3  = (inst >> 27) & 0x1F;
            float32 fma_a = lane->fpr[rs1];
            float32 fma_b = lane->fpr[rs2];
            float32 fma_c = lane->fpr[rs3];
            apply_rm(funct3, &lane->fp_status);
            if (rd < GPGPU_NUM_FREGS)
                lane->fpr[rd] = float32_muladd(fma_a, fma_b, fma_c,
                                               float_muladd_negate_c |
                                               float_muladd_negate_product,
                                               &lane->fp_status);
            break;
        }

        default:
            qemu_log_mask(LOG_UNIMP,
                          "gpgpu: unimplemented opcode 0x%02x at pc=0x%x\n",
                          opcode, pc);
            break;
        }

        /* x0 is always 0 */
        lane->gpr[0] = 0;

        if (lane_done) {
            lane->active = false;
            warp->active_mask &= ~(1u << li);
        } else {
            lane->pc = next_pc;
        }
    } /* for each lane */

    return (warp->active_mask == 0) ? 1 : 0;
}

/* ============================================================
 * Execute one warp until all lanes complete or max_cycles
 * ============================================================ */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    for (uint32_t cycle = 0; cycle < max_cycles; cycle++) {
        int ret = exec_one_inst(s, warp);
        if (ret != 0) {
            return (ret == 1) ? 0 : ret;
        }
    }
    qemu_log_mask(LOG_GUEST_ERROR,
                  "gpgpu: warp exceeded max_cycles=%u\n", max_cycles);
    return -1;
}

/* ============================================================
 * Execute full kernel
 * ============================================================ */
int gpgpu_core_exec_kernel(GPGPUState *s)
{
    GPGPUKernelParams *k = &s->kernel;
    uint32_t grid_x = k->grid_dim[0];
    uint32_t grid_y = k->grid_dim[1];
    uint32_t grid_z = k->grid_dim[2];
    uint32_t blk_x  = k->block_dim[0];
    uint32_t blk_y  = k->block_dim[1];
    uint32_t blk_z  = k->block_dim[2];
    uint32_t pc     = (uint32_t)k->kernel_addr;
    uint32_t max_cyc = 100000;

    uint32_t threads_per_block = blk_x * blk_y * blk_z;
    uint32_t warps_per_block   = (threads_per_block + GPGPU_WARP_SIZE - 1)
                                  / GPGPU_WARP_SIZE;

    for (uint32_t bz = 0; bz < grid_z; bz++) {
        for (uint32_t by = 0; by < grid_y; by++) {
            for (uint32_t bx = 0; bx < grid_x; bx++) {
                uint32_t block_id[3] = { bx, by, bz };
                uint32_t block_id_linear = bz * grid_y * grid_x
                                         + by * grid_x + bx;

                for (uint32_t wi = 0; wi < warps_per_block; wi++) {
                    uint32_t tid_base = wi * GPGPU_WARP_SIZE;
                    uint32_t remaining = threads_per_block - tid_base;
                    uint32_t num_threads = (remaining > GPGPU_WARP_SIZE)
                                          ? GPGPU_WARP_SIZE : remaining;

                    GPGPUWarp warp;
                    gpgpu_core_init_warp(&warp, pc, tid_base,
                                        block_id, num_threads,
                                        wi, block_id_linear);

                    int ret = gpgpu_core_exec_warp(s, &warp, max_cyc);
                    if (ret != 0) {
                        s->error_status |= GPGPU_ERR_KERNEL_FAULT;
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}
