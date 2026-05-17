/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"
#include "gpgpu_core.h"

/*
 * gpgpu_ctrl_read - BAR0 MMIO 读处理函数
 *
 * 当宿主机软件（测题或驱动）通过 PCI MMIO 读取 BAR0 中的某个偏移时，
 * QEMU 会调用此函数。addr 是相对于 BAR0 基址的偏移，size 固定为 4（32 位）。
 *
 * 地址映射（详见硬件手册 §4.2 和 §5~§11）：
 *   0x0000–0x00FF  设备信息寄存器（只读）
 *   0x0100–0x01FF  全局控制/状态寄存器
 *   0x0200–0x02FF  中断控制寄存器
 *   0x0300–0x03FF  内核分发寄存器
 *   0x0400–0x04FF  DMA 引擎寄存器
 *   0x1000–0x1FFF  SIMT 线程上下文寄存器
 *   0x2000–0x2FFF  同步寄存器
 */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque;

    switch (addr) {
    /* ---- 5. 设备信息寄存器 (0x0000–0x00FF, 只读) ---- */
    case GPGPU_REG_DEV_ID:
        /* 返回固定的 ASCII 标识 "GPPU" = 0x47505055，硬件手册 §5 */
        return GPGPU_DEV_ID_VALUE;

    case GPGPU_REG_DEV_VERSION:
        /* 返回版本号 v1.0.0 = 0x00010000，硬件手册 §5 */
        return GPGPU_DEV_VERSION_VALUE;

    case GPGPU_REG_DEV_CAPS:
        /*
         * 能力寄存器：将三个配置参数打包进 32 位
         *   bit  7:0  - NUM_CUS      (计算单元数量)
         *   bit 15:8  - WARPS_PER_CU (每个 CU 的 warp 数)
         *   bit 23:16 - WARP_SIZE    (每个 warp 的线程数)
         *   bit 31:24 - Reserved (0)
         * 硬件手册 §5.1
         */
        return (s->num_cus & 0xFF) |
               ((s->warps_per_cu & 0xFF) << 8) |
               ((s->warp_size & 0xFF) << 16);

    case GPGPU_REG_VRAM_SIZE_LO:
        /* VRAM 大小的低 32 位，硬件手册 §5 */
        return (uint32_t)(s->vram_size & 0xFFFFFFFF);

    case GPGPU_REG_VRAM_SIZE_HI:
        /* VRAM 大小的高 32 位（默认 64MB 时为 0），硬件手册 §5 */
        return (uint32_t)(s->vram_size >> 32);

    /* ---- 6. 全局控制/状态寄存器 (0x0100–0x01FF) ---- */
    case GPGPU_REG_GLOBAL_CTRL:
        /* 返回当前全局控制寄存器值，硬件手册 §6.2 */
        return s->global_ctrl;

    case GPGPU_REG_GLOBAL_STATUS:
        /* 返回设备状态（READY/BUSY/ERROR），硬件手册 §6.3 */
        return s->global_status;

    case GPGPU_REG_ERROR_STATUS:
        /* 返回错误状态，写 1 清除（W1C），硬件手册 §6.4 */
        return s->error_status;

    /* ---- 7. 中断控制寄存器 (0x0200–0x02FF) ---- */
    case GPGPU_REG_IRQ_ENABLE:
        /* 返回中断使能掩码，硬件手册 §7 */
        return s->irq_enable;

    case GPGPU_REG_IRQ_STATUS:
        /* 返回当前挂起的中断状态，硬件手册 §7 */
        return s->irq_status;

    case GPGPU_REG_IRQ_ACK:
        /* IRQ_ACK 是只写寄存器，读取返回 0 */
        return 0;

    /* ---- 8. 内核分发寄存器 (0x0300–0x03FF) ---- */
    case GPGPU_REG_KERNEL_ADDR_LO:
        return (uint32_t)(s->kernel.kernel_addr & 0xFFFFFFFF);

    case GPGPU_REG_KERNEL_ADDR_HI:
        return (uint32_t)(s->kernel.kernel_addr >> 32);

    case GPGPU_REG_KERNEL_ARGS_LO:
        return (uint32_t)(s->kernel.kernel_args & 0xFFFFFFFF);

    case GPGPU_REG_KERNEL_ARGS_HI:
        return (uint32_t)(s->kernel.kernel_args >> 32);

    case GPGPU_REG_GRID_DIM_X:
        return s->kernel.grid_dim[0];

    case GPGPU_REG_GRID_DIM_Y:
        return s->kernel.grid_dim[1];

    case GPGPU_REG_GRID_DIM_Z:
        return s->kernel.grid_dim[2];

    case GPGPU_REG_BLOCK_DIM_X:
        return s->kernel.block_dim[0];

    case GPGPU_REG_BLOCK_DIM_Y:
        return s->kernel.block_dim[1];

    case GPGPU_REG_BLOCK_DIM_Z:
        return s->kernel.block_dim[2];

    case GPGPU_REG_SHARED_MEM_SIZE:
        return s->kernel.shared_mem_size;

    case GPGPU_REG_DISPATCH:
        /* DISPATCH 是只写寄存器，读取返回 0 */
        return 0;

    /* ---- 9. DMA 引擎寄存器 (0x0400–0x04FF) ---- */
    case GPGPU_REG_DMA_SRC_LO:
        return (uint32_t)(s->dma.src_addr & 0xFFFFFFFF);

    case GPGPU_REG_DMA_SRC_HI:
        return (uint32_t)(s->dma.src_addr >> 32);

    case GPGPU_REG_DMA_DST_LO:
        return (uint32_t)(s->dma.dst_addr & 0xFFFFFFFF);

    case GPGPU_REG_DMA_DST_HI:
        return (uint32_t)(s->dma.dst_addr >> 32);

    case GPGPU_REG_DMA_SIZE:
        return s->dma.size;

    case GPGPU_REG_DMA_CTRL:
        return s->dma.ctrl;

    case GPGPU_REG_DMA_STATUS:
        return s->dma.status;

    /* ---- 10. SIMT 线程上下文寄存器 (0x1000–0x1FFF) ---- */
    case GPGPU_REG_THREAD_ID_X:
        return s->simt.thread_id[0];

    case GPGPU_REG_THREAD_ID_Y:
        return s->simt.thread_id[1];

    case GPGPU_REG_THREAD_ID_Z:
        return s->simt.thread_id[2];

    case GPGPU_REG_BLOCK_ID_X:
        return s->simt.block_id[0];

    case GPGPU_REG_BLOCK_ID_Y:
        return s->simt.block_id[1];

    case GPGPU_REG_BLOCK_ID_Z:
        return s->simt.block_id[2];

    case GPGPU_REG_WARP_ID:
        return s->simt.warp_id;

    case GPGPU_REG_LANE_ID:
        return s->simt.lane_id;

    /* ---- 11. 同步寄存器 (0x2000–0x2FFF) ---- */
    case GPGPU_REG_BARRIER:
        /* BARRIER 是只写寄存器，读取返回 0 */
        return 0;

    case GPGPU_REG_THREAD_MASK:
        return s->simt.thread_mask;

    default:
        qemu_log_mask(LOG_UNIMP,
                      "gpgpu: ctrl_read: unimplemented register at 0x%"
                      HWADDR_PRIx "\n", addr);
        return 0;
    }
}

/*
 * gpgpu_ctrl_write - BAR0 MMIO 写处理函数
 *
 * 当宿主机软件通过 PCI MMIO 写入 BAR0 中的某个偏移时，QEMU 调用此函数。
 * addr 是相对于 BAR0 基址的偏移，val 是写入值，size 固定为 4（32 位）。
 *
 * 注意：
 *  - 设备信息寄存器（0x0000–0x00FF）是只读的，写入被忽略
 *  - GLOBAL_CTRL.RESET 写 1 后自动清零，并触发软复位
 *  - ERROR_STATUS 和 IRQ_ACK 是写 1 清除（W1C）寄存器
 */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;

    switch (addr) {
    /* ---- 设备信息寄存器（只读）：忽略写入 ---- */
    case GPGPU_REG_DEV_ID:
    case GPGPU_REG_DEV_VERSION:
    case GPGPU_REG_DEV_CAPS:
    case GPGPU_REG_VRAM_SIZE_LO:
    case GPGPU_REG_VRAM_SIZE_HI:
        /* 只读，静默丢弃 */
        break;

    /* ---- 全局控制寄存器 ---- */
    case GPGPU_REG_GLOBAL_CTRL:
        /*
         * 检查 RESET 位（bit 1）：写 1 触发软复位后自动清零
         * 软复位清除所有控制/状态/SIMT 上下文，但不清除 VRAM
         * 硬件手册 §6.5
         */
        if (val & GPGPU_CTRL_RESET) {
            /* 触发软复位：清零所有状态，恢复到就绪状态 */
            s->global_ctrl = 0;
            s->global_status = GPGPU_STATUS_READY;
            s->error_status = 0;
            s->irq_enable = 0;
            s->irq_status = 0;
            memset(&s->kernel, 0, sizeof(s->kernel));
            memset(&s->dma, 0, sizeof(s->dma));
            memset(&s->simt, 0, sizeof(s->simt));
            timer_del(s->dma_timer);
            timer_del(s->kernel_timer);
            /* RESET 位不写回，实现自动清零 */
        } else {
            /* 保存控制寄存器值（ENABLE 等非 RESET 位） */
            s->global_ctrl = (uint32_t)val;
        }
        break;

    case GPGPU_REG_ERROR_STATUS:
        /*
         * 写 1 清除（W1C）：写入的位为 1 时清除对应错误标志
         * 硬件手册 §6.4
         */
        s->error_status &= ~(uint32_t)val;
        break;

    case GPGPU_REG_GLOBAL_STATUS:
        /* 只读，静默丢弃 */
        break;

    /* ---- 中断控制寄存器 ---- */
    case GPGPU_REG_IRQ_ENABLE:
        s->irq_enable = (uint32_t)val;
        break;

    case GPGPU_REG_IRQ_STATUS:
        /* 只读，静默丢弃 */
        break;

    case GPGPU_REG_IRQ_ACK:
        /* 写 1 清除挂起的中断状态，硬件手册 §7 */
        s->irq_status &= ~(uint32_t)val;
        break;

    /* ---- 内核分发寄存器 ---- */
    case GPGPU_REG_KERNEL_ADDR_LO:
        s->kernel.kernel_addr =
            (s->kernel.kernel_addr & 0xFFFFFFFF00000000ULL) | (uint32_t)val;
        break;

    case GPGPU_REG_KERNEL_ADDR_HI:
        s->kernel.kernel_addr =
            (s->kernel.kernel_addr & 0x00000000FFFFFFFFULL) |
            ((uint64_t)(uint32_t)val << 32);
        break;

    case GPGPU_REG_KERNEL_ARGS_LO:
        s->kernel.kernel_args =
            (s->kernel.kernel_args & 0xFFFFFFFF00000000ULL) | (uint32_t)val;
        break;

    case GPGPU_REG_KERNEL_ARGS_HI:
        s->kernel.kernel_args =
            (s->kernel.kernel_args & 0x00000000FFFFFFFFULL) |
            ((uint64_t)(uint32_t)val << 32);
        break;

    case GPGPU_REG_GRID_DIM_X:
        s->kernel.grid_dim[0] = (uint32_t)val;
        break;

    case GPGPU_REG_GRID_DIM_Y:
        s->kernel.grid_dim[1] = (uint32_t)val;
        break;

    case GPGPU_REG_GRID_DIM_Z:
        s->kernel.grid_dim[2] = (uint32_t)val;
        break;

    case GPGPU_REG_BLOCK_DIM_X:
        s->kernel.block_dim[0] = (uint32_t)val;
        break;

    case GPGPU_REG_BLOCK_DIM_Y:
        s->kernel.block_dim[1] = (uint32_t)val;
        break;

    case GPGPU_REG_BLOCK_DIM_Z:
        s->kernel.block_dim[2] = (uint32_t)val;
        break;

    case GPGPU_REG_SHARED_MEM_SIZE:
        s->kernel.shared_mem_size = (uint32_t)val;
        break;

    case GPGPU_REG_DISPATCH:
        /*
         * 写任意值触发内核执行，硬件手册 §8.2
         * 触发条件：设备已使能 && 不忙 && Grid/Block 维度均 > 0 && 地址有效
         * 若条件不满足，设置 INVALID_CMD 错误
         */
        if (!(s->global_ctrl & GPGPU_CTRL_ENABLE)) {
            /* 设备未使能，写 DISPATCH 无效 */
            s->error_status |= GPGPU_ERR_INVALID_CMD;
        } else if (s->global_status & GPGPU_STATUS_BUSY) {
            /* 设备正忙，写 DISPATCH 无效 */
            s->error_status |= GPGPU_ERR_INVALID_CMD;
        } else if (!s->kernel.grid_dim[0] || !s->kernel.grid_dim[1] ||
                   !s->kernel.grid_dim[2] || !s->kernel.block_dim[0] ||
                   !s->kernel.block_dim[1] || !s->kernel.block_dim[2]) {
            /* Grid/Block 维度为 0，写 DISPATCH 无效 */
            s->error_status |= GPGPU_ERR_INVALID_CMD;
        } else {
            /* 启动内核执行引擎 */
            s->global_status |= GPGPU_STATUS_BUSY;
            int ret = gpgpu_core_exec_kernel(s);
            s->global_status &= ~GPGPU_STATUS_BUSY;
            if (ret != 0) {
                s->global_status |= GPGPU_STATUS_ERROR;
            }
            s->global_status |= GPGPU_STATUS_READY;
            /* 触发内核完成中断（如果已使能） */
            if (s->irq_enable & GPGPU_IRQ_KERNEL_DONE) {
                s->irq_status |= GPGPU_IRQ_KERNEL_DONE;
            }
        }
        break;

    /* ---- DMA 引擎寄存器 ---- */
    case GPGPU_REG_DMA_SRC_LO:
        s->dma.src_addr =
            (s->dma.src_addr & 0xFFFFFFFF00000000ULL) | (uint32_t)val;
        break;

    case GPGPU_REG_DMA_SRC_HI:
        s->dma.src_addr =
            (s->dma.src_addr & 0x00000000FFFFFFFFULL) |
            ((uint64_t)(uint32_t)val << 32);
        break;

    case GPGPU_REG_DMA_DST_LO:
        s->dma.dst_addr =
            (s->dma.dst_addr & 0xFFFFFFFF00000000ULL) | (uint32_t)val;
        break;

    case GPGPU_REG_DMA_DST_HI:
        s->dma.dst_addr =
            (s->dma.dst_addr & 0x00000000FFFFFFFFULL) |
            ((uint64_t)(uint32_t)val << 32);
        break;

    case GPGPU_REG_DMA_SIZE:
        s->dma.size = (uint32_t)val;
        break;

    case GPGPU_REG_DMA_CTRL:
        s->dma.ctrl = (uint32_t)val;
        if (val & GPGPU_DMA_START) {
            uint64_t src  = s->dma.src_addr;
            uint64_t dst  = s->dma.dst_addr;
            uint32_t sz   = s->dma.size;

            s->dma.status = GPGPU_DMA_BUSY;
            /* VRAM 内部搬运 */
            if (src + sz <= s->vram_size && dst + sz <= s->vram_size && sz > 0) {
                memmove(s->vram_ptr + dst, s->vram_ptr + src, sz);
                s->dma.status = GPGPU_DMA_COMPLETE;
                if (s->irq_enable & GPGPU_IRQ_DMA_DONE) {
                    s->irq_status |= GPGPU_IRQ_DMA_DONE;
                }
            } else {
                s->dma.status = GPGPU_DMA_ERROR;
                s->error_status |= GPGPU_ERR_DMA_FAULT;
                if (s->irq_enable & GPGPU_IRQ_ERROR) {
                    s->irq_status |= GPGPU_IRQ_ERROR;
                }
            }
        }
        break;

    case GPGPU_REG_DMA_STATUS:
        /* 只读，静默丢弃 */
        break;

    /* ---- SIMT 线程上下文寄存器 ---- */
    case GPGPU_REG_THREAD_ID_X:
        s->simt.thread_id[0] = (uint32_t)val;
        break;

    case GPGPU_REG_THREAD_ID_Y:
        s->simt.thread_id[1] = (uint32_t)val;
        break;

    case GPGPU_REG_THREAD_ID_Z:
        s->simt.thread_id[2] = (uint32_t)val;
        break;

    case GPGPU_REG_BLOCK_ID_X:
        s->simt.block_id[0] = (uint32_t)val;
        break;

    case GPGPU_REG_BLOCK_ID_Y:
        s->simt.block_id[1] = (uint32_t)val;
        break;

    case GPGPU_REG_BLOCK_ID_Z:
        s->simt.block_id[2] = (uint32_t)val;
        break;

    case GPGPU_REG_WARP_ID:
        s->simt.warp_id = (uint32_t)val;
        break;

    case GPGPU_REG_LANE_ID:
        s->simt.lane_id = (uint32_t)val;
        break;

    /* ---- 同步寄存器 ---- */
    case GPGPU_REG_BARRIER:
        /* 写任意值发出 Barrier 信号：计数加一，达到 target 时重置 */
        s->simt.barrier_count++;
        if (s->simt.barrier_target > 0 &&
            s->simt.barrier_count >= s->simt.barrier_target) {
            s->simt.barrier_count = 0;
            s->simt.barrier_active = false;
        } else {
            s->simt.barrier_active = true;
        }
        break;

    case GPGPU_REG_THREAD_MASK:
        s->simt.thread_mask = (uint32_t)val;
        break;

    default:
        qemu_log_mask(LOG_UNIMP,
                      "gpgpu: ctrl_write: unimplemented register at 0x%"
                      HWADDR_PRIx " val=0x%" PRIx64 "\n", addr, val);
        break;
    }
}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/*
 * gpgpu_vram_read - BAR2 VRAM 读处理函数
 *
 * VRAM 是一块连续的物理内存（由 vram_ptr 指向），GPU 代码和数据都存储在这里。
 * 支持 1~8 字节的访问（由 MemoryRegionOps.impl 配置）。
 * addr 是相对于 BAR2 基址的偏移，size 是本次访问的字节数。
 */
static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *s = opaque;
    uint64_t val = 0;

    /* 越界检测：addr + size 超出 vram_size 时返回 0 并记录错误 */
    if (addr + size > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: vram_read: out of bounds addr=0x%" HWADDR_PRIx
                      " size=%u vram_size=0x%" PRIx64 "\n",
                      addr, size, s->vram_size);
        return 0;
    }

    /*
     * 将 vram_ptr（uint8_t*）中的字节按照访问大小组装成整数返回。
     * 使用 memcpy 避免对齐问题（RISC-V 要求地址对齐，但宿主机侧不强制）。
     */
    memcpy(&val, s->vram_ptr + addr, size);
    return val;
}

/*
 * gpgpu_vram_write - BAR2 VRAM 写处理函数
 *
 * 将 val 的低 size 字节写入 vram_ptr[addr]。
 * 对越界写入记录日志并丢弃。
 */
static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *s = opaque;

    /* 越界检测 */
    if (addr + size > s->vram_size) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "gpgpu: vram_write: out of bounds addr=0x%" HWADDR_PRIx
                      " size=%u vram_size=0x%" PRIx64 "\n",
                      addr, size, s->vram_size);
        return;
    }

    /* 将 val 的低 size 字节写入 VRAM 缓冲区 */
    memcpy(s->vram_ptr + addr, &val, size);
}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)size;
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)val;
    (void)size;
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

static void gpgpu_dma_complete(void *opaque)
{
    GPGPUState *s = opaque;
    s->dma.status = GPGPU_DMA_COMPLETE;
    if (s->irq_enable & GPGPU_IRQ_DMA_DONE) {
        s->irq_status |= GPGPU_IRQ_DMA_DONE;
    }
}

static void gpgpu_kernel_complete(void *opaque)
{
    GPGPUState *s = opaque;
    s->global_status &= ~GPGPU_STATUS_BUSY;
    s->global_status |= GPGPU_STATUS_READY;
    if (s->irq_enable & GPGPU_IRQ_KERNEL_DONE) {
        s->irq_status |= GPGPU_IRQ_KERNEL_DONE;
    }
}

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr) {
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }

    /* BAR 0: control registers */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: VRAM */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_TYPE_64 |
                     PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: doorbell registers */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0, 0xFE000,
                  &s->ctrl_mmio, 0, 0xFF000,
                  0, errp)) {
        g_free(s->vram_ptr);
        return;
    }

    msi_init(pdev, 0, 1, true, false, errp);

    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);
    s->kernel_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,
                                   gpgpu_kernel_complete, s);

    s->global_status = GPGPU_STATUS_READY;
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    timer_free(s->dma_timer);
    timer_free(s->kernel_timer);
    g_free(s->vram_ptr);
    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));
    memset(&s->simt, 0, sizeof(s->simt));
    timer_del(s->dma_timer);
    timer_del(s->kernel_timer);
    if (s->vram_ptr) {
        memset(s->vram_ptr, 0, s->vram_size);
    }
}

static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()
    }
};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name          = TYPE_GPGPU,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init    = gpgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { }
    },
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)
