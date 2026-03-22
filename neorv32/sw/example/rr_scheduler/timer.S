# ──────────────────────────────────────────────────────────────
# timer.S  –  timer init + interrupt handler (preemptive layer)
#
# The trap_vector saves only CALLER-SAVED registers as an
# "interrupt frame" (18 words = 72 bytes), then calls into C
# which invokes the same context_switch used by yield().
#
# Stack layout for a preempted task:
#
#   [task stack data …]
#   [interrupt frame: ra,t0-t6,a0-a7,mepc,mstatus = 72 bytes]
#   [cooperative frame: ra,s0-s11 = 52 bytes]  ← tasks[i].sp
#
# Stack layout for a cooperatively-yielded task:
#
#   [task stack data …]
#   [cooperative frame: ra,s0-s11 = 52 bytes]  ← tasks[i].sp
#
# In both cases tasks[i].sp always points to a cooperative
# frame, so context_switch never needs to know the difference.
# ──────────────────────────────────────────────────────────────

.global trap_vector
.global timer_init

# CLINT addresses for QEMU virt machine (rv32, hart 0)
.equ CLINT_BASE,     0xFFF40000           # was 0x02000000
.equ MTIMECMP_LO,    CLINT_BASE + 0x4000
.equ MTIMECMP_HI,    CLINT_BASE + 0x4004
.equ MTIME_LO,       CLINT_BASE + 0xBFF8
.equ TIMER_INTERVAL, 90000000         # ~1 s at 10 MHz

# ── Interrupt-frame offsets (caller-saved + CSRs) ─────────────
# 18 words = 72 bytes, padded to 80 for 16-byte alignment
.equ IF_RA,       0
.equ IF_T0,       4
.equ IF_T1,       8
.equ IF_T2,      12
.equ IF_T3,      16
.equ IF_T4,      20
.equ IF_T5,      24
.equ IF_T6,      28
.equ IF_A0,      32
.equ IF_A1,      36
.equ IF_A2,      40
.equ IF_A3,      44
.equ IF_A4,      48
.equ IF_A5,      52
.equ IF_A6,      56
.equ IF_A7,      60
.equ IF_MEPC,    64
.equ IF_MSTATUS, 68
.equ IF_SIZE,    80          # 72 rounded up to 16-byte alignment


# ──────────────────────────────────────────────────────────────
# timer_init  –  arm the first timer deadline
# ──────────────────────────────────────────────────────────────
timer_init:
    li   t0, MTIME_LO
    lw   t1, 0(t0)
    li   t2, TIMER_INTERVAL
    add  t1, t1, t2

    li   t0, MTIMECMP_HI
    sw   zero, 0(t0)            # hi = 0 first (avoids spurious)
    li   t0, MTIMECMP_LO
    sw   t1, 0(t0)
    ret


# ──────────────────────────────────────────────────────────────
# trap_vector  –  machine-mode trap entry
# ──────────────────────────────────────────────────────────────
.align 4
trap_vector:
    # ── save interrupt frame (caller-saved only) ──────────────
    addi sp, sp, -IF_SIZE

    sw   ra,  IF_RA(sp)
    sw   t0,  IF_T0(sp)
    sw   t1,  IF_T1(sp)
    sw   t2,  IF_T2(sp)
    sw   t3,  IF_T3(sp)
    sw   t4,  IF_T4(sp)
    sw   t5,  IF_T5(sp)
    sw   t6,  IF_T6(sp)
    sw   a0,  IF_A0(sp)
    sw   a1,  IF_A1(sp)
    sw   a2,  IF_A2(sp)
    sw   a3,  IF_A3(sp)
    sw   a4,  IF_A4(sp)
    sw   a5,  IF_A5(sp)
    sw   a6,  IF_A6(sp)
    sw   a7,  IF_A7(sp)

    csrr t0, mepc
    sw   t0, IF_MEPC(sp)
    csrr t0, mstatus
    sw   t0, IF_MSTATUS(sp)

    # ── identify cause ────────────────────────────────────────
    csrr t0, mcause

    li   t1, 0x80000007        # machine timer interrupt
    beq  t0, t1, .handle_timer

    j    .irq_restore          # unknown – return without switching

.handle_timer:
    # Acknowledge: mtimecmp = mtime + INTERVAL
    li   t0, MTIME_LO
    lw   t1, 0(t0)
    li   t2, TIMER_INTERVAL
    add  t1, t1, t2
    li   t0, MTIMECMP_LO
    sw   t1, 0(t0)

    # Call C scheduler.  preempt_schedule() will internally call
    # context_switch(), which saves callee-saved regs (52 bytes)
    # ON TOP of the interrupt frame we just created.
    # When context_switch returns (possibly as a different task),
    # we land right here and fall through to .irq_restore to
    # unwind THAT task's interrupt frame.
    call preempt_schedule

.irq_restore:
    # ── restore interrupt frame ───────────────────────────────
    lw   t0, IF_MSTATUS(sp)
    csrw mstatus, t0
    lw   t0, IF_MEPC(sp)
    csrw mepc, t0

    lw   ra,  IF_RA(sp)
    lw   t0,  IF_T0(sp)
    lw   t1,  IF_T1(sp)
    lw   t2,  IF_T2(sp)
    lw   t3,  IF_T3(sp)
    lw   t4,  IF_T4(sp)
    lw   t5,  IF_T5(sp)
    lw   t6,  IF_T6(sp)
    lw   a0,  IF_A0(sp)
    lw   a1,  IF_A1(sp)
    lw   a2,  IF_A2(sp)
    lw   a3,  IF_A3(sp)
    lw   a4,  IF_A4(sp)
    lw   a5,  IF_A5(sp)
    lw   a6,  IF_A6(sp)
    lw   a7,  IF_A7(sp)

    addi sp, sp, IF_SIZE
    mret