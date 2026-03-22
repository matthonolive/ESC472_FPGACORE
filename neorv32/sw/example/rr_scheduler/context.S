# ──────────────────────────────────────────────────────────────
# context_switch  –  cooperative (callee-saved registers only)
#
#   void context_switch(uint32_t **old_sp, uint32_t *new_sp);
#       a0 = &old_sp    a1 = new_sp
#
# Saves:  ra, s0-s11  =  13 registers  =  52 bytes
# Cycle budget: 13 sw + 1 sw(sp) + 1 mv + 13 lw + 1 addi + 1 ret
#             = 30 instructions  ≈  30–40 cycles   (< 50 target)
# ──────────────────────────────────────────────────────────────

.global context_switch
.global task_trampoline

context_switch:
    addi sp, sp, -52
    sw   ra,   0(sp)
    sw   s0,   4(sp)
    sw   s1,   8(sp)
    sw   s2,  12(sp)
    sw   s3,  16(sp)
    sw   s4,  20(sp)
    sw   s5,  24(sp)
    sw   s6,  28(sp)
    sw   s7,  32(sp)
    sw   s8,  36(sp)
    sw   s9,  40(sp)
    sw   s10, 44(sp)
    sw   s11, 48(sp)

    sw   sp, 0(a0)              # *old_sp = sp

    mv   sp, a1                 # sp = new_sp

    lw   ra,   0(sp)
    lw   s0,   4(sp)
    lw   s1,   8(sp)
    lw   s2,  12(sp)
    lw   s3,  16(sp)
    lw   s4,  20(sp)
    lw   s5,  24(sp)
    lw   s6,  28(sp)
    lw   s7,  32(sp)
    lw   s8,  36(sp)
    lw   s9,  40(sp)
    lw   s10, 44(sp)
    lw   s11, 48(sp)

    addi sp, sp, 52
    ret                         # jumps to restored ra


# ──────────────────────────────────────────────────────────────
# task_trampoline  –  first-ever entry into a new task
#
# The initial cooperative frame sets:
#     ra  = task_trampoline
#     s0  = real task function
#     s1  = task_exit  (safety net)
#
# context_switch restores these then does  ret → here.
# We set ra = task_exit so a returning task doesn't crash,
# then jump to the actual task function.
# ──────────────────────────────────────────────────────────────

task_trampoline:
    mv   ra, s1                 # ra = task_exit
    jr   s0                     # jump to task entry