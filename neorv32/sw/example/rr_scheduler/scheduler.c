#include "scheduler.h"

task_t   tasks[MAX_TASKS];
uint32_t stacks[MAX_TASKS][STACK_SIZE];

int current_task = -1;
int num_tasks    = 0;

/* ── assembly entry points (context.S) ───────────────────────── */
extern void context_switch(uint32_t **old_sp, uint32_t *new_sp);
extern void task_trampoline(void);

/* ── helpers ─────────────────────────────────────────────────── */

static void task_exit(void) {
    /* Safety net if a task function returns. */
    while (1)
        __asm__ volatile("wfi");
}

static inline int pick_next(void) {
    /* Round-robin among active tasks.
     * Loop body: 1 add + 1 modulo + 1 load + 1 branch ≈ 5–8 cycles
     * Worst case (MAX_TASKS iterations): ~20–30 cycles (< 30 target) */
    int next = current_task;
    for (int i = 0; i < num_tasks; i++) {
        next = (next + 1) % num_tasks;
        if (tasks[next].active)
            return next;
    }
    return current_task;               /* fallback: stay on same task */
}

/* ── public API ──────────────────────────────────────────────── */

void scheduler_init(void) {
    for (int i = 0; i < MAX_TASKS; i++)
        tasks[i].active = 0;
}

int create_task(void (*func)(void)) {
    if (num_tasks >= MAX_TASKS) return -1;

    int id = num_tasks++;

    /*
     * Build an initial COOPERATIVE frame (13 words = 52 bytes)
     * that context_switch will restore + ret into task_trampoline.
     *
     * Cooperative frame layout (matches context.S):
     *   sp[0]  = ra   →  task_trampoline
     *   sp[1]  = s0   →  func  (actual task entry point)
     *   sp[2]  = s1   →  task_exit  (safety-net return address)
     *   sp[3..12] = s2-s11 → 0
     */
    uint32_t *sp = &stacks[id][STACK_SIZE];
    sp -= 13;

    for (int i = 0; i < 13; i++)
        sp[i] = 0;

    sp[0] = (uint32_t)task_trampoline;  /* ra  */
    sp[1] = (uint32_t)func;             /* s0  */
    sp[2] = (uint32_t)task_exit;        /* s1  */

    tasks[id].sp     = sp;
    tasks[id].active = 1;

    return id;
}

/*
 * yield()  –  cooperative path
 *
 * Disables interrupts around the switch to prevent a timer
 * firing mid-context_switch.  context_switch saves 13 regs
 * (ra, s0-s11) = 52 bytes.  Total: < 50 cycles for the switch.
 */
void yield(void) {
    if (num_tasks < 2) return;

    /* Disable machine interrupts. */
    uint32_t mstatus;
    __asm__ volatile("csrrc %0, mstatus, %1"
                     : "=r"(mstatus) : "r"(1u << 3));

    int prev = current_task;
    int next = pick_next();

    if (next != prev) {
        current_task = next;
        context_switch(&tasks[prev].sp, tasks[next].sp);
    }

    /* Re-enable interrupts (restore MIE bit). */
    if (mstatus & (1u << 3))
        __asm__ volatile("csrs mstatus, %0" :: "r"(1u << 3));
}

/*
 * schedule()  –  called once from main() to launch the first task.
 *
 * We switch away from main's stack into task 0's cooperative
 * frame via context_switch.  main's SP is saved into a dummy
 * variable (we never return to main).
 */
void schedule(void) {
    if (num_tasks == 0) return;

    current_task = 0;

    uint32_t *dummy_sp;
    context_switch(&dummy_sp, tasks[0].sp);

    /* unreachable */
}

/*
 * preempt_schedule()  –  called from trap_vector on timer interrupt.
 *
 * At this point the interrupt frame (caller-saved regs, 80 bytes)
 * is already on the current task's stack.  We call context_switch
 * which pushes the cooperative frame (callee-saved, 52 bytes) on
 * top of that.
 *
 * When context_switch returns (possibly as a different task), we
 * return into trap_vector's .irq_restore, which pops that task's
 * interrupt frame and mrets back into it.
 *
 * Interrupts are already disabled by hardware (mstatus.MIE cleared
 * on trap entry), so no guard needed here.
 */
void preempt_schedule(void) {
    int prev = current_task;
    int next = pick_next();

    if (next != prev) {
        current_task = next;
        context_switch(&tasks[prev].sp, tasks[next].sp);
    }
    /* return → trap_vector's .irq_restore */
}