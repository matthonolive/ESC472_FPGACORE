#include "scheduler.h"

task_t   tasks[MAX_TASKS];
uint32_t stacks[MAX_TASKS][STACK_SIZE];

int current_task = -1;
int num_tasks    = 0;

/* ── assembly entry points (context.S) ───────────────────────── */
extern void context_switch(uint32_t **old_sp, uint32_t *new_sp);
extern void task_trampoline(void);

/* ── defined in main.c ───────────────────────────────────────── */
extern volatile int uart_busy;

/* ── helpers ─────────────────────────────────────────────────── */

static void task_exit(void) {
    while (1)
        __asm__ volatile("wfi");
}

static inline int pick_next(void) {
    int next = current_task;
    for (int i = 0; i < num_tasks; i++) {
        next = (next + 1) % num_tasks;
        if (tasks[next].active)
            return next;
    }
    return current_task;
}

/* ── public API ──────────────────────────────────────────────── */

void scheduler_init(void) {
    for (int i = 0; i < MAX_TASKS; i++)
        tasks[i].active = 0;
}

int create_task(void (*func)(void)) {
    if (num_tasks >= MAX_TASKS) return -1;

    int id = num_tasks++;

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

void yield(void) {
    if (num_tasks < 2) return;

    uint32_t mstatus;
    __asm__ volatile("csrrc %0, mstatus, %1"
                     : "=r"(mstatus) : "r"(1u << 3));

    int prev = current_task;
    int next = pick_next();

    if (next != prev) {
        current_task = next;
        context_switch(&tasks[prev].sp, tasks[next].sp);
    }

    if (mstatus & (1u << 3))
        __asm__ volatile("csrs mstatus, %0" :: "r"(1u << 3));
}

void schedule(void) {
    if (num_tasks == 0) return;

    current_task = 0;

    uint32_t *dummy_sp;
    context_switch(&dummy_sp, tasks[0].sp);
}

/*
 * preempt_schedule()  –  called from trap_vector on timer interrupt.
 *
 * Skip context-switch if a task is mid-UART-packet (uart_busy == 1)
 * to prevent byte corruption on the shared UART line.
 */
void preempt_schedule(void) {
    if (uart_busy)
        return;

    int prev = current_task;
    int next = pick_next();

    if (next != prev) {
        current_task = next;
        context_switch(&tasks[prev].sp, tasks[next].sp);
    }
}