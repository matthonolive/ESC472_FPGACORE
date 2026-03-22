#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <stdint.h>

#define MAX_TASKS   4
#define STACK_SIZE  512     /* words (2 KB per task stack) */

typedef struct {
    uint32_t *sp;           /* saved stack pointer (cooperative frame) */
    uint8_t   active;       /* 1 = runnable                           */
} task_t;

/* Core API */
void scheduler_init(void);
int  create_task(void (*func)(void));
void yield(void);           /* cooperative: context_switch, 13 regs   */
void schedule(void);        /* launch first task (called once)        */

/* Called from trap_vector (timer.S) on timer interrupt.
 * Internally calls context_switch() to do the actual swap.           */
void preempt_schedule(void);

#endif