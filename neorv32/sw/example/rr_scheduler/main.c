#include <stdint.h>
#include "scheduler.h"

// NEORV32 UART0 registers
#define UART0_CTRL (*(volatile uint32_t *)0xFFFFFFA0)
#define UART0_DATA (*(volatile uint32_t *)0xFFFFFFA4)

void print(const char *s) {
    while (*s) {
        // wait until TX FIFO is not full (bit 18 of CTRL)
        while (UART0_CTRL & (1 << 18));
        UART0_DATA = (uint32_t)*s++;
    }
}

static void print_int(int v) {
    char buf[12];
    int  i = 0;
    if (v == 0) { print("0"); return; }
    while (v > 0) { buf[i++] = '0' + (v % 10); v /= 10; }
    while (i--) {
        char str[2] = { buf[i], '\0' };
        print(str);
    }
}

/* ── Task 1: cooperative yielding ──────────────────────────── */
void task1(void) {
    int count = 0;
    while (1) {
        print("Task 1 running (");
        print_int(count++);
        print(")\n");
        for (volatile int i = 0; i < 100000; i++);
        yield();
    }
}

/* ── Task 2: also yields cooperatively ─────────────────────── */
void task2(void) {
    int count = 0;
    while (1) {
        print("Task 2 running (");
        print_int(count++);
        print(")\n");
        for (volatile int i = 0; i < 100000; i++);
        yield();
    }
}

/* ── Task 3: exercises single-precision FPU ────────────────── */
void task3(void) {
    float accum = 0.0;
    int    iter  = 0;
    while (1) {
        for (int k = 0; k < 1000; k++) {
            int n = iter * 1000 + k;
            float term = 1.0 / (2.0 * n + 1.0);
            if (n & 1) accum -= term;
            else       accum += term;
        }
        iter++;
        print("Task 3 (FP): pi ~ ");
        int approx = (int)(accum * 4.0);
        print_int(approx);
        print(".");
        int frac = (int)((accum * 4.0 - approx) * 10000);
        if (frac < 0) frac = -frac;
        if (frac < 1000) print("0");
        if (frac < 100)  print("0");
        if (frac < 10)   print("0");
        print_int(frac);
        print("\n");
        yield();
    }
}

int main(void) {

    // Point mtvec to our trap handler
    extern void trap_vector(void);
    __asm__ volatile("csrw mtvec, %0" :: "r"((uint32_t)trap_vector & ~0x3));
    scheduler_init();

    create_task(task1);
    create_task(task2);
    create_task(task3);

    schedule();      /* launches first task – never returns */

    while (1);
}