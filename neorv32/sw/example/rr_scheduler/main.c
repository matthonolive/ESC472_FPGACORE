#include <stdint.h>
#include "neorv32.h"
#include "scheduler.h"

void print(const char *s) {
    while (*s) {
        neorv32_uart_putc(NEORV32_UART0, *s++);
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

/* ── UART byte helpers ─────────────────────────────────────── */
static void uart_putc(uint8_t c) {
    neorv32_uart_putc(NEORV32_UART0, (char)c);
}

static uint8_t uart_getc(void) {
    return (uint8_t)neorv32_uart_getc(NEORV32_UART0);
}

/* ── float ↔ bytes ─────────────────────────────────────────── */
typedef union { float f; uint8_t b[4]; } f2b_t;

static float recv_float(void) {
    f2b_t u;
    for (int i = 0; i < 4; i++)
        u.b[i] = uart_getc();
    return u.f;
}

static void send_float(float v) {
    f2b_t u;
    u.f = v;
    for (int i = 0; i < 4; i++)
        uart_putc(u.b[i]);
}

/* ── Equalizer constants (must match laptop_bridge.py) ─────── */
#define RX_FFE_PRE   3
#define RX_FFE_POST  10
#define RX_FFE_LEN   (RX_FFE_PRE + 1 + RX_FFE_POST)
#define N_DFE        1
#define MU_FFE       0.01f
#define MU_DFE       0.00f
#define SYNC_RX      0xAA
#define SYNC_TX      0x55

/* ── Multi-task support ────────────────────────────────────── */
#define MAX_TASKS    4

/* Per-task tap storage (global so it doesn't eat stack) */
static float rx_ffe[MAX_TASKS][RX_FFE_LEN];
static float dfe[MAX_TASKS][N_DFE];

/* Monotonic counter — each task1 instance grabs a unique ID */
static volatile int task_id_counter = 0;

/* ── Task 1: equalizer over UART ───────────────────────────── */
void task1(void) {
    int my_id = task_id_counter++;

    /* Initialise this task's taps */
    for (int k = 0; k < RX_FFE_LEN; k++)
        rx_ffe[my_id][k] = 0.0f;
    rx_ffe[my_id][RX_FFE_PRE] = 1.0f;
    for (int k = 0; k < N_DFE; k++)
        dfe[my_id][k] = 0.0f;

    float rx_buf[RX_FFE_LEN];
    float d_hist[N_DFE];

    while (1) {
        /* ── Wait for sync byte ──────────────────────────────── */
        uint8_t sync;
        do { sync = uart_getc(); } while (sync != SYNC_RX);

        /* ── Read (and discard) the task-ID the laptop sent ──── */
        uart_getc();

        /* ── Read payload ────────────────────────────────────── */
        float error = recv_float();
        for (int k = 0; k < RX_FFE_LEN; k++)
            rx_buf[k] = recv_float();
        for (int k = 0; k < N_DFE; k++)
            d_hist[k] = recv_float();

        /* ── LMS update ──────────────────────────────────────── */
        for (int k = 0; k < RX_FFE_LEN; k++)
            rx_ffe[my_id][k] += MU_FFE * error * rx_buf[k];
        for (int k = 0; k < N_DFE; k++)
            dfe[my_id][k] -= MU_DFE * error * d_hist[k];

        /* ── Send response: SYNC + task-ID + taps ────────────── */
        uart_putc(SYNC_TX);
        uart_putc((uint8_t)my_id);
        for (int k = 0; k < RX_FFE_LEN; k++)
            send_float(rx_ffe[my_id][k]);
        for (int k = 0; k < N_DFE; k++)
            send_float(dfe[my_id][k]);

        yield();
    }
}

/* ── Task 2: silent busy-wait (no UART prints!) ────────────── */
/*    Keeps scheduler exercised without corrupting the protocol */
void task2(void) {
    while (1) {
        for (volatile int i = 0; i < 500000; i++);
        yield();
    }
}

int main(void) {
    // Initialize UART0 at 19200 baud
    neorv32_uart_setup(NEORV32_UART0, 19200, 0);

    print("RR Scheduler starting...\n");

    // Point mtvec to our trap handler
    extern void trap_vector(void);
    __asm__ volatile("csrw mtvec, %0" :: "r"((uint32_t)trap_vector & ~0x3));

    scheduler_init();
    create_task(task1);
    create_task(task1); // second equalizer task

    schedule();
    while (1);
}