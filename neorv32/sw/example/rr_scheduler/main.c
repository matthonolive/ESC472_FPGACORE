#include <stdint.h>
#include "neorv32.h"
#include "scheduler.h"

void print(const char *s) {
    while (*s) {
        neorv32_uart_putc(NEORV32_UART0, *s++);
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

/* ── Command byte values (laptop → FPGA) ───────────────────── */
#define CMD_NORMAL   0x00
#define CMD_RESET    0x01

/* ── Status byte values (FPGA → laptop) ────────────────────── */
#define STATUS_OK    0x00
#define STATUS_RESET 0x01

/* ── Lane / multi-task constants ───────────────────────────── */
#define NUM_LANES    3

/* Per-lane tap storage */
static float rx_ffe[NUM_LANES][RX_FFE_LEN];
static float dfe[NUM_LANES][N_DFE];

/* Monotonic counter — each lane task grabs a unique ID */
static volatile int task_id_counter = 0;

/* UART serialisation — only the task holding the token may use
 * the UART.  The token rotates 0 → 1 → … → NUM_LANES-1 → 0.   */
volatile int uart_token = 0;

/* Set to 1 while a task is mid-packet so the timer ISR does
 * not preempt and let another task read UART bytes.             */
volatile int uart_busy = 0;

/* ── Helper: (re)initialise a lane's taps ──────────────────── */
static void init_lane_taps(int id) {
    for (int k = 0; k < RX_FFE_LEN; k++)
        rx_ffe[id][k] = 0.0f;
    rx_ffe[id][RX_FFE_PRE] = 1.0f;
    for (int k = 0; k < N_DFE; k++)
        dfe[id][k] = 0.0f;
}

/* ═══════════════════════════════════════════════════════════════
 * Lane task — one instance per SerDes lane (10 total).
 *
 * Protocol (per iteration):
 *   Laptop → FPGA : SYNC_RX | task_id | cmd | error(f32) | rx_buf | d_hist
 *   FPGA → Laptop : SYNC_TX | task_id | status | rx_ffe  | dfe
 *
 * cmd=0x01 tells this lane to reset its taps before the LMS
 * update. The FPGA echoes status=0x01 to confirm.
 * ═══════════════════════════════════════════════════════════════ */
void lane_task(void) {
    int my_id = task_id_counter++;
    init_lane_taps(my_id);

    float rx_buf[RX_FFE_LEN];
    float d_hist[N_DFE];

    while (1) {
        /* ── Wait for our turn on the UART ───────────────────── */
        while (uart_token != my_id)
            yield();

        /* ── Lock UART against preemption ────────────────────── */
        uart_busy = 1;

        /* ── Receive packet from laptop ──────────────────────── */
        uint8_t sync;
        do { sync = uart_getc(); } while (sync != SYNC_RX);

        uart_getc();                    /* task-ID (for framing) */
        uint8_t cmd = uart_getc();      /* command byte          */

        float error = recv_float();
        for (int k = 0; k < RX_FFE_LEN; k++)
            rx_buf[k] = recv_float();
        for (int k = 0; k < N_DFE; k++)
            d_hist[k] = recv_float();

        /* ── Handle reset command ────────────────────────────── */
        uint8_t did_reset = 0;
        if (cmd == CMD_RESET) {
            init_lane_taps(my_id);
            did_reset = 1;
        }

        /* ── LMS tap update ──────────────────────────────────── */
        for (int k = 0; k < RX_FFE_LEN; k++)
            rx_ffe[my_id][k] += MU_FFE * error * rx_buf[k];
        for (int k = 0; k < N_DFE; k++)
            dfe[my_id][k] -= MU_DFE * error * d_hist[k];

        /* ── Send response to laptop ─────────────────────────── */
        uart_putc(SYNC_TX);
        uart_putc((uint8_t)my_id);
        uart_putc(did_reset ? STATUS_RESET : STATUS_OK);
        for (int k = 0; k < RX_FFE_LEN; k++)
            send_float(rx_ffe[my_id][k]);
        for (int k = 0; k < N_DFE; k++)
            send_float(dfe[my_id][k]);

        /* ── Release UART, pass token to next lane ───────────── */
        uart_busy = 0;
        uart_token = (my_id + 1) % NUM_LANES;
        yield();
    }
}

/* ═══════════════════════════════════════════════════════════════ */

int main(void) {
    neorv32_uart_setup(NEORV32_UART0, 19200, 0);

    print("10-lane SerDes equalizer  —  RR scheduler\n");

    /* Point mtvec to our trap handler */
    extern void trap_vector(void);
    __asm__ volatile("csrw mtvec, %0" :: "r"((uint32_t)trap_vector & ~0x3));

    scheduler_init();

    for (int i = 0; i < NUM_LANES; i++)
        create_task(lane_task);

    schedule();
    while (1);
}