#!/usr/bin/env python3
"""
laptop_bridge_debug.py — Laptop-side channel simulation (FPGA disabled)

Usage:
    python laptop_bridge_debug.py --channel channel_taps.txt [--max-iters N] [--eye] [--quiet]

Dependencies:
    pip install numpy matplotlib
"""

import argparse
import struct
import time
import sys
import numpy as np

# ── Equalizer dimensions (must match FPGA main.c) ─────────────────────
RX_FFE_PRE  = 5
RX_FFE_POST = 10
RX_FFE_LEN  = RX_FFE_PRE + 1 + RX_FFE_POST   # 14
N_DFE       = 1

# ── SerDes simulation parameters ──────────────────────────────────────
OSF         = 16
N_BIT       = 2048
ADC_BITS    = 8
TX_FFE_PRE  = 0
TX_FFE_POST = 0
TX_FFE_LEN  = TX_FFE_PRE + 1 + TX_FFE_POST

# ── LMS parameters (for software sim) ─────────────────────────────────
MU_FFE  = 0.0001
MU_DFE  = 0.00


# ═══════════════════════════════════════════════════════════════════════
# Channel simulation
# ═══════════════════════════════════════════════════════════════════════

def load_channel(filename):
    taps = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                taps.append(float(line))
    return np.array(taps)


def generate_prbs(n_bit):
    rng = np.random.default_rng(42)
    return rng.choice([-1.0, 1.0], size=n_bit)


def upsample(bits, osf):
    return np.repeat(bits, osf)


def apply_tx_ffe(bits_osf, tx_ffe, osf):
    out = np.copy(bits_osf)
    n = len(bits_osf)
    for i in range(TX_FFE_PRE * osf, n):
        acc = 0.0
        for k in range(TX_FFE_LEN):
            idx = i + (k - TX_FFE_PRE) * osf
            if 0 <= idx < n:
                acc += tx_ffe[k] * bits_osf[idx]
        out[i] = acc
    return out


def apply_channel_fir(signal, h_fir):
    return np.convolve(signal, h_fir, mode='full')[:len(signal)]


def adc_quantize(x, bits):
    clamped = np.clip(x, -1.0, 1.0)
    levels = 1 << bits
    xq = np.round((clamped + 1.0) * (levels - 1) / 2.0).astype(int)
    return xq * 2.0 / (levels - 1) - 1.0


def simple_cdr(signal, osf, n_cdr=1000):
    best_phase = 0
    best_metric = -1.0
    for phase in range(osf):
        samples = signal[phase::osf][:n_cdr]
        metric = np.mean(np.abs(samples))
        if metric > best_metric:
            best_metric = metric
            best_phase = phase
    return best_phase


def build_signal_chain(channel_file):
    """Build the full transmit → channel → ADC signal chain. Returns
    (post_adc, bits, sample_phase)."""
    h_fir = load_channel(channel_file)
    print(f"Loaded channel: {len(h_fir)} taps from {channel_file}")

    bits = generate_prbs(N_BIT)
    bits_osf = upsample(bits, OSF)

    tx_ffe = np.zeros(TX_FFE_LEN)
    tx_ffe[TX_FFE_PRE] = 1.0
    tx_signal = apply_tx_ffe(bits_osf, tx_ffe, OSF)

    post_channel = apply_channel_fir(tx_signal, h_fir)
    post_adc = adc_quantize(post_channel, ADC_BITS)

    sample_phase = simple_cdr(post_adc, OSF)
    print(f"CDR sample phase: {sample_phase}")

    return post_adc, bits, sample_phase


# ═══════════════════════════════════════════════════════════════════════
# Software-only LMS equalizer
# ═══════════════════════════════════════════════════════════════════════

def run_sim(channel_file, max_iters=None, verbose=True, show_eye=False):
    post_adc, bits, sample_phase = build_signal_chain(channel_file)

    print("Running software-only LMS (--sim mode)")

    # ── State ─────────────────────────────────────────────────
    rx_buf   = np.zeros(RX_FFE_LEN)
    d_hist   = np.zeros(N_DFE)
    rx_ffe   = np.zeros(RX_FFE_LEN)
    rx_ffe[RX_FFE_PRE] = 1.0
    dfe_taps = np.zeros(N_DFE)

    n_symbols = 0
    mse_acc = 0.0
    report_interval = 100
    mse_history = []

    eq_output = np.copy(post_adc).astype(float)

    total_samples = len(post_adc)
    start_idx = (TX_FFE_PRE + 1) * OSF

    for pt in range(start_idx, total_samples, 1):
        if pt + sample_phase >= total_samples:
            break

        sample = post_adc[pt + sample_phase]

        # Shift sample into buffer
        rx_buf[1:] = rx_buf[:-1]
        rx_buf[0] = sample

        # Equalizer output
        y_ffe = np.dot(rx_ffe, rx_buf)
        y = y_ffe + np.dot(dfe_taps, d_hist)
        eq_output[pt + sample_phase] = y

        # Decision
        d_hat = 1.0 if y >= 0.0 else -1.0

        # Training target
        sym_idx = pt // 1
        desired = bits[sym_idx] if sym_idx < N_BIT else d_hat

        error = desired - y
        mse_acc += error * error
        n_symbols += 1

        # ── LMS update ─────────────────────────────────────────
        rx_ffe  += MU_FFE * error * rx_buf
        dfe_taps -= MU_DFE * error * d_hist

        # Update decision history
        if N_DFE > 1:
            d_hist[1:] = d_hist[:-1]
        d_hist[0] = d_hat

        # Reporting
        if verbose and n_symbols % report_interval == 0:
            mse = mse_acc / report_interval
            mse_history.append(mse)
            mse_acc = 0.0
            print(f"[{n_symbols:5d}] MSE={mse:.6f}  "
                  f"FFE_main={rx_ffe[RX_FFE_PRE]:+.4f}  "
                  f"DFE[0]={dfe_taps[0]:+.4f}")

        if max_iters and n_symbols >= max_iters:
            break

    print(f"\nDone: {n_symbols} symbols processed (software LMS)")
    print(f"Final RX FFE taps: {rx_ffe}")
    print(f"Final DFE taps:    {dfe_taps}")

    if show_eye and n_symbols > 100:
        plot_eye_diagrams(post_adc, eq_output, sample_phase, OSF, bits)
        if mse_history:
            plot_convergence(mse_history)
    elif show_eye:
        print("Not enough symbols for eye diagram (need >100)")

    return rx_ffe, dfe_taps


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_eye_diagrams(post_adc, eq_output, sample_phase, osf, bits):
    import matplotlib.pyplot as plt

    n_traces = min(500, len(bits) - 10)
    start_sym = TX_FFE_PRE + 5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t_ui = np.linspace(-1, 1, 2 * osf)

    # Before equalization
    ax = axes[0]
    for i in range(start_sym, start_sym + n_traces):
        center = i * osf + sample_phase
        start = center - osf
        end = center + osf
        if start >= 0 and end < len(post_adc):
            trace = post_adc[start:end]
            if len(trace) == 2 * osf:
                ax.plot(t_ui, trace, color='blue', alpha=0.03, linewidth=0.5)
    ax.set_title('Eye Diagram — Before Equalization')
    ax.set_xlabel('Time (UI)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Sample point')
    ax.legend()

    # After equalization
    ax = axes[1]
    for i in range(start_sym, start_sym + min(n_traces, len(eq_output))):
        center = i * osf + sample_phase
        start = center - osf
        end = center + osf
        if start >= 0 and end < len(eq_output):
            trace = eq_output[start:end]
            if len(trace) == 2 * osf:
                ax.plot(t_ui, trace, color='green', alpha=0.03, linewidth=0.5)
    ax.set_title('Eye Diagram — After Equalization')
    ax.set_xlabel('Time (UI)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Sample point')
    ax.legend()

    plt.tight_layout()
    plt.savefig('eye_diagram.png', dpi=150)
    print("Eye diagram saved to eye_diagram.png")
    plt.show()


def plot_convergence(mse_history):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(mse_history, linewidth=1.5)
    ax.set_title('LMS Convergence')
    ax.set_xlabel('Symbol (×100)')
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    print("Convergence plot saved to convergence.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Laptop-side channel sim (FPGA disabled)')
    parser.add_argument('--channel',
                        help='Path to channel taps file', required=True)
    parser.add_argument('--max-iters', type=int, default=None,
                        help='Max symbols to process')
    parser.add_argument('--eye', action='store_true',
                        help='Show eye diagrams after equalization')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress periodic MSE output')

    args = parser.parse_args()

    # Run software-only LMS unconditionally
    run_sim(args.channel, max_iters=args.max_iters,
            verbose=not args.quiet, show_eye=args.eye)