#!/usr/bin/env python3
"""
laptop_bridge.py  —  Laptop-side channel simulation + UART bridge to FPGA equalizer
 
Usage:
    # Normal FPGA mode
    python laptop_bridge.py --port COM3 --channel channel_taps.txt --eye
 
    # Debug mode — hex-dump every byte on the wire
    python laptop_bridge.py --port COM3 --channel channel_taps.txt --debug
 
    # Listen mode — just print raw bytes from FPGA (no sending)
    python laptop_bridge.py --port COM3 --listen
 
    # Sim mode — software-only LMS, no FPGA needed (proves algorithm works)
    python laptop_bridge.py --channel channel_taps.txt --sim --eye
 
Dependencies:
    pip install pyserial numpy matplotlib
"""
 
import argparse
import struct
import time
import sys
import numpy as np
 
# ── Equalizer dimensions (must match FPGA main.c) ─────────────────────
RX_FFE_PRE  = 3
RX_FFE_POST = 10
RX_FFE_LEN  = RX_FFE_PRE + 1 + RX_FFE_POST   # 14
N_DFE       = 1
 
# ── Sync bytes ─────────────────────────────────────────────────────────
SYNC_TX = 0xAA   # laptop → FPGA
SYNC_RX = 0x55   # FPGA → laptop
 
# ── SerDes simulation parameters ──────────────────────────────────────
OSF         = 16
N_BIT       = 2048
ADC_BITS    = 5
TX_FFE_PRE  = 4
TX_FFE_POST = 0
TX_FFE_LEN  = TX_FFE_PRE + 1 + TX_FFE_POST
 
# ── LMS parameters (for --sim mode) ──────────────────────────────────
MU_FFE  = 0.01
MU_DFE  = 0.005
 
 
# ═══════════════════════════════════════════════════════════════════════
# Channel simulation (laptop side)
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
# Packet I/O
# ═══════════════════════════════════════════════════════════════════════
 
def build_error_packet(error, rx_buf, d_hist):
    """Build the binary packet: SYNC_TX | error(f32) | rx_buf(f32×14) | d_hist(f32×1)"""
    pkt = bytearray([SYNC_TX])
    pkt += struct.pack('<f', error)
    for v in rx_buf:
        pkt += struct.pack('<f', v)
    for v in d_hist:
        pkt += struct.pack('<f', v)
    return bytes(pkt)
 
 
def send_error_packet(ser, error, rx_buf, d_hist, debug=False):
    pkt = build_error_packet(error, rx_buf, d_hist)
    if debug:
        print(f"  TX ({len(pkt)} bytes): {pkt.hex(' ')}")
    ser.write(pkt)
    ser.flush()
 
 
def recv_tap_packet(ser, debug=False, timeout_retries=3):
    """Receive taps from FPGA.  Scans for SYNC_RX, then reads the float payload.
    With debug=True, prints every byte received for diagnosis."""
 
    n_floats = RX_FFE_LEN + N_DFE        # 15 floats = 60 bytes
    payload_len = n_floats * 4
 
    # ── Scan for sync byte ────────────────────────────────────
    scan_count = 0
    max_scan = payload_len * 4  # give up after scanning way too many bytes
 
    while scan_count < max_scan:
        b = ser.read(1)
        if len(b) == 0:
            raise TimeoutError("Timed out waiting for FPGA sync byte "
                               f"(scanned {scan_count} bytes)")
        scan_count += 1
        if debug:
            print(f"  RX scan [{scan_count:3d}]: 0x{b[0]:02X}"
                  f"{'  ← SYNC!' if b[0] == SYNC_RX else ''}")
        if b[0] == SYNC_RX:
            break
    else:
        raise TimeoutError(f"Never found SYNC_RX (0x{SYNC_RX:02X}) in "
                           f"{max_scan} bytes")
 
    # ── Read payload ──────────────────────────────────────────
    raw = ser.read(payload_len)
    if debug:
        print(f"  RX payload ({len(raw)}/{payload_len} bytes): "
              f"{raw[:20].hex(' ')}{'...' if len(raw) > 20 else ''}")
    if len(raw) < payload_len:
        raise TimeoutError(f"Short payload: got {len(raw)}/{payload_len} bytes")
 
    values = struct.unpack(f'<{n_floats}f', raw)
    rx_ffe = np.array(values[:RX_FFE_LEN])
    dfe    = np.array(values[RX_FFE_LEN:])
    return rx_ffe, dfe
 
 
# ═══════════════════════════════════════════════════════════════════════
# Listen mode — just show what the FPGA sends
# ═══════════════════════════════════════════════════════════════════════
 
def run_listen(port, baud, duration=10):
    """Listen to raw bytes from the FPGA for `duration` seconds."""
    import serial
    ser = serial.Serial(port, baud, timeout=0.5)
    print(f"Listening on {port} @ {baud} for {duration}s  (Ctrl-C to stop)")
    print("─" * 60)
 
    t0 = time.time()
    total = 0
    try:
        while time.time() - t0 < duration:
            data = ser.read(ser.in_waiting or 1)
            if data:
                total += len(data)
                # Try to show as text if printable, otherwise hex
                try:
                    text = data.decode('ascii')
                    if text.isprintable() or text.strip():
                        print(f"  TEXT ({len(data):3d}B): {text.rstrip()}")
                    else:
                        print(f"  HEX  ({len(data):3d}B): {data.hex(' ')}")
                except UnicodeDecodeError:
                    print(f"  HEX  ({len(data):3d}B): {data.hex(' ')}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    print("─" * 60)
    print(f"Total received: {total} bytes in {time.time()-t0:.1f}s")
 
 
# ═══════════════════════════════════════════════════════════════════════
# Sim mode — software-only LMS (no FPGA)
# ═══════════════════════════════════════════════════════════════════════
 
def run_sim(channel_file, max_iters=None, verbose=True, show_eye=False):
    """Pure-software LMS equalizer — proves the algorithm converges without
    needing the FPGA at all."""
 
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
 
    for pt in range(start_idx, total_samples, OSF):
        if pt + sample_phase >= total_samples:
            break
 
        sample = post_adc[pt + sample_phase]
 
        # Shift sample into buffer
        rx_buf[1:] = rx_buf[:-1]
        rx_buf[0] = sample
 
        # Equalizer output
        y_ffe = np.dot(rx_ffe, rx_buf)
        y = y_ffe - np.dot(dfe_taps, d_hist)
        eq_output[pt + sample_phase] = y
 
        # Decision
        d_hat = 1.0 if y >= 0.0 else -1.0
 
        # Training target
        sym_idx = pt // OSF
        desired = bits[sym_idx] if sym_idx < N_BIT else d_hat
 
        error = desired - y
        mse_acc += error * error
        n_symbols += 1
 
        # ── LMS update (what the FPGA would do) ──────────────
        rx_ffe  += MU_FFE * error * rx_buf
        dfe_taps -= MU_DFE * error * d_hist
 
        # Update decision history AFTER using it for LMS
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
 
    # ── Final report ──────────────────────────────────────────
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
# FPGA bridge mode
# ═══════════════════════════════════════════════════════════════════════
 
def drain_startup(ser, wait=1.0):
    """Drain any startup text the FPGA prints on boot."""
    ser.reset_input_buffer()
    time.sleep(wait)
    startup_text = b""
    while ser.in_waiting:
        startup_text += ser.read(ser.in_waiting)
        time.sleep(0.1)
    if startup_text:
        print(f"FPGA startup: {startup_text.decode(errors='replace').strip()}")
    else:
        print("(no startup text from FPGA)")
 
 
def run_bridge(port, baud, channel_file, max_iters=None, verbose=True,
               show_eye=False, debug=False):
    """Main loop: simulate channel on laptop, do LMS on FPGA."""
    import serial
 
    post_adc, bits, sample_phase = build_signal_chain(channel_file)
 
    # ── Open serial port ──────────────────────────────────────
    print(f"Opening {port} @ {baud}...")
    ser = serial.Serial(port, baud, timeout=2.0)
    drain_startup(ser, wait=1.5)
    print(f"Serial open: {port} @ {baud}")
 
    # ── Compute packet sizes for sanity check ─────────────────
    tx_pkt_len = 1 + (1 + RX_FFE_LEN + N_DFE) * 4    # SYNC + floats
    rx_pkt_len = 1 + (RX_FFE_LEN + N_DFE) * 4         # SYNC + floats
    print(f"Packet sizes — TX: {tx_pkt_len} bytes, expected RX: {rx_pkt_len} bytes")
    print(f"Starting equalization loop...")
    if debug:
        print("(debug mode: showing raw bytes)")
    print()
 
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
 
    for pt in range(start_idx, total_samples, OSF):
        if pt + sample_phase >= total_samples:
            break
 
        sample = post_adc[pt + sample_phase]
 
        # Shift sample into FFE buffer
        rx_buf[1:] = rx_buf[:-1]
        rx_buf[0] = sample
 
        # Compute equalizer output with current FPGA taps
        y_ffe = np.dot(rx_ffe, rx_buf)
        y = y_ffe - np.dot(dfe_taps, d_hist)
        eq_output[pt + sample_phase] = y
 
        # Decision
        d_hat = 1.0 if y >= 0.0 else -1.0
 
        # Training target
        sym_idx = pt // OSF
        desired = bits[sym_idx] if sym_idx < N_BIT else d_hat
 
        error = desired - y
        mse_acc += error * error
        n_symbols += 1
 
        # ── Send to FPGA ──────────────────────────────────────
        if debug:
            print(f"--- Symbol {n_symbols} ---")
            print(f"  error={error:+.4f}  sample={sample:+.4f}  "
                  f"desired={desired:+.1f}  y={y:+.4f}")
 
        send_error_packet(ser, float(error), rx_buf.tolist(), d_hist.tolist(),
                          debug=debug)
 
        # ── Receive updated taps from FPGA ────────────────────
        try:
            rx_ffe, dfe_taps = recv_tap_packet(ser, debug=debug)
        except TimeoutError as e:
            print(f"\nTimeout at symbol {n_symbols}: {e}")
            # Show what's sitting in the buffer
            leftover = ser.read(ser.in_waiting) if ser.in_waiting else b""
            if leftover:
                print(f"  Leftover in buffer ({len(leftover)} bytes): "
                      f"{leftover[:40].hex(' ')}")
                try:
                    print(f"  As text: {leftover.decode('ascii', errors='replace')}")
                except Exception:
                    pass
            else:
                print("  (serial buffer is empty — FPGA sent nothing back)")
            break
 
        if debug:
            print(f"  Got taps: FFE_main={rx_ffe[RX_FFE_PRE]:+.6f}  "
                  f"DFE[0]={dfe_taps[0]:+.6f}")
 
        # Update decision history AFTER LMS update
        # (must use the OLD d_hist for the gradient, matching sim mode)
        if N_DFE > 1:
            d_hist[1:] = d_hist[:-1]
        d_hist[0] = d_hat
 
        # ── Periodic reporting ────────────────────────────────
        if verbose and n_symbols % report_interval == 0:
            mse = mse_acc / report_interval
            mse_history.append(mse)
            mse_acc = 0.0
            main_tap = rx_ffe[RX_FFE_PRE]
            print(f"[{n_symbols:5d}] MSE={mse:.6f}  "
                  f"FFE_main={main_tap:+.4f}  DFE[0]={dfe_taps[0]:+.4f}")
 
        if max_iters and n_symbols >= max_iters:
            break
 
    # ── Final report ──────────────────────────────────────────
    print(f"\nDone: {n_symbols} symbols processed")
    print(f"Final RX FFE taps: {rx_ffe}")
    print(f"Final DFE taps:    {dfe_taps}")
 
    ser.close()
 
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
        description='Laptop-side channel sim + UART bridge to FPGA equalizer')
    parser.add_argument('--port', default='COM3',
                        help='Serial port (default: COM3)')
    parser.add_argument('--baud', type=int, default=19200,
                        help='Baud rate (default: 19200)')
    parser.add_argument('--channel',
                        help='Path to channel taps file')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='Max symbols to process')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress periodic MSE output')
    parser.add_argument('--eye', action='store_true',
                        help='Show eye diagrams after equalization')
    parser.add_argument('--debug', action='store_true',
                        help='Hex-dump all serial traffic')
    parser.add_argument('--listen', action='store_true',
                        help='Just listen to FPGA output (no sending)')
    parser.add_argument('--listen-time', type=int, default=10,
                        help='Seconds to listen in --listen mode (default: 10)')
    parser.add_argument('--sim', action='store_true',
                        help='Software-only LMS (no FPGA, no serial port)')
 
    args = parser.parse_args()
 
    if args.listen:
        run_listen(args.port, args.baud, duration=args.listen_time)
 
    elif args.sim:
        if not args.channel:
            parser.error("--sim requires --channel")
        run_sim(args.channel, max_iters=args.max_iters,
                verbose=not args.quiet, show_eye=args.eye)
 
    else:
        if not args.channel:
            parser.error("FPGA mode requires --channel")
        run_bridge(args.port, args.baud, args.channel,
                   max_iters=args.max_iters, verbose=not args.quiet,
                   show_eye=args.eye, debug=args.debug)