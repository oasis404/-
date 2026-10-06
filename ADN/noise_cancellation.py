"""
Noise Cancellation using LMS and RLS Adaptive Filters
Implementation from scratch without external libraries (except struct for binary I/O)
"""

import struct
import math
import random


# Diagonal loading applied to the RLS inverse correlation matrix P.
# A small positive floor keeps P positive-definite and prevents the
# weights from diverging on long signals.
DIAG_LOAD = 1e-9


# ============================================================================
# WAV File I/O Functions (from scratch)
# ============================================================================

def read_wav(filename):
    """
    Read a WAV file and return sample rate and audio data.
    Returns: (sample_rate, num_channels, audio_data)
    audio_data is a list of samples (normalized to -1.0 to 1.0)
    """
    with open(filename, 'rb') as f:
        # Read RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError("Not a valid WAV file (missing RIFF header)")
        
        file_size = struct.unpack('<I', f.read(4))[0]
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError("Not a valid WAV file (missing WAVE header)")
        
        # Read fmt chunk
        fmt_chunk = f.read(4)
        if fmt_chunk != b'fmt ':
            raise ValueError("Missing fmt chunk")
        
        fmt_size = struct.unpack('<I', f.read(4))[0]
        audio_format = struct.unpack('<H', f.read(2))[0]  # 1 = PCM
        num_channels = struct.unpack('<H', f.read(2))[0]
        sample_rate = struct.unpack('<I', f.read(4))[0]
        byte_rate = struct.unpack('<I', f.read(4))[0]
        block_align = struct.unpack('<H', f.read(2))[0]
        bits_per_sample = struct.unpack('<H', f.read(2))[0]
        
        # Skip any extra fmt bytes
        if fmt_size > 16:
            f.read(fmt_size - 16)
        
        # Find data chunk
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                raise ValueError("No data chunk found")
            chunk_size = struct.unpack('<I', f.read(4))[0]
            
            if chunk_id == b'data':
                break
            else:
                # Skip this chunk
                f.read(chunk_size)
        
        # Read audio data
        num_samples = chunk_size // (bits_per_sample // 8) // num_channels
        audio_data = []
        
        if bits_per_sample == 16:
            for _ in range(num_samples * num_channels):
                sample = struct.unpack('<h', f.read(2))[0]
                # Normalize to -1.0 to 1.0
                audio_data.append(sample / 32768.0)
        elif bits_per_sample == 8:
            for _ in range(num_samples * num_channels):
                sample = struct.unpack('B', f.read(1))[0]
                # Convert unsigned 8-bit to signed and normalize
                audio_data.append((sample - 128) / 128.0)
        else:
            raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")
        
        return sample_rate, num_channels, audio_data


def write_wav(filename, sample_rate, num_channels, audio_data, bits_per_sample=16):
    """
    Write audio data to a WAV file.
    audio_data should be a list of samples (normalized to -1.0 to 1.0)
    """
    num_samples = len(audio_data) // num_channels
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * num_channels * bits_per_sample // 8
    
    with open(filename, 'wb') as f:
        # Write RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        
        # Write fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # fmt chunk size
        f.write(struct.pack('<H', 1))   # audio format (PCM)
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))
        
        # Write data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        
        # Write audio samples
        if bits_per_sample == 16:
            for sample in audio_data:
                # Clamp and convert to 16-bit integer
                sample = max(-1.0, min(1.0, sample))
                int_sample = int(sample * 32767)
                f.write(struct.pack('<h', int_sample))
        elif bits_per_sample == 8:
            for sample in audio_data:
                # Clamp and convert to 8-bit unsigned integer
                sample = max(-1.0, min(1.0, sample))
                int_sample = int((sample + 1.0) * 127.5)
                f.write(struct.pack('B', int_sample))


# ============================================================================
# Noise Generation
# ============================================================================

def generate_white_noise(length, amplitude=0.1):
    """Generate white noise with specified amplitude."""
    return [random.uniform(-amplitude, amplitude) for _ in range(length)]


def generate_colored_noise(length, amplitude=0.1, color='pink'):
    """
    Generate colored noise (pink or brown).
    Pink noise: 1/f spectrum
    Brown noise: 1/f^2 spectrum
    """
    white = generate_white_noise(length, amplitude)
    
    if color == 'white':
        return white
    
    # Simple approximation using running sum
    colored = [0.0] * length
    state = 0.0
    
    if color == 'pink':
        alpha = 0.5
    elif color == 'brown':
        alpha = 0.9
    else:
        alpha = 0.5
    
    for i in range(length):
        state = alpha * state + (1 - alpha) * white[i]
        colored[i] = state
    
    # Normalize
    max_val = max(abs(s) for s in colored)
    if max_val > 0:
        colored = [s * amplitude / max_val for s in colored]
    
    return colored


def add_noise_to_signal(signal, noise_level=0.1, noise_type='white'):
    """Add synthetic noise to a signal."""
    noise = generate_colored_noise(len(signal), noise_level, noise_type)
    return [signal[i] + noise[i] for i in range(len(signal))], noise


# ============================================================================
# LMS Adaptive Filter
# ============================================================================

def lms_filter(corrupted_signal, reference_noise, filter_order=32, step_size=0.01):
    """
    Least Mean Squares (LMS) adaptive filter.
    
    Args:
        corrupted_signal: The noisy signal to be cleaned
        reference_noise: Reference noise signal (correlated with the noise in corrupted_signal)
        filter_order: Number of filter coefficients
        step_size: Learning rate (mu)
    
    Returns:
        cleaned_signal: Filtered output
        error_signal: Error signal (desired - filtered)
        weights_history: History of weight updates (for visualization)
    """
    n = len(corrupted_signal)
    weights = [0.0] * filter_order
    cleaned_signal = [0.0] * n
    error_signal = [0.0] * n
    weights_history = []
    
    # Pad reference noise
    padded_ref = [0.0] * (filter_order - 1) + reference_noise
    
    for i in range(n):
        # Extract input vector (last filter_order samples)
        x = padded_ref[i:i + filter_order]
        x.reverse()  # Reverse for proper convolution
        
        # Filter output (estimate of noise)
        y = sum(weights[j] * x[j] for j in range(filter_order))
        
        # Error signal (desired signal = corrupted - estimated noise)
        error = corrupted_signal[i] - y
        error_signal[i] = error
        cleaned_signal[i] = error
        
        # Update weights: w(n+1) = w(n) + mu * e(n) * x(n)
        for j in range(filter_order):
            weights[j] += step_size * error * x[j]
        
        # Store weights periodically for visualization
        if i % 1000 == 0:
            weights_history.append(list(weights))
    
    return cleaned_signal, error_signal, weights_history


# ============================================================================
# RLS Adaptive Filter
# ============================================================================

def rls_filter(corrupted_signal, reference_noise, filter_order=32, forgetting_factor=0.99, delta=1.0, verbose=True):
    """
    Recursive Least Squares (RLS) adaptive filter.
    
    Args:
        corrupted_signal: The noisy signal to be cleaned
        reference_noise: Reference noise signal
        filter_order: Number of filter coefficients
        forgetting_factor: Lambda (0 < lambda <= 1), typically 0.95-0.99
        delta: Initialization parameter for P matrix (typically 0.1-1.0)
    
    Returns:
        cleaned_signal: Filtered output
        error_signal: Error signal
        weights_history: History of weight updates
    """
    n = len(corrupted_signal)
    weights = [0.0] * filter_order
    
    # Initialize inverse correlation matrix P = delta^-1 * I
    P = [[0.0] * filter_order for _ in range(filter_order)]
    for i in range(filter_order):
        P[i][i] = 1.0 / delta
    
    cleaned_signal = [0.0] * n
    error_signal = [0.0] * n
    weights_history = []
    
    # Pad reference noise
    padded_ref = [0.0] * (filter_order - 1) + reference_noise
    
    # RLS costs O(N^2) per sample, so a long signal takes minutes in pure
    # Python. Emit progress so the run does not look hung.
    progress_step = max(1, n // 20)
    if verbose:
        print(f"    RLS processing {n} samples, this takes a while...", flush=True)
    
    for i in range(n):
        # Extract input vector
        x = padded_ref[i:i + filter_order]
        x.reverse()
        
        # Filter output
        y = sum(weights[j] * x[j] for j in range(filter_order))
        
        # Error signal
        error = corrupted_signal[i] - y
        error_signal[i] = error
        cleaned_signal[i] = error
        
        # RLS update
        # Compute P * x
        Px = [sum(P[j][k] * x[k] for k in range(filter_order)) for j in range(filter_order)]
        
        # Compute x^T * P * x
        xTPx = sum(x[j] * Px[j] for j in range(filter_order))
        
        # Compute gain vector: g = P * x / (lambda + x^T * P * x)
        denominator = forgetting_factor + xTPx
        gain = [Px[j] / denominator for j in range(filter_order)]
        
        # Update weights: w(n+1) = w(n) + g * e(n)
        for j in range(filter_order):
            weights[j] += gain[j] * error
        
        # Update P matrix: P(n+1) = (P(n) - g * x^T * P(n)) / lambda
        for j in range(filter_order):
            for k in range(filter_order):
                P[j][k] = (P[j][k] - gain[j] * Px[k]) / forgetting_factor

        # Numerical-stability guards.
        # RLS keeps an inverse correlation matrix P that must stay symmetric
        # and positive-definite. Over long signals (tens of thousands of
        # samples) round-off accumulates, P loses positive-definiteness, the
        # weights blow up and the output turns into NaN. Re-symmetrising and
        # adding a small diagonal load (diagonal loading) keeps P well-behaved
        # without materially changing the result.
        for j in range(filter_order):
            for k in range(j + 1, filter_order):
                avg = 0.5 * (P[j][k] + P[k][j])
                P[j][k] = avg
                P[k][j] = avg
            if P[j][j] < DIAG_LOAD:
                P[j][j] = DIAG_LOAD

        # Guard against a non-finite error signal poisoning the recursion.
        if not math.isfinite(error):
            for j in range(filter_order):
                weights[j] = 0.0
                for k in range(filter_order):
                    P[j][k] = 0.0
            for j in range(filter_order):
                P[j][j] = 1.0 / delta
            cleaned_signal[i] = 0.0
            error_signal[i] = 0.0
            continue

        # Store weights periodically
        if i % 1000 == 0:
            weights_history.append(list(weights))

        # Report progress so a multi-minute run does not look hung
        if verbose and n > 0 and (i % progress_step == 0):
            pct = 100.0 * i / n
            print(f"\r    RLS: {pct:5.1f}%  ({i}/{n})", end="", flush=True)

    if verbose:
        print("\r    RLS: 100.0%  done" + " " * 20, flush=True)

    return cleaned_signal, error_signal, weights_history


# ============================================================================
# Signal Processing Utilities
# ============================================================================

def calculate_snr(clean_signal, noisy_signal):
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = sum(s * s for s in clean_signal) / len(clean_signal)
    noise = [noisy_signal[i] - clean_signal[i] for i in range(len(clean_signal))]
    noise_power = sum(n * n for n in noise) / len(noise)

    if noise_power == 0:
        return float('inf')
    if signal_power == 0 or not math.isfinite(noise_power) or not math.isfinite(signal_power):
        return float('nan')

    snr = 10 * math.log10(signal_power / noise_power)
    return snr


def calculate_mse(signal1, signal2):
    """Calculate Mean Squared Error between two signals."""
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length")
    
    mse = sum((signal1[i] - signal2[i]) ** 2 for i in range(len(signal1))) / len(signal1)
    return mse


def normalize_signal(signal):
    """Normalize signal to -1.0 to 1.0 range."""
    max_val = max(abs(s) for s in signal)
    if max_val == 0:
        return signal
    return [s / max_val for s in signal]


# ============================================================================
# Visualization Functions (text-based and data export)
# ============================================================================

def save_plot_data(filename, *signals, labels=None):
    """
    Save signal data to a text file for external plotting.
    Each column represents a signal.
    """
    if labels is None:
        labels = [f"Signal_{i+1}" for i in range(len(signals))]
    
    with open(filename, 'w') as f:
        # Write header
        f.write("Sample\t" + "\t".join(labels) + "\n")
        
        # Write data
        max_len = max(len(s) for s in signals)
        for i in range(max_len):
            f.write(f"{i}")
            for signal in signals:
                if i < len(signal):
                    f.write(f"\t{signal[i]:.6f}")
                else:
                    f.write("\t")
            f.write("\n")
    
    print(f"Plot data saved to {filename}")


def print_statistics(original, corrupted, cleaned_lms, cleaned_rls):
    """Print signal statistics and performance metrics."""
    print("\n" + "="*60)
    print("SIGNAL STATISTICS AND PERFORMANCE METRICS")
    print("="*60)
    
    print(f"\nOriginal signal length: {len(original)} samples")
    
    # Calculate SNR
    snr_corrupted = calculate_snr(original, corrupted)
    snr_lms = calculate_snr(original, cleaned_lms)
    snr_rls = calculate_snr(original, cleaned_rls)
    
    print(f"\nSignal-to-Noise Ratio (SNR):")
    print(f"  Corrupted signal: {snr_corrupted:.2f} dB")
    print(f"  LMS cleaned:      {snr_lms:.2f} dB (improvement: {snr_lms - snr_corrupted:.2f} dB)")
    print(f"  RLS cleaned:      {snr_rls:.2f} dB (improvement: {snr_rls - snr_corrupted:.2f} dB)")
    
    # Calculate MSE
    mse_corrupted = calculate_mse(original, corrupted)
    mse_lms = calculate_mse(original, cleaned_lms)
    mse_rls = calculate_mse(original, cleaned_rls)
    
    print(f"\nMean Squared Error (MSE):")
    print(f"  Corrupted signal: {mse_corrupted:.6f}")
    print(f"  LMS cleaned:      {mse_lms:.6f} (reduction: {(1 - mse_lms/mse_corrupted)*100:.1f}%)")
    print(f"  RLS cleaned:      {mse_rls:.6f} (reduction: {(1 - mse_rls/mse_corrupted)*100:.1f}%)")
    
    print("\n" + "="*60)


def create_simple_visualization(original, corrupted, cleaned_lms, cleaned_rls, sample_rate):
    """Create a simple text-based visualization of the signals."""
    print("\n" + "="*60)
    print("SIGNAL VISUALIZATION (First 100 samples)")
    print("="*60)
    
    # Show first 100 samples
    num_samples = min(100, len(original))
    
    print("\nSample | Original | Corrupted | LMS Clean | RLS Clean")
    print("-" * 60)
    
    for i in range(0, num_samples, 10):  # Show every 10th sample
        print(f"{i:6d} | {original[i]:8.4f} | {corrupted[i]:9.4f} | {cleaned_lms[i]:9.4f} | {cleaned_rls[i]:9.4f}")
    
    print("\n" + "="*60)
