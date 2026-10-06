"""
Visualization script for noise cancellation results.
Creates plots using matplotlib to visualize the signal processing results.
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Install it with: pip install matplotlib")


def load_signal_data(filename):
    """Load signal data from text file."""
    data = {}
    with open(filename, 'r') as f:
        # Read header
        header = f.readline().strip().split('\t')
        labels = header[1:]  # Skip 'Sample' column
        
        # Initialize data lists
        for label in labels:
            data[label] = []
        
        # Read data
        for line in f:
            if line.strip():
                values = line.strip().split('\t')
                for i, label in enumerate(labels):
                    if i + 1 < len(values) and values[i + 1]:
                        data[label].append(float(values[i + 1]))
    
    return data, labels


def create_visualizations():
    """Create comprehensive visualizations of the noise cancellation results."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot create visualizations without matplotlib.")
        return
    
    print("Creating visualizations...")
    
    # Load signal data
    signals, signal_labels = load_signal_data('signal_comparison.txt')
    errors, error_labels = load_signal_data('error_signals.txt')
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Noise Cancellation Results: LMS vs RLS Adaptive Filters', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Time-domain signals (full view)
    ax = axes[0, 0]
    sample_indices = range(len(signals['Original']))
    ax.plot(sample_indices, signals['Original'], label='Original', alpha=0.7, linewidth=0.5)
    ax.plot(sample_indices, signals['Corrupted'], label='Corrupted', alpha=0.7, linewidth=0.5)
    ax.set_title('Original vs Corrupted Signal')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time-domain signals (zoomed)
    ax = axes[0, 1]
    zoom_samples = 500
    ax.plot(range(zoom_samples), signals['Original'][:zoom_samples], 
            label='Original', linewidth=1.5)
    ax.plot(range(zoom_samples), signals['Corrupted'][:zoom_samples], 
            label='Corrupted', alpha=0.7, linewidth=1)
    ax.set_title('Original vs Corrupted Signal (Zoomed)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: LMS Cleaned Signal
    ax = axes[1, 0]
    ax.plot(sample_indices, signals['Original'], label='Original', alpha=0.7, linewidth=0.5)
    ax.plot(sample_indices, signals['LMS_Cleaned'], label='LMS Cleaned', alpha=0.7, linewidth=0.5)
    ax.set_title('LMS Filter Result')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: RLS Cleaned Signal
    ax = axes[1, 1]
    ax.plot(sample_indices, signals['Original'], label='Original', alpha=0.7, linewidth=0.5)
    ax.plot(sample_indices, signals['RLS_Cleaned'], label='RLS Cleaned', alpha=0.7, linewidth=0.5)
    ax.set_title('RLS Filter Result')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Error signals comparison
    ax = axes[2, 0]
    error_indices = range(len(errors['LMS_Error']))
    ax.plot(error_indices, errors['LMS_Error'], label='LMS Error', alpha=0.7, linewidth=0.5)
    ax.plot(error_indices, errors['RLS_Error'], label='RLS Error', alpha=0.7, linewidth=0.5)
    ax.set_title('Error Signals (LMS vs RLS)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: All signals comparison (zoomed)
    ax = axes[2, 1]
    zoom_samples = 500
    ax.plot(range(zoom_samples), signals['Original'][:zoom_samples], 
            label='Original', linewidth=1.5, color='black')
    ax.plot(range(zoom_samples), signals['Corrupted'][:zoom_samples], 
            label='Corrupted', alpha=0.5, linewidth=1)
    ax.plot(range(zoom_samples), signals['LMS_Cleaned'][:zoom_samples], 
            label='LMS Cleaned', alpha=0.7, linewidth=1)
    ax.plot(range(zoom_samples), signals['RLS_Cleaned'][:zoom_samples], 
            label='RLS Cleaned', alpha=0.7, linewidth=1, linestyle='--')
    ax.set_title('All Signals Comparison (Zoomed)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_cancellation_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'noise_cancellation_results.png'")
    
    # Create a second figure for frequency domain analysis
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold')
    
    # Simple FFT approximation (magnitude spectrum)
    def compute_magnitude_spectrum(signal, max_freq=2000):
        """Compute magnitude spectrum of signal."""
        n = len(signal)
        spectrum = []
        freqs = []
        
        # Compute for first max_freq frequency bins
        for k in range(min(max_freq, n // 2)):
            real = 0
            imag = 0
            for i in range(n):
                angle = -2 * 3.14159265359 * k * i / n
                real += signal[i] * (angle).__cos__() if hasattr(angle, '__cos__') else signal[i] * __import__('math').cos(angle)
                imag += signal[i] * (angle).__sin__() if hasattr(angle, '__sin__') else signal[i] * __import__('math').sin(angle)
            
            magnitude = (real * real + imag * imag) ** 0.5
            spectrum.append(magnitude)
            freqs.append(k)
        
        return freqs, spectrum
    
    import math
    
    # Simplified spectrum visualization using windowed energy
    def compute_energy_spectrum(signal, window_size=100):
        """Compute energy in frequency bands."""
        n = len(signal)
        num_bands = min(100, n // window_size)
        energies = []
        
        for band in range(num_bands):
            start = band * window_size
            end = min(start + window_size, n)
            energy = sum(s * s for s in signal[start:end]) / (end - start)
            energies.append(energy)
        
        return list(range(num_bands)), energies
    
    # Plot energy spectra
    for idx, (signal_name, ax) in enumerate([
        ('Original', axes2[0, 0]),
        ('Corrupted', axes2[0, 1]),
        ('LMS_Cleaned', axes2[1, 0]),
        ('RLS_Cleaned', axes2[1, 1])
    ]):
        freqs, spectrum = compute_energy_spectrum(signals[signal_name])
        ax.plot(freqs, spectrum, linewidth=1)
        ax.set_title(f'{signal_name} Signal - Energy Distribution')
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel('Energy')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('frequency_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved frequency analysis to 'frequency_analysis.png'")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    create_visualizations()
