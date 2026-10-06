"""
Process real noisy audio using LMS and RLS adaptive filters.
This script is designed for audio that already contains noise.
"""

import os
from noise_cancellation import (
    read_wav, write_wav, lms_filter, rls_filter, 
    normalize_signal, save_plot_data, calculate_mse
)


def estimate_noise_reference(noisy_signal, noise_estimation_method='copy'):
    """
    Estimate a noise reference signal from the noisy audio.
    
    Methods:
    - 'copy': Use the noisy signal itself as reference (assumes noise is correlated)
    - 'delayed': Use a delayed version of the signal
    - 'high_pass': Extract high-frequency components as noise reference
    """
    
    if noise_estimation_method == 'copy':
        # Simple approach: use the signal itself
        return noisy_signal.copy() if hasattr(noisy_signal, 'copy') else list(noisy_signal)
    
    elif noise_estimation_method == 'delayed':
        # Use a delayed version (helps with echo-like noise)
        delay = 10
        return [0.0] * delay + noisy_signal[:-delay]
    
    elif noise_estimation_method == 'high_pass':
        # Simple high-pass filter to extract noise
        alpha = 0.95
        filtered = [0.0] * len(noisy_signal)
        prev_input = 0.0
        prev_output = 0.0
        
        for i in range(len(noisy_signal)):
            filtered[i] = alpha * (prev_output + noisy_signal[i] - prev_input)
            prev_input = noisy_signal[i]
            prev_output = filtered[i]
        
        return filtered
    
    else:
        return noisy_signal


def main():
    """Main execution function for processing real noisy audio."""
    
    print("="*70)
    print("NOISE CANCELLATION FOR REAL NOISY AUDIO")
    print("Using LMS and RLS Adaptive Filters")
    print("="*70)
    
    # Configuration
    input_file = "audio.wav"  # Your noisy audio file
    cleaned_lms_file = "cleaned_LMS.wav"
    cleaned_rls_file = "cleaned_RLS.wav"
    
    # Filter parameters
    FILTER_ORDER = 64  # Increased for better noise reduction
    LMS_STEP_SIZE = 0.005  # Smaller step size for stability
    RLS_FORGETTING_FACTOR = 0.98  # Slightly lower for better adaptation
    RLS_DELTA = 0.1  # Smaller delta for better initialization
    
    # Noise reference estimation method
    NOISE_METHOD = 'copy'  # Options: 'copy', 'delayed', 'high_pass'
    
    print(f"\nConfiguration:")
    print(f"  Filter order: {FILTER_ORDER}")
    print(f"  LMS step size: {LMS_STEP_SIZE}")
    print(f"  RLS forgetting factor: {RLS_FORGETTING_FACTOR}")
    print(f"  Noise estimation method: {NOISE_METHOD}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"\n❌ Error: Input file '{input_file}' not found.")
        print(f"Please upload your noisy audio file and name it '{input_file}'")
        print(f"Or modify the 'input_file' variable in this script.")
        return
    
    # Step 1: Load noisy audio
    print(f"\nStep 1: Loading noisy audio from '{input_file}'...")
    try:
        sample_rate, num_channels, noisy_audio = read_wav(input_file)
        print(f"  ✓ Sample rate: {sample_rate} Hz")
        print(f"  ✓ Channels: {num_channels}")
        print(f"  ✓ Samples: {len(noisy_audio)}")
        print(f"  ✓ Duration: {len(noisy_audio) / sample_rate / num_channels:.2f} seconds")
        
        # Convert to mono if stereo
        if num_channels == 2:
            print("  ✓ Converting stereo to mono...")
            mono_audio = []
            for i in range(0, len(noisy_audio), 2):
                mono_audio.append((noisy_audio[i] + noisy_audio[i+1]) / 2)
            noisy_audio = mono_audio
            num_channels = 1
        
    except Exception as e:
        print(f"  ❌ Error loading audio file: {e}")
        return
    
    # Step 2: Estimate noise reference
    print(f"\nStep 2: Estimating noise reference signal...")
    print(f"  Method: {NOISE_METHOD}")
    
    reference_noise = estimate_noise_reference(noisy_audio, NOISE_METHOD)
    print(f"  ✓ Generated reference signal with {len(reference_noise)} samples")
    
    # Step 3: Apply LMS adaptive filter
    print(f"\nStep 3: Applying LMS adaptive filter...")
    print(f"  This may take a moment for long audio files...")
    
    cleaned_lms, error_lms, weights_lms = lms_filter(
        noisy_audio,
        reference_noise,
        filter_order=FILTER_ORDER,
        step_size=LMS_STEP_SIZE
    )
    
    # Normalize and save
    cleaned_lms = normalize_signal(cleaned_lms)
    write_wav(cleaned_lms_file, sample_rate, num_channels, cleaned_lms)
    print(f"  ✓ Saved LMS-cleaned audio to '{cleaned_lms_file}'")
    
    # Step 4: Apply RLS adaptive filter
    print(f"\nStep 4: Applying RLS adaptive filter...")
    print(f"  This will take longer than LMS due to higher complexity...")
    
    cleaned_rls, error_rls, weights_rls = rls_filter(
        noisy_audio,
        reference_noise,
        filter_order=FILTER_ORDER,
        forgetting_factor=RLS_FORGETTING_FACTOR,
        delta=RLS_DELTA
    )
    
    # Normalize and save
    cleaned_rls = normalize_signal(cleaned_rls)
    write_wav(cleaned_rls_file, sample_rate, num_channels, cleaned_rls)
    print(f"  ✓ Saved RLS-cleaned audio to '{cleaned_rls_file}'")
    
    # Step 5: Generate statistics
    print("\nStep 5: Generating statistics...")
    
    # Calculate MSE (using noisy as reference since we don't have clean original)
    mse_lms = calculate_mse(noisy_audio, cleaned_lms)
    mse_rls = calculate_mse(noisy_audio, cleaned_rls)
    
    print("\n" + "="*70)
    print("PROCESSING STATISTICS")
    print("="*70)
    
    print(f"\nMean Squared Error (vs. original noisy signal):")
    print(f"  LMS cleaned: {mse_lms:.6f}")
    print(f"  RLS cleaned: {mse_rls:.6f}")
    
    # Calculate signal statistics
    noisy_power = sum(s * s for s in noisy_audio) / len(noisy_audio)
    lms_power = sum(s * s for s in cleaned_lms) / len(cleaned_lms)
    rls_power = sum(s * s for s in cleaned_rls) / len(cleaned_rls)
    
    print(f"\nSignal Power:")
    print(f"  Noisy input:  {noisy_power:.6f}")
    print(f"  LMS cleaned:  {lms_power:.6f} ({(lms_power/noisy_power)*100:.1f}% of input)")
    print(f"  RLS cleaned:  {rls_power:.6f} ({(rls_power/noisy_power)*100:.1f}% of input)")
    
    # Save plot data for visualization
    plot_length = min(5000, len(noisy_audio))
    save_plot_data(
        "signal_comparison_real.txt",
        noisy_audio[:plot_length],
        cleaned_lms[:plot_length],
        cleaned_rls[:plot_length],
        labels=["Noisy_Input", "LMS_Cleaned", "RLS_Cleaned"]
    )
    
    save_plot_data(
        "error_signals_real.txt",
        error_lms[:plot_length],
        error_rls[:plot_length],
        labels=["LMS_Error", "RLS_Error"]
    )
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. {cleaned_lms_file} - Cleaned using LMS filter")
    print(f"  2. {cleaned_rls_file} - Cleaned using RLS filter")
    print(f"  3. signal_comparison_real.txt - Signal data for plotting")
    print(f"  4. error_signals_real.txt - Error signal data")
    print("\n💡 Tips for better results:")
    print("  - Try different FILTER_ORDER values (32, 64, 128)")
    print("  - Adjust LMS_STEP_SIZE (0.001 to 0.01)")
    print("  - Experiment with NOISE_METHOD ('copy', 'delayed', 'high_pass')")
    print("  - Listen to both output files and compare quality")
    print("="*70)


if __name__ == "__main__":
    main()
