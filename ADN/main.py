"""
Main script for noise cancellation using LMS and RLS adaptive filters.
"""

import os
from noise_cancellation import (
    read_wav, write_wav, add_noise_to_signal,
    lms_filter, rls_filter, normalize_signal,
    save_plot_data, print_statistics, create_simple_visualization
)


def generate_test_audio(filename, duration=3, sample_rate=44100):
    """
    Generate a simple test audio file with sine waves.
    """
    import math
    
    num_samples = duration * sample_rate
    audio_data = []
    
    # Create a combination of sine waves (musical chord)
    frequencies = [440, 554.37, 659.25]  # A major chord (A4, C#5, E5)
    
    for i in range(num_samples):
        t = i / sample_rate
        sample = 0.0
        for freq in frequencies:
            sample += 0.3 * math.sin(2 * math.pi * freq * t)
        audio_data.append(sample)
    
    # Add some amplitude modulation for interest
    for i in range(num_samples):
        t = i / sample_rate
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 2 * t)  # 2 Hz modulation
        audio_data[i] *= envelope
    
    write_wav(filename, sample_rate, 1, audio_data)
    print(f"Generated test audio file: {filename}")
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples: {num_samples}")


def main():
    """Main execution function."""
    
    print("="*70)
    print("NOISE CANCELLATION USING LMS AND RLS ADAPTIVE FILTERS")
    print("="*70)
    
    # Configuration
    input_file = "audio.wav"
    corrupted_file = "corrupted.wav"
    cleaned_lms_file = "cleaned_LMS.wav"
    cleaned_rls_file = "cleaned_RLS.wav"
    
    # Filter parameters
    FILTER_ORDER = 32
    LMS_STEP_SIZE = 0.01
    RLS_FORGETTING_FACTOR = 0.99
    RLS_DELTA = 1.0
    
    # Noise parameters
    NOISE_LEVEL = 0.15  # Amplitude of noise
    NOISE_TYPE = 'white'  # 'white', 'pink', or 'brown'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"\nInput file '{input_file}' not found.")
        print("Generating a test audio file...")
        generate_test_audio(input_file, duration=3, sample_rate=44100)
        print()
    
    # Step 1: Load original audio
    print("\nStep 1: Loading original audio...")
    try:
        sample_rate, num_channels, original_audio = read_wav(input_file)
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {num_channels}")
        print(f"  Samples: {len(original_audio)}")
        
        # Convert to mono if stereo
        if num_channels == 2:
            print("  Converting stereo to mono...")
            mono_audio = []
            for i in range(0, len(original_audio), 2):
                mono_audio.append((original_audio[i] + original_audio[i+1]) / 2)
            original_audio = mono_audio
            num_channels = 1
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    # Step 2: Add synthetic noise
    print(f"\nStep 2: Adding synthetic {NOISE_TYPE} noise...")
    print(f"  Noise level: {NOISE_LEVEL}")
    corrupted_audio, reference_noise = add_noise_to_signal(
        original_audio, 
        noise_level=NOISE_LEVEL, 
        noise_type=NOISE_TYPE
    )
    
    # Save corrupted audio
    write_wav(corrupted_file, sample_rate, num_channels, corrupted_audio)
    print(f"  Saved corrupted audio to '{corrupted_file}'")
    
    # Step 3: Apply LMS adaptive filter
    print(f"\nStep 3: Applying LMS adaptive filter...")
    print(f"  Filter order: {FILTER_ORDER}")
    print(f"  Step size (mu): {LMS_STEP_SIZE}")
    
    cleaned_lms, error_lms, weights_lms = lms_filter(
        corrupted_audio,
        reference_noise,
        filter_order=FILTER_ORDER,
        step_size=LMS_STEP_SIZE
    )
    
    # Normalize and save
    cleaned_lms = normalize_signal(cleaned_lms)
    write_wav(cleaned_lms_file, sample_rate, num_channels, cleaned_lms)
    print(f"  Saved LMS-cleaned audio to '{cleaned_lms_file}'")
    
    # Step 4: Apply RLS adaptive filter
    print(f"\nStep 4: Applying RLS adaptive filter...")
    print(f"  Filter order: {FILTER_ORDER}")
    print(f"  Forgetting factor (lambda): {RLS_FORGETTING_FACTOR}")
    print(f"  Delta: {RLS_DELTA}")
    
    cleaned_rls, error_rls, weights_rls = rls_filter(
        corrupted_audio,
        reference_noise,
        filter_order=FILTER_ORDER,
        forgetting_factor=RLS_FORGETTING_FACTOR,
        delta=RLS_DELTA
    )
    
    # Normalize and save
    cleaned_rls = normalize_signal(cleaned_rls)
    write_wav(cleaned_rls_file, sample_rate, num_channels, cleaned_rls)
    print(f"  Saved RLS-cleaned audio to '{cleaned_rls_file}'")
    
    # Step 5: Generate statistics and visualizations
    print("\nStep 5: Generating statistics and visualizations...")
    
    # Print performance metrics
    print_statistics(original_audio, corrupted_audio, cleaned_lms, cleaned_rls)
    
    # Create simple visualization
    create_simple_visualization(
        original_audio, 
        corrupted_audio, 
        cleaned_lms, 
        cleaned_rls,
        sample_rate
    )
    
    # Save plot data for external visualization
    # Save first 5000 samples for clarity
    plot_length = min(5000, len(original_audio))
    save_plot_data(
        "signal_comparison.txt",
        original_audio[:plot_length],
        corrupted_audio[:plot_length],
        cleaned_lms[:plot_length],
        cleaned_rls[:plot_length],
        labels=["Original", "Corrupted", "LMS_Cleaned", "RLS_Cleaned"]
    )
    
    # Save error signals
    save_plot_data(
        "error_signals.txt",
        error_lms[:plot_length],
        error_rls[:plot_length],
        labels=["LMS_Error", "RLS_Error"]
    )
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. {corrupted_file} - Audio with added noise")
    print(f"  2. {cleaned_lms_file} - Cleaned using LMS filter")
    print(f"  3. {cleaned_rls_file} - Cleaned using RLS filter")
    print(f"  4. signal_comparison.txt - Signal data for plotting")
    print(f"  5. error_signals.txt - Error signal data for plotting")
    print("\nYou can now listen to the audio files to compare the results!")
    print("="*70)


if __name__ == "__main__":
    main()
