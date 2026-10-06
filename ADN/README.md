# Noise Cancellation using LMS and RLS Adaptive Filters

A complete implementation of audio noise cancellation using **Least Mean Squares (LMS)** and **Recursive Least Squares (RLS)** adaptive filtering algorithms, built entirely from scratch in Python without external signal processing libraries.

## 🎯 Features

- **From-Scratch Implementation**: All core algorithms implemented without using NumPy, SciPy, or other signal processing libraries
- **WAV File I/O**: Custom WAV file reader and writer supporting 8-bit and 16-bit PCM audio
- **Dual Adaptive Filters**:
  - **LMS (Least Mean Squares)**: Lightweight, gradient-descent based adaptive filter
  - **RLS (Recursive Least Squares)**: More computationally intensive but faster convergence
- **Synthetic Noise Generation**: White, pink, and brown noise generation
- **Performance Metrics**: SNR improvement and MSE calculations
- **Visualization Support**: Data export for external plotting tools

## 📁 Project Structure

```
ADN/
├── noise_cancellation.py  # Core implementation (filters, WAV I/O, utilities)
├── main.py                # Main execution script
├── visualize.py           # Visualization script (requires matplotlib)
├── audio.wav              # Input audio file (generated if not present)
├── corrupted.wav          # Output: Audio with added noise
├── cleaned_LMS.wav        # Output: LMS-filtered audio
├── cleaned_RLS.wav        # Output: RLS-filtered audio
├── signal_comparison.txt  # Data: Signal comparisons for plotting
└── error_signals.txt      # Data: Error signals for analysis
```

## 🚀 Quick Start

### Basic Usage

Simply run the main script:

```bash
python main.py
```

This will:
1. Load `audio.wav` (or generate a test audio file if not present)
2. Add synthetic noise to create `corrupted.wav`
3. Apply LMS filter to create `cleaned_LMS.wav`
4. Apply RLS filter to create `cleaned_RLS.wav`
5. Display performance metrics and statistics
6. Export data for visualization

### With Visualization

If you have matplotlib installed:

```bash
python visualize.py
```

This creates:
- `noise_cancellation_results.png` - Time-domain signal comparisons
- `frequency_analysis.png` - Frequency domain analysis

## 🔧 Configuration

Edit the parameters in `main.py`:

```python
# Filter parameters
FILTER_ORDER = 32              # Number of filter coefficients
LMS_STEP_SIZE = 0.01          # LMS learning rate (mu)
RLS_FORGETTING_FACTOR = 0.99  # RLS forgetting factor (lambda)
RLS_DELTA = 1.0               # RLS initialization parameter

# Noise parameters
NOISE_LEVEL = 0.15            # Amplitude of added noise
NOISE_TYPE = 'white'          # 'white', 'pink', or 'brown'
```

## 📊 Algorithm Details

### LMS (Least Mean Squares)

The LMS algorithm updates filter weights using gradient descent:

```
w(n+1) = w(n) + μ * e(n) * x(n)
```

Where:
- `w(n)` = filter weights at time n
- `μ` = step size (learning rate)
- `e(n)` = error signal
- `x(n)` = input signal vector

**Advantages**:
- Low computational complexity: O(N) per sample
- Simple implementation
- Stable for small step sizes

**Disadvantages**:
- Slower convergence
- Performance depends on step size selection

### RLS (Recursive Least Squares)

The RLS algorithm minimizes the weighted least squares cost function:

```
w(n+1) = w(n) + g(n) * e(n)
g(n) = P(n) * x(n) / (λ + x^T(n) * P(n) * x(n))
P(n+1) = (P(n) - g(n) * x^T(n) * P(n)) / λ
```

Where:
- `g(n)` = gain vector
- `P(n)` = inverse correlation matrix
- `λ` = forgetting factor (0 < λ ≤ 1)

**Advantages**:
- Faster convergence than LMS
- Better tracking of time-varying systems
- Superior noise reduction performance

**Disadvantages**:
- Higher computational complexity: O(N²) per sample
- Requires more memory for P matrix
- Potential numerical stability issues

## 📈 Performance Metrics

The system calculates and displays:

1. **Signal-to-Noise Ratio (SNR)**:
   ```
   SNR = 10 * log10(Signal Power / Noise Power)
   ```
   Higher is better. Improvement shown for both filters.

2. **Mean Squared Error (MSE)**:
   ```
   MSE = (1/N) * Σ(original[i] - cleaned[i])²
   ```
   Lower is better. Percentage reduction shown.

## 🎵 Audio File Format

The implementation supports WAV files with:
- **Sample rates**: Any (tested with 44100 Hz)
- **Bit depth**: 8-bit or 16-bit PCM
- **Channels**: Mono or stereo (stereo converted to mono)

## 📝 Example Output

```
======================================================================
NOISE CANCELLATION USING LMS AND RLS ADAPTIVE FILTERS
======================================================================

Step 1: Loading original audio...
  Sample rate: 44100 Hz
  Channels: 1
  Samples: 132300

Step 2: Adding synthetic white noise...
  Noise level: 0.15
  Saved corrupted audio to 'corrupted.wav'

Step 3: Applying LMS adaptive filter...
  Filter order: 32
  Step size (mu): 0.01
  Saved LMS-cleaned audio to 'cleaned_LMS.wav'

Step 4: Applying RLS adaptive filter...
  Filter order: 32
  Forgetting factor (lambda): 0.99
  Saved RLS-cleaned audio to 'cleaned_RLS.wav'

============================================================
SIGNAL STATISTICS AND PERFORMANCE METRICS
============================================================

Signal-to-Noise Ratio (SNR):
  Corrupted signal: 5.23 dB
  LMS cleaned:      18.45 dB (improvement: 13.22 dB)
  RLS cleaned:      22.67 dB (improvement: 17.44 dB)

Mean Squared Error (MSE):
  Corrupted signal: 0.012500
  LMS cleaned:      0.002134 (reduction: 82.9%)
  RLS cleaned:      0.000987 (reduction: 92.1%)
```

## 🔬 Technical Implementation Details

### WAV File I/O

Custom implementation reads and writes WAV files by:
1. Parsing RIFF/WAVE headers
2. Reading fmt chunk for audio parameters
3. Locating and reading data chunk
4. Converting PCM samples to normalized float (-1.0 to 1.0)

### Noise Generation

- **White Noise**: Uniform random distribution
- **Pink Noise**: 1/f spectrum using recursive filtering
- **Brown Noise**: 1/f² spectrum using higher alpha value

### Filter Implementation

Both filters use:
- Circular buffering for input samples
- Dot product for filter output calculation
- Adaptive weight updates based on error signal

## 🎓 Educational Value

This project demonstrates:
- Digital signal processing fundamentals
- Adaptive filtering theory
- Binary file I/O in Python
- Algorithm implementation without libraries
- Performance analysis and metrics

## 📦 Dependencies

**Core functionality** (no dependencies):
- Python 3.6+ (uses only `struct`, `math`, `random` from standard library)

**Optional** (for visualization):
- matplotlib (for creating plots)

## 🛠️ Troubleshooting

**Issue**: "Input file 'audio.wav' not found"
- **Solution**: The script will automatically generate a test audio file

**Issue**: Poor noise cancellation performance
- **Solution**: Adjust filter parameters:
  - Increase `FILTER_ORDER` for more complex noise patterns
  - Adjust `LMS_STEP_SIZE` (try 0.001 to 0.1)
  - Modify `NOISE_LEVEL` to match your audio

**Issue**: RLS filter is slow
- **Solution**: This is expected due to O(N²) complexity. Reduce `FILTER_ORDER` or use LMS for faster processing

## 📚 References

- Haykin, S. (2002). *Adaptive Filter Theory*. Prentice Hall.
- Widrow, B., & Stearns, S. D. (1985). *Adaptive Signal Processing*. Prentice Hall.
- Diniz, P. S. R. (2012). *Adaptive Filtering: Algorithms and Practical Implementation*. Springer.

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

This is an educational implementation. Feel free to:
- Experiment with different filter parameters
- Add new noise types
- Implement additional adaptive filter algorithms (NLMS, APA, etc.)
- Improve visualization capabilities

## ✨ Acknowledgments

Implemented from scratch following adaptive filtering theory and digital signal processing principles.

---

**Note**: This implementation prioritizes clarity and educational value over performance. For production use, consider libraries like NumPy, SciPy, or specialized DSP frameworks.
