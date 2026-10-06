# Processing Real Noisy Audio

This guide explains how to use the noise cancellation system with your own noisy audio files.

## Quick Start

1. **Upload your noisy audio file** and name it `audio.wav` in the project directory
2. **Run the processing script**:
   ```bash
   python process_noisy_audio.py
   ```
3. **Listen to the results**:
   - `cleaned_LMS.wav` - LMS filter output
   - `cleaned_RLS.wav` - RLS filter output

## How It Works

When processing real noisy audio (without adding synthetic noise), the adaptive filters need a **noise reference signal**. Since we don't have a separate noise recording, the script estimates the noise reference using one of these methods:

### Noise Estimation Methods

#### 1. Copy Method (Default)
```python
NOISE_METHOD = 'copy'
```
- Uses the noisy signal itself as the reference
- Works well when noise is somewhat correlated with the signal
- Best for: Background hum, stationary noise

#### 2. Delayed Method
```python
NOISE_METHOD = 'delayed'
```
- Uses a time-delayed version of the signal
- Helps with echo-like or reverberant noise
- Best for: Echo, reverberation

#### 3. High-Pass Method
```python
NOISE_METHOD = 'high_pass'
```
- Extracts high-frequency components as noise reference
- Uses a simple high-pass filter
- Best for: High-frequency hiss, white noise

## Tuning Parameters

Edit `process_noisy_audio.py` to adjust these parameters:

### Filter Order
```python
FILTER_ORDER = 64  # Try: 32, 64, 128, 256
```
- **Lower values (32)**: Faster processing, less noise reduction
- **Higher values (128-256)**: Better noise reduction, slower processing
- **Recommended**: Start with 64, increase if needed

### LMS Step Size
```python
LMS_STEP_SIZE = 0.005  # Try: 0.001 to 0.01
```
- **Smaller values (0.001)**: More stable, slower convergence
- **Larger values (0.01)**: Faster convergence, may be unstable
- **Recommended**: 0.005 for speech

### RLS Forgetting Factor
```python
RLS_FORGETTING_FACTOR = 0.98  # Try: 0.95 to 0.99
```
- **Lower values (0.95)**: Faster adaptation to changing noise
- **Higher values (0.99)**: More stable, better for stationary noise
- **Recommended**: 0.98 for speech

## Example Configurations

### For Speech with Background Noise
```python
FILTER_ORDER = 64
LMS_STEP_SIZE = 0.005
RLS_FORGETTING_FACTOR = 0.98
NOISE_METHOD = 'copy'
```

### For Speech with Echo
```python
FILTER_ORDER = 128
LMS_STEP_SIZE = 0.003
RLS_FORGETTING_FACTOR = 0.97
NOISE_METHOD = 'delayed'
```

### For High-Frequency Hiss
```python
FILTER_ORDER = 32
LMS_STEP_SIZE = 0.01
RLS_FORGETTING_FACTOR = 0.99
NOISE_METHOD = 'high_pass'
```

## Workflow

1. **Upload your audio**: Place `audio.wav` in the project folder
2. **Run initial processing**: `python process_noisy_audio.py`
3. **Listen to results**: Compare LMS and RLS outputs
4. **Adjust if needed**: 
   - If noise remains: Increase `FILTER_ORDER`
   - If speech is distorted: Decrease `LMS_STEP_SIZE`
   - If results are unstable: Try different `NOISE_METHOD`
5. **Re-run**: Test with new parameters

## Understanding the Output

### Signal Power Comparison
The script shows signal power for each version:
```
Signal Power:
  Noisy input:  0.012500
  LMS cleaned:  0.008750 (70.0% of input)
  RLS cleaned:  0.007500 (60.0% of input)
```

- **Lower percentage** = More noise removed (but check if speech is preserved!)
- **Too low** = Speech may be attenuated

### Error Signals
- Saved to `error_signals_real.txt`
- Shows how the filters converge over time
- Smaller error = better noise cancellation

## Limitations

⚠️ **Important Notes**:

1. **No Ground Truth**: Since we don't have the original clean audio, we can't calculate exact SNR improvement
2. **Noise Reference Quality**: The quality of noise cancellation depends heavily on the noise reference estimation
3. **Speech Distortion**: Aggressive filtering may distort speech - always listen to the results
4. **Stationary Noise Works Best**: Adaptive filters work best with consistent background noise

## Troubleshooting

### Problem: Output sounds worse than input
**Solution**: 
- Reduce `FILTER_ORDER` to 32
- Decrease `LMS_STEP_SIZE` to 0.001
- Try different `NOISE_METHOD`

### Problem: Noise is still present
**Solution**:
- Increase `FILTER_ORDER` to 128 or 256
- Try `NOISE_METHOD = 'high_pass'`
- Increase processing time (RLS is better but slower)

### Problem: Speech sounds muffled
**Solution**:
- Decrease `FILTER_ORDER`
- Use `NOISE_METHOD = 'copy'`
- Reduce `LMS_STEP_SIZE`

### Problem: Processing is too slow
**Solution**:
- Use LMS only (comment out RLS section)
- Reduce `FILTER_ORDER` to 32
- Process shorter audio segments

## Advanced: Custom Noise Reference

If you have a separate recording of just the noise, you can use it directly:

```python
# In process_noisy_audio.py, after loading noisy_audio:
noise_sample_rate, noise_channels, reference_noise = read_wav("noise_only.wav")

# Make sure it's the same length
if len(reference_noise) < len(noisy_audio):
    # Repeat noise to match length
    reference_noise = reference_noise * (len(noisy_audio) // len(reference_noise) + 1)
reference_noise = reference_noise[:len(noisy_audio)]
```

This will give the best results!

## Next Steps

1. Upload your `audio.wav` file
2. Run `python process_noisy_audio.py`
3. Listen to the outputs
4. Adjust parameters if needed
5. Share your results!
