"""
Spectrogram comparison — the clearest way to *see* noise removal.

A waveform plot shows amplitude over time, so broadband noise just looks like
a fuzzy band. A spectrogram shows energy per frequency over time, so removed
noise shows up as visibly darkened bands.

Important detail: every panel is normalised against the *same* reference
(the corrupted signal's peak). If each panel were normalised against itself,
a cleaned signal would look as bright as the noisy one and you could not
compare them at all.

Requires matplotlib:
    python -m pip install matplotlib

Output:
    spectrogram_comparison.png
"""

import math

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available. Install it with: pip install matplotlib")

from noise_cancellation import (read_wav, add_noise_to_signal, lms_filter,
                                rls_filter, normalize_signal)

# Only the first few seconds: a 20 s spectrogram is unreadable, and this is a
# quick visual check.
PLOT_SECONDS = 2.0
MAX_HZ = 2000          # ANC mainly targets the low band; keep the image short
FILTER_ORDER = 32


def compute_spectrogram(signal, sample_rate, nfft=1024, hop=256):
    """Return (times, freqs, db) using a Hann window. dB is absolute, not per-panel."""
    win = np.hanning(nfft)
    frames = []
    for start in range(0, len(signal) - nfft, hop):
        chunk = np.asarray(signal[start:start + nfft], dtype=float) * win
        spec = np.fft.rfft(chunk)
        frames.append(20 * np.log10(np.abs(spec) + 1e-12))
    if not frames:
        return np.array([]), np.array([]), np.empty((0, 0))

    db = np.vstack(frames)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    times = np.arange(db.shape[0]) * hop / sample_rate
    return times, freqs, db


def main():
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot render spectrogram without matplotlib.")
        return

    print("Loading audio...")
    sample_rate, num_channels, audio = read_wav('audio.wav')
    if num_channels == 2:
        audio = [(audio[i] + audio[i + 1]) / 2 for i in range(0, len(audio), 2)]

    seg = audio[:int(sample_rate * PLOT_SECONDS)]
    print("  using first %.1f s (%d samples)" % (PLOT_SECONDS, len(seg)))

    print("Adding noise...")
    corrupted, noise = add_noise_to_signal(seg, 0.15, 'white')

    print("Running LMS...")
    cleaned_lms, _, _ = lms_filter(corrupted, noise,
                                   filter_order=FILTER_ORDER, step_size=0.01)
    cleaned_lms = normalize_signal(cleaned_lms)

    print("Running RLS (a bit slow, watch the progress bar)...")
    cleaned_rls, _, _ = rls_filter(corrupted, noise,
                                   filter_order=FILTER_ORDER,
                                   forgetting_factor=0.99, delta=1.0, verbose=True)
    cleaned_rls = normalize_signal(cleaned_rls)

    print("Computing spectrograms...")
    panels = [
        ("1. Original (clean reference)", seg),
        ("2. Corrupted (noise added)", corrupted),
        ("3. After LMS filter", cleaned_lms),
        ("4. After RLS filter", cleaned_rls),
    ]

    specs = []
    for _, sig in panels:
        specs.append(compute_spectrogram(sig, sample_rate))

    # One shared colour scale across all panels, taken from the noisy signal.
    # This is what makes the panels visually comparable.
    bin_hz = sample_rate / 1024.0
    max_bin = int(MAX_HZ / bin_hz)
    _, _, ref_db = specs[1]
    vmin = float(np.percentile(ref_db[:max_bin, :], 5))
    vmax = float(np.percentile(ref_db[:max_bin, :], 99.5))
    print("  shared colour scale: %.1f .. %.1f dB" % (vmin, vmax))

    print("Rendering...")
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             constrained_layout=True)
    fig.suptitle('Spectrogram: where the noise was removed\n'
                 '(dark = quiet, bright = loud; 0-%d Hz, first %.0f s)'
                 % (MAX_HZ, PLOT_SECONDS),
                 fontsize=14, fontweight='bold')

    for ax, (title, _), (times, freqs, db) in zip(axes, panels, specs):
        band = db[:max_bin, :]
        im = ax.imshow(band,
                       aspect='auto',
                       origin='lower',
                       extent=[times[0], times[-1] if len(times) else 0, 0, MAX_HZ],
                       cmap='magma',
                       vmin=vmin, vmax=vmax)
        ax.set_ylabel('Hz')
        ax.set_title(title, fontsize=11, loc='left')
        ax.grid(False)

    axes[-1].set_xlabel('Time (s)')
    # Use constrained_layout instead of tight_layout: tight_layout does not
    # know about the colorbar axes and the bar ends up overlapping the plots.
    fig.colorbar(im, ax=axes, label='Level (dB)', pad=0.02, fraction=0.025)
    plt.savefig('spectrogram_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved spectrogram_comparison.png")


if __name__ == '__main__':
    main()
