import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.fft import fft
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 1. Get audio file paths from user
# -------------------------------
print("Enter the path or name of the first audio file (with .wav, .mp3, etc.):")
AUDIO1_PATH = input().strip()

print("Enter the path or name of the second audio file:")
AUDIO2_PATH = input().strip()

# -------------------------------
# 2. Load and preprocess audio (mono, same sample rate)
# -------------------------------
def load_audio_mono(path, sr_target=16000):
    y, sr = librosa.load(path, sr=None)
    # Convert to mono if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    # Resample to common sample rate
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
    return y, sr_target

y1, sr1 = load_audio_mono(AUDIO1_PATH)
y2, sr2 = load_audio_mono(AUDIO2_PATH)

print(f"Audio 1 shape: {y1.shape}, sample rate: {sr1}")
print(f"Audio 2 shape: {y2.shape}, sample rate: {sr2}")

# -------------------------------
# 3. Compute magnitude spectra (frequency domain)
# -------------------------------
def compute_spectrum(y, sr, n_fft=2048):
    # Take a central chunk to keep size similar
    L = min(len(y1), len(y2))
    y = y[:L]
    # Zero‑pad to n_fft if needed
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    else:
        y = y[:n_fft]
    # FFT and magnitude
    Y = fft(y)
    Y_mag = np.abs(Y[:n_fft//2])  # only positive frequencies
    freqs = np.linspace(0, sr/2, n_fft//2)
    return freqs, Y_mag

freqs1, mag1 = compute_spectrum(y1, sr1)
freqs2, mag2 = compute_spectrum(y2, sr2)

# -------------------------------
# 4. Plot both spectra together (frequency‑domain graph)
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(freqs1, mag1, label="Audio 1", alpha=0.7)
plt.plot(freqs2, mag2, label="Audio 2", alpha=0.7)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency‑Domain Magnitude Spectra (Both Audio Files)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("frequency_comparison.png", dpi=150)
plt.show()

# -------------------------------
# 5. Compute numerical similarity score
# -------------------------------
# Normalize magnitude vectors
mag1_norm = mag1 / (np.linalg.norm(mag1) + 1e-10)
mag2_norm = mag2 / (np.linalg.norm(mag2) + 1e-10)

# Reshape for sklearn
mag1_norm = mag1_norm.reshape(1, -1)
mag2_norm = mag2_norm.reshape(1, -1)

# Cosine similarity (between -1 and 1)
similarity = cosine_similarity(mag1_norm, mag2_norm)[0, 0]
print(f"\nFrequency‑domain similarity score (cosine): {similarity:.4f}")

# Normalize to [0,1] where 1 = very similar
sim_score_01 = (similarity + 1) / 2
print(f"Normalized similarity [0,1]: {sim_score_01:.4f}")
