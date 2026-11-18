# Original calm ambient piano piece for the housewarming invitation.
# Composed here (melody, chords and sound are all synthesised; no samples, no existing song).
# Usage: python compose-music.py out.wav   ->  then encode to assets/music.mp3 with ffmpeg
import sys, wave, numpy as np

SR = 44100
BPM = 62
BEAT = 60 / BPM
BARS = 16
L = int(BARS * 4 * BEAT * SR)
TAIL = int(6 * SR)
outL = np.zeros(L + TAIL)
outR = np.zeros(L + TAIL)
rng = np.random.default_rng(11)
hz = lambda m: 440 * 2 ** ((m - 69) / 12)


def add(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i < 0:                                   # notes nudged before 0 lose their first few ms
        sig, i = sig[-i:], 0
    n = min(len(sig), len(outL) - i)
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    outL[i:i + n] += sig[:n] * gain * l * 1.414
    outR[i:i + n] += sig[:n] * gain * r * 1.414


def piano(m, vel=0.5, length=7.0):
    """Soft felt-piano tone: stretched partials, each decaying at its own rate."""
    f0 = hz(m)
    n = int(length * SR)
    t = np.arange(n) / SR
    B = 0.00035                                  # string inharmonicity
    reg = np.clip((72 - m) / 30, 0, 1)            # lower notes ring longer
    sig = np.zeros(n)
    for k in range(1, 14):
        fk = f0 * k * np.sqrt(1 + B * k * k)
        if fk > 9000:
            break
        amp = (1 / k ** 1.25) * np.exp(-0.18 * k * (1 - vel * 0.6))
        T = (2.2 + 3.5 * reg) / (1 + 0.45 * (k - 1))
        sig += amp * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-t / T)
    sig += 0.5 * (1 / 1.0) * np.sin(2 * np.pi * f0 * 1.0007 * t) * np.exp(-t / (3 + 4 * reg)) * 0.35  # chorus string
    att = np.clip(t / 0.006, 0, 1)
    knock = rng.normal(0, 1, n) * np.exp(-t / 0.012) * 0.03     # soft hammer felt
    knock = np.convolve(knock, np.ones(20) / 20, "same")
    tail = np.clip((length - t) / 0.8, 0, 1)
    return (sig * att + knock) * tail * vel


# ---- harmony: left-hand broken chords (one note per beat, pedal down) ----
LH = {
    "G": [43, 50, 59, 66], "Bm": [47, 54, 62, 69], "C": [48, 55, 64, 71], "Am": [45, 52, 60, 67],
    "Em": [40, 47, 55, 62], "Dsus": [38, 45, 55, 57], "G/B": [47, 50, 55, 62],
}
prog = ["G", "Bm", "C", "Am", "G", "Em", "C", "Dsus", "C", "G/B", "Am", "Dsus", "Em", "C", "Am", "G"]
for b, c in enumerate(prog):
    for i, m in enumerate(LH[c]):
        human = rng.uniform(0.0, 0.02)
        add(piano(m, vel=0.42 if i == 0 else 0.27, length=6.5), (b * 4 + i) * BEAT + human,
            pan=-0.25 + 0.12 * i, gain=0.55)

# ---- melody (right hand), sparse and singing ----
mel = [
    (71, 2), (74, 2),            (78, 3), (76, 1),          (74, 2), (71, 2),          (72, 2), (69, 1), (71, 1),
    (74, 4),                     (79, 2), (78, 1), (76, 1), (76, 2), (74, 1), (71, 1), (69, 4),
    (None, 1), (67, 1), (71, 1), (76, 1),  (74, 3), (71, 1),  (72, 2), (76, 2),          (74, 4),
    (71, 1), (76, 1), (79, 2),   (78, 2), (76, 2),          (72, 1.5), (71, .5), (69, 2), (67, 4),
]
pos = 0.0
for m, beats in mel:
    if m is not None:
        add(piano(m, vel=0.5, length=min(7.5, beats * BEAT + 4)), pos * BEAT + rng.uniform(0, 0.02),
            pan=0.18, gain=0.62)
        add(piano(m + 12, vel=0.18, length=4), pos * BEAT + 0.004, pan=0.3, gain=0.18)   # soft octave shimmer
    pos += beats

# ---- warm pad under everything (very quiet) ----
PAD = {"G": [55, 59, 62, 66], "Bm": [54, 59, 62, 66], "C": [55, 60, 64, 67], "Am": [57, 60, 64, 67],
       "Em": [55, 59, 62, 64], "Dsus": [55, 57, 62, 64], "G/B": [55, 59, 62, 67]}
for b, c in enumerate(prog):
    dur = 4 * BEAT + 1.2
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * hz(m) * t) + 0.25 * np.sin(4 * np.pi * hz(m) * t) for m in PAD[c])
    e = np.minimum(np.clip(t / 1.4, 0, 1), np.clip((dur - t) / 1.2, 0, 1))
    add(s * e * (1 + 0.05 * np.sin(2 * np.pi * 0.2 * t)) / 4, max(0.0, b * 4 * BEAT - 0.3), pan=0.0, gain=0.05)

# ---- space: long soft hall + a gentle echo ----
def ir(seed, secs=4.2, damp=1.6):
    n = int(secs * SR)
    r = np.random.default_rng(seed).normal(0, 1, n) * np.exp(-np.arange(n) / SR * damp)
    return np.convolve(r, np.ones(30) / 30, "same")


def conv(x, h):
    N = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(h, N), N)[:len(x)]


wetL, wetR = conv(outL, ir(1)), conv(outR, ir(2))
wetL *= np.std(outL) / np.std(wetL)
wetR *= np.std(outR) / np.std(wetR)
d = int(0.75 * BEAT * SR)
echoL = np.concatenate([np.zeros(d), outR[:-d]]) * 0.18
echoR = np.concatenate([np.zeros(d), outL[:-d]]) * 0.18
mixL = outL * 0.62 + wetL * 0.55 + echoL
mixR = outR * 0.62 + wetR * 0.55 + echoR

# ---- seamless loop: fold the ringing tail back onto the start ----
mixL[:TAIL] += mixL[L:L + TAIL]
mixR[:TAIL] += mixR[L:L + TAIL]
st = np.stack([mixL[:L], mixR[:L]], 1)
st = np.tanh(st / np.max(np.abs(st)) * 1.05) * 0.85
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
print(sys.argv[1], round(L / SR, 2), "s")
