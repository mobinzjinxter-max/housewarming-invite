# Original devotional instrumental for the housewarming invite (composed here, no samples).
import sys, wave, numpy as np
SR = 44100; BPM = 70; BEAT = 60/BPM; BARS = 16
L = int(BARS*4*BEAT*SR)
TAIL = int(4*SR)
outL = np.zeros(L+TAIL); outR = np.zeros(L+TAIL)
rng = np.random.default_rng(7)
f = lambda m: 440*2**((m-69)/12)

def add(sig, start, pan=0.0, gain=1.0):
    i = int(start*SR); n = min(len(sig), len(outL)-i)
    l = np.cos((pan+1)*np.pi/4); r = np.sin((pan+1)*np.pi/4)
    outL[i:i+n] += sig[:n]*gain*l*1.414; outR[i:i+n] += sig[:n]*gain*r*1.414

def env(n, a, r, sus=1.0):
    e = np.full(n, sus); A = min(int(a*SR), n); R = min(int(r*SR), n)
    e[:A] = np.linspace(0, sus, A); e[n-R:] *= np.linspace(1, 0, R); return e

# ---- harmonium pad (chords) ----
CH = {"D":[50,57,62,66],"G":[50,55,59,62],"A":[49,52,57,61],"Bm":[47,54,59,62],"Em":[52,55,59,64],
      "D/F#":[54,57,62,66],"Asus":[50,52,57,62]}
prog = ["D","G","A","D","Bm","G","Em","A","G","D/F#","Em","A","D","G","Asus","D"]
for b, c in enumerate(prog):
    t0 = b*4*BEAT; dur = 4*BEAT+0.5; n = int(dur*SR); t = np.arange(n)/SR
    sig = np.zeros(n)
    for m in CH[c]:
        for det in (-0.0018, 0.0018):
            fr = f(m)*(1+det)
            for h in range(1, 9):
                sig += np.sin(2*np.pi*fr*h*t + h) * (1/h**1.35)
    sig *= env(n, 0.7, 0.9) * (1 + 0.04*np.sin(2*np.pi*0.35*t))
    add(sig/ (len(CH[c])*6), t0, pan=-0.15, gain=0.22)

# ---- low drone (D + A) ----
t = np.arange(len(outL))/SR
dr = (np.sin(2*np.pi*f(38)*t) + .6*np.sin(2*np.pi*f(45)*t) + .3*np.sin(2*np.pi*f(50)*t))
dr *= 0.85 + 0.15*np.sin(2*np.pi*t/ (8*BEAT))
outL += dr*0.045; outR += dr*0.045

# ---- bamboo flute melody ----
mel = [ (69,2),(66,1),(69,1), (71,2),(69,1),(67,1), (64,1.5),(66,.5),(67,1),(64,1), (66,3),(None,1),
        (74,2),(73,1),(71,1), (71,2),(69,1),(67,1), (67,1.5),(66,.5),(64,1),(67,1), (69,3),(None,1),
        (71,1),(74,1),(74,1.5),(71,.5), (69,2),(66,1),(69,1), (67,1),(66,1),(64,1),(67,1), (69,2),(73,1),(76,1),
        (78,2),(76,1),(74,1), (71,2),(74,1),(71,1), (69,1.5),(67,.5),(66,1),(64,1), (62,3.6),(None,.4) ]
pos = 0.0
for m, beats in mel:
    if m is not None:
        dur = beats*BEAT*0.97 + 0.12; n = int(dur*SR); tt = np.arange(n)/SR
        vib = 1 + 0.0045*np.sin(2*np.pi*5.2*tt) * np.clip((tt-0.25)/0.4, 0, 1)
        ph = 2*np.pi*np.cumsum(f(m+12)*vib)/SR
        tone = np.sin(ph) + .22*np.sin(2*ph+.3) + .07*np.sin(3*ph)
        breath = rng.normal(0, 1, n); breath = np.convolve(np.convolve(breath, np.ones(24)/24, "same"), np.ones(24)/24, "same") * .05 * np.clip(1 - tt/0.35, 0.25, 1)
        e = env(n, 0.09, 0.22)
        add((tone+breath)*e, pos*BEAT, pan=0.05, gain=0.16)
    pos += beats

# ---- temple / church bell accents ----
def bell(m, start, gain, pan):
    n = int(3.2*SR); tt = np.arange(n)/SR; fr = f(m)
    s = sum(a*np.sin(2*np.pi*fr*r*tt)*np.exp(-tt*d) for r,a,d in [(1,1,1.3),(2.0,.55,2.0),(3.01,.35,3),(4.17,.22,4),(5.43,.12,6)])
    s *= np.clip(tt/0.004, 0, 1); add(s, start, pan, gain)
roots = {"D":74,"G":79,"A":76,"Bm":78,"Em":76,"D/F#":78,"Asus":76}
for b, c in enumerate(prog):
    bell(roots[c]+12, b*4*BEAT, 0.05, -0.45)
    if b % 2 == 1: bell(roots[c]+7+12, b*4*BEAT + 2*BEAT, 0.03, 0.45)
bell(86, 0, 0.07, 0.0)

# ---- hall reverb (stereo convolution) ----
def ir(seed):
    n = int(2.6*SR); r = np.random.default_rng(seed).normal(0,1,n) * np.exp(-np.arange(n)/SR*2.6)
    return np.convolve(r, np.ones(12)/12, "same")
def conv(x, h):
    N = 1 << int(np.ceil(np.log2(len(x)+len(h))))
    return np.fft.irfft(np.fft.rfft(x, N)*np.fft.rfft(h, N), N)[:len(x)]
wetL = conv(outL, ir(1)); wetR = conv(outR, ir(2))
wetL *= np.std(outL)/np.std(wetL); wetR *= np.std(outR)/np.std(wetR)
mixL = outL*0.72 + wetL*0.42; mixR = outR*0.72 + wetR*0.42

# ---- seamless loop: fold the tail back onto the start ----
mixL[:TAIL] += mixL[L:L+TAIL]; mixR[:TAIL] += mixR[L:L+TAIL]
mixL, mixR = mixL[:L], mixR[:L]
st = np.stack([mixL, mixR], 1)
st = np.tanh(st / np.max(np.abs(st)) * 1.1) * 0.82
pcm = (st*32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print(sys.argv[1], round(L/SR,2), "s")
