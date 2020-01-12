import librosa
import os
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math


def gauss(x, m, s):
    return np.exp((-(x-m)**2)/(2*s*s))

def gauss_mx(x, m, s):
    return np.exp((-(np.minimum(x, m)-m)**2)/(2*s*s))

def thresh(x, th):
    return np.array(x) > th

def find_signal(smp, sr, nd_softness = 1.5, nd_threshold = 0.05, nd_timesmooth = 0.5, draw = False):
    vv = librosa.onset.onset_strength(smp, sr, n_fft=2048, hop_length=512)
    vv = np.convolve(vv, np.ones(50)/50., mode = 'same')
    vv[0:50] = np.ones(50) * vv[51]
    vv[-50:] = np.ones(50) * vv[-51]
    plt.show()

    hh, be = np.histogram(vv, bins=10)
    peaks, _ = scipy.signal.find_peaks(hh)

    if len(peaks) == 0:
        raise Exception('No signal volume peak found')

    pw = scipy.signal.peak_widths(hh, peaks, rel_height=0.5)
    pwL = pw[2][-1]
    pwR = pw[3][-1]

    speak_center = (be[peaks[-1]] + be[peaks[-1]+1])/2
    speak_lpos = be[math.floor(pwL)] + (be[math.ceil(pwL)] - be[math.floor(pwL)])*(pwL - math.floor(pwL))
    speak_rpos = be[math.floor(pwR)] + (be[math.ceil(pwR)] - be[math.floor(pwR)])*(pwR - math.floor(pwR))
    speak_center = np.mean([(speak_rpos + speak_lpos)/2, speak_center])
    speak_sigma = nd_softness * (speak_rpos - speak_lpos) / 2.355

    noise_prob_chunk = np.array(gauss_mx(vv, speak_center, speak_sigma)) # Noise probability in 512-length chunks
    noise_prob_chunk_th = thresh(noise_prob_chunk, nd_threshold)

    nd_timesmooth = round(nd_timesmooth * sr / 512.)
    # noise_prob_chunk = np.convolve(noise_prob_chunk, np.ones(nd_timesmooth)/nd_timesmooth, mode = 'same')
    noise_prob_chunk_th = stretch_threshold(noise_prob_chunk_th, nd_timesmooth)

    noise_prob = []
    for v in noise_prob_chunk_th:
        noise_prob.append(np.ones(512) * v)

    noise_prob = np.array(noise_prob).flatten()
    noise_prob.resize(len(smp))
    noise_prob = np.convolve(noise_prob, np.ones(512)/512, mode = 'same')
    noise_prob_th = thresh(noise_prob.copy(), 0.5)

    if draw:
        fig = plt.figure(figsize=(8,6))

        ax = fig.add_subplot(2, 2, 1)
        plt.plot(vv)
        plt.plot(noise_prob_chunk)
        plt.plot(noise_prob_chunk_th)

        gg = np.arange(0., 3., 0.1)
        gs = gauss_mx(gg, speak_center, speak_sigma) * hh[peaks[-1]]
        ax = fig.add_subplot(2, 2, 2)
        ax.hist(vv, bins=10)
        ax.plot(gg, gs)

        ax = fig.add_subplot(2, 2, 3)
        ax.plot(gauss_mx(vv, speak_center, speak_sigma))
        ax.plot(vv)

        ax = fig.add_subplot(2, 2, 4)
        ax.plot(smp)
        ax.plot(noise_prob)
        ax.plot(noise_prob_th)

        plt.show()

    return noise_prob_th

def build_gate_filter_bank(sr, fmin = 10, fmax = 20000, nfilt = 10):
    freqs = np.geomspace(fmin, fmax, num=nfilt+1)
    filters = [scipy.signal.butter(1, [freqs[i], freqs[i+1]], 'bandpass', fs=sr, output='sos') for i in range(len(freqs)-1)]
    return filters, freqs

def apply_gate_filter_bank(smp, filters):
    fs = []
    for f in filters:
        # Apply 2 times! LR2 filter!
        ss = scipy.signal.sosfilt(f, smp)
        ss = scipy.signal.sosfilt(f, smp)
        fs.append(ss)
    return np.array(fs)

def moving_average(x, n, padding = True):
    n = round(n/2)*2
    cumsum = np.cumsum(x)
    smp = (cumsum[n:] - cumsum[:-n]) / float(n)
    if padding:
        return np.concatenate([np.ones(int(n/2)) * smp[0], smp, np.ones(int(n/2)) * smp[-1]])
    else:
        return smp

def stretch_threshold(x, s):
    d = (np.array(x) > 0.5).astype(int)
    edge_detect = np.concatenate([[0], d[1:] - d[:-1]])
    all_edges = np.argwhere(edge_detect).flatten()
    if len(all_edges) == 0 or s == 0:
        return x

    for e in all_edges:
        if edge_detect[e] > 0:
            f = max(e-s, 0)
            d[f:e] = np.ones(e - f)
        else:
            f = min(len(d), e+s)
            d[e:f] = np.ones(f-e)
    return d

def signal_volume(smp, sr, window=0.4):
    smp = np.abs(smp)**2
    wlen = round(window * sr)
    smp = moving_average(smp, wlen)
    return np.sqrt(smp)

def avg_signal_volume(smp):
    smp = np.abs(smp) ** 2
    return math.sqrt(np.mean(smp))

def max_signal_volume(smp, sr):
    return np.max(signal_volume(smp, sr))

def attack_release(x, a, r):
    x = np.array(x).astype(int)
    a = int(round(a))
    r = int(round(r))
    a_form = np.linspace(0, 1, num=a)
    r_form = np.linspace(0, -1, num=r)

    edge_detect = np.concatenate([x[1:]-x[:-1], [0]])
    all_edges = np.argwhere(edge_detect).flatten()
    if len(all_edges) == 0:
        return x

    d = np.ones(all_edges[0])*x[0]
    ppos = 0

    for i, e in enumerate(all_edges):
        next_pos = len(x)
        if i != len(all_edges)-1:
            next_pos = all_edges[i+1]
        cur_len = next_pos - e
        if e == 0:
            pd = x[0]
        else:
            pd = d[-1]
        if edge_detect[e] > 0:
            nf = []
            if cur_len > a:
                nf = np.ones(cur_len - a)
            d = np.concatenate([d, np.clip(pd+a_form[0:min(a, cur_len)], 0, 1), nf])
        else:
            nf = []
            if cur_len > r:
                nf = np.zeros(cur_len - r)
            d = np.concatenate([d, np.clip(pd+r_form[0:min(r, cur_len)], 0, 1), nf])
    return d

def compute_AR_freq(x, freqs, x_at_freq = 2000):
    return np.minimum(x * np.ones(len(freqs)-1), (x * x_at_freq) / np.convolve(freqs, [0.5, 0.5], mode='valid'))

def sound_filtered_gate(smp,  sr, thresholds,
                        gate_filters, gate_filters_freqs=None,
                        stretch_gate = 0, draw = False, gate_attack = 0.005, gate_release = 0.4):
    # Make sure that attacks and releases are arrays.
    gate_attack = (np.array([gate_attack])*sr).flatten()
    gate_release = (np.array([gate_release])*sr).flatten()

    if len(gate_attack) != len(gate_filters):
        gate_attack = compute_AR_freq(gate_attack[0], freqs = gate_filters_freqs)
    if len(gate_release) != len(gate_filters):
        gate_release = compute_AR_freq(gate_release[0], freqs = gate_filters_freqs)

    voice_filtered = apply_gate_filter_bank(smp, gate_filters)
    voice_volume = [signal_volume(v, sr, 0.05) for v in voice_filtered]
    voice_volume_th = [thresh(voice_volume[i], thresholds[i]) for i in range(len(voice_volume))]
    if stretch_gate != 0:
        voice_volume_th = [stretch_threshold(v, sr * stretch_gate) for v in voice_volume_th]

    voice_volume_gate = [attack_release(v, gate_attack[i], gate_release[i]) for i, v in enumerate(voice_volume_th)]

    voice_filtered = np.array(voice_filtered) * np.array(voice_volume_gate)
    output = np.sum(voice_filtered, axis=0)
    output = output / np.max(output)

    if draw:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(2, 2, 1)
        ax.plot(smp)
        for v in voice_volume_gate:
            ax.plot(v)
        ax = fig.add_subplot(2, 2, 2)
        for v in voice_volume:
            ax.hist(v)
        ax = fig.add_subplot(2, 2, 3)
        ax.plot(gate_attack / sr)
        ax.plot(gate_release / sr)
        plt.show()

    return output

def process_sound(ifn, channel=-1,
                  gate_filters = None, gate_filters_freqs = None,
                  process_sr = 22050, output_sr = None,
                  nt = 0.5, min_threshold_db = 6,
                  no_silence = False, draw=False):
    # Create gate filters if they are not passed
    if gate_filters is None or gate_filters_freqs is None:
        gate_filters, gate_filters_freqs = build_gate_filter_bank(process_sr, 30, 10000, 10)

    # Load the sound: mono if channel == -1, otherwise channel #channel

    if channel == -1:
        smp, sr = librosa.load(ifn, mono=True)
    else:
        smp, sr = librosa.load(ifn, mono=False)

    # Resample to target sample rate
    if output_sr is None:
        output_sr = sr
    smp = librosa.resample(smp, sr, process_sr)

    if channel != -1:
        smp = smp[channel]

    sr = process_sr

    # HP filter: delete all low frequencies
    hp_filter = scipy.signal.butter(2, 15, 'hp', fs=sr, output='sos')
    smp = scipy.signal.sosfilt(hp_filter, smp)
    smp = smp / np.max(smp)

    # Find noise
    sig_prob = find_signal(smp, sr, draw=draw)
    voice_signal = np.array([smp[i] for i in range(len(smp)) if sig_prob[i] == 1])
    noise_signal = np.array([smp[i] for i in range(len(smp)) if sig_prob[i] == 0])

    if len(noise_signal) == 0:
        raise Exception('No noise found')
    if len(voice_signal) == 0:
        raise Exception('No voice found')

    # Find optimal noise threshold for each frequency band
    noise_filt = apply_gate_filter_bank(noise_signal, gate_filters)
    noise_avg_volume = np.array([avg_signal_volume(v) for v in noise_filt])
    voice_filt = apply_gate_filter_bank(voice_signal, gate_filters)
    voice_avg_volume = np.array([avg_signal_volume(v) for v in voice_filt])


    min_th = 10 ** (min_threshold_db / 20)
    noise_avg_volume *= min_th

    noise_thresholds = noise_avg_volume + np.maximum((voice_avg_volume - noise_avg_volume) * nt, np.zeros(len(gate_filters)))

    if draw:
        plt.clf()
        plt.plot(noise_avg_volume, label = 'Noise volume +' + str(min_threshold_db) + 'dB')
        plt.plot(voice_avg_volume, label = 'Voice volume')
        plt.plot(noise_thresholds, label = 'Threshold')
        plt.legend()
        plt.show()



    if no_silence:
        smp = voice_signal
    output = sound_filtered_gate(smp, sr,
                                 gate_filters=gate_filters, gate_filters_freqs=gate_filters_freqs,
                                 thresholds=noise_thresholds, draw=draw)

    if process_sr != output_sr:
        output = librosa.resample(output, process_sr, output_sr)

    return output


process_sr = 22050
output_sr = 44100

gate_filters, gate_filters_freqs = build_gate_filter_bank(process_sr, 30, 10000, 30)

smp = process_sound('test/in.wav',
              channel = -1,
              gate_filters=gate_filters, gate_filters_freqs = gate_filters_freqs,
              process_sr=process_sr, output_sr = output_sr,
              nt = 0.5, min_threshold_db=6,
              no_silence=True,
              draw = True)

librosa.output.write_wav('test/out.wav', smp, output_sr)
