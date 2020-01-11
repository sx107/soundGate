# soundGate
This is a simple python script to clear the noise out of audio files and remove the silence parts, mostly for voice recordings. This script automatically detects silence in audio files and determines the noise threshold based on it, in case if you need to batch-process a lot of files with different noise levels.

Uses librosa, scipy, numpy and matplotlib to plot itermediate results if needed.

### Usage
The simplest usage example is provided at the end of the script:

```python
process_sr = 22050
output_sr = 16000

gate_filters = build_gate_filter_bank(process_sr, 30, 10000, 10)
smp = process_sound('test/in.wav',
              channel = 0,
              gate_filters=gate_filters,
              process_sr=process_sr, output_sr = output_sr,
              nt = 0.05,
              no_silence=True,
              draw = True)

librosa.output.write_wav('test/out.wav', smp, output_sr)
```

### Main functions
<ul>
  <li><b>find_signal</b>(smp, sr, nd_softness, nd_threshold, nd_timesmooth, nd_timethreshold, draw)<br>
    Returns an array with a length equal to the input sound length in samples (<i>len(smp)</i>) containing 1 for areas with voice, and 0 for silence areas. Parameters:
    <ul>
      <li>smp, sr - input sound sample and it's sample rate</li>
      <li>nd_softness - softness of the noise-voice gate. Lower values will make the gate more harsh, higher values will make the gate softer. 1.0 for standard softness.</li>
      <li>nd_threshold - threshold for the noise-voice gate</li>
      <li>nd_timesmooth - determines how much the signal areas will be enlargen in seconds. If set to 0.5, for example, then approx. 0.5 seconds of silence will be present in front and in the back of each voice area.</li>
      <li>nd_timethreshold - better leave it at 0.1, if too short areas containing voice are not picked up, lower it.</li>
      <li>draw - if set to true, this function will draw some intermediate results using matplotlib.</li>
    </ul>
  </li>
  <li><b>sound_filtered_gate</b>(smp,  sr, thresholds, gate_filters, spread_gate, draw, gate_attack, gate_release)<br>
    Applies a per-frequency band gate to the sound smp. Returns a noise-filtered sound sample. Parameters:
    <ul>
      <li>smp, sr - input sound sample and it's sample rate</li>
      <li>thresholds - an array of thresholds (floats) for each frequency band.</li>
      <li>gate_filters - bank (array) of SOS filters to split the sound into frequency bands. This bank can be built using <b>build_gate_filter_bank</b>.</li>
      <li>spread_gate - how much the areas which passed the gate should be spread in time, in seconds. Pretty much the same as the nd_timesmooth in previous function.</li>
      <li>gate_attack - gate attack, in seconds.</li>
      <li>gate_release - gate release, in seconds.</li>
      <li>draw - if set to true, this script will draw the original sample waveform along with lines indicating  in which areas each frequency band volume passed the gate.</li>
    </ul> 
  </li>
  <li><b>process_sound</b>(ifn, channel, gate_filters, process_sr, output_sr, nt, no_silence, draw)<br>
  Loads the sound ifn, applies a highpass filter, finds noise and voice areas and then processes the sample with the sound_filtered_gate function using automatically found thresholds. Returns an array containing the processed sound. Parameters:
    <ul>
      <li>ifn - input filename.</li>
      <li>channel - which audio channel to use. If set to -1, the sound will be converted to mono.</li>
      <li>gate_filters - see gate_filters parameter for the previous function</li>
      <li>process_sr - processing sample rate</li>
      <li>output_sr - output sample rate. If set to None, the output sample rate will be the same as the input one.</li>
      <li>nt - noise threshold, from 0 to 1. 0 is the noise volume, 1 is the average voice volume.</li>
      <li>no_silence - if set to True, only areas containing voice will be processed and returned.</li>
      <li>draw - whether to set the draw to true for previous two functions.</li>
    </ul>
  </li>
</ul>

### How it works
#### find_signal()
First of all, using `librosa.onset.onset_strength()` onset volume is found in each time frame of 512 samples. This array is then smoothed using moving average with length of 50 samples. Then, a histogram with 10 bins is build based in this data. Usually it contains two peaks: one near zero (silence) and one a bit higher (voice). The second peak's position is found using `scipy.signal.find_peaks`. The peak is "approximated" with a gauss function with sigma set to `nd_softness * fwhm / 2.355`, where fwhm is the full-width half maximum found using `scipy.signal.peak_widths`. Next, the value of this gauss function for each time frame is found. Frames where the obtained value is lower than nd_threshold (silence) are set to 0, others (voice) are set to 1. Finally, the voice areas are stretched by using a moving average with length set to nd_timesmooth in seconds and then applying a threshold again, with threshold being nd_timethreshold. Before returning the final array, it is resized to match the `len(smp)`.

#### sound_filtered_gate()
The input sample first of all is split into different frequency bands using the provided `gate_filters` filter bank using LR2 filter. For each separate frequency band the volume dependence on time is found using `signal_volume()` with 0.1 second window. This array is then thresholded, with threshold being set by `thresholds` for each frequency band separately. Areas containing `1` are then stretched by `spread_gate` seconds, and, finally, an attack-release envelope is built for this gate signal using `attack_release()`. This `attack_release()` function currently works very slowly and is the performance bottleneck. Finally, obtained attack-release enveloped are applied to each frequency band (multiplied with them) and all frequency bands are summed back together, which constitutes the output sound.

#### process_sound()
Load the sound using `librosa.load`, grab the required `channel`, resample it to `process_sr`. A high-pass filter is then applied to filter out any DC offset and the sound is normalized. The sound sample is then split into noise and voice using `find_signal()`. Both noise and voice are split into different frequency bands, average volume is found for each frequency band. Noise thresholds are then determined as `noise_avg_volume + (voice_avg_volume - noise_avg_volume) * nt` for each frequency band. Finally, the sound is processed using `sound_filtered_gate()` with determined noise thresholds. If `no_silence` is set to `True`, then only areas containing voice (found recently using `find_signal()`) are processed and returned. Before output, the sound is resampled to `output_sr`.

### Utility functions
<ul>
  <li>gauss(value, mean, sigma) - gauss function (normalized to 1 on peak)/li>
  <li>gauss_mx(value, mean, sigma) - same, but returns 1 for all values > mean</li>
  <lithresh(x, th) - applies a certain threshold to an array, returning an array containing either 0 or 1.</li>
  <li>moving_average(x, n) - moving average using numpy.cumsum (implementation taken from https://stackoverflow.com/questions/13728392/moving-average-or-running-mean)</li>
  <li>spread_threshold(x, n) - "spreads" areas in array containing "1" by n elements. Works by applying a moving average and a threshold then</li>
  <li>signal_volume(smp, sr, window) - returns the dependence of signal volume on time. smp is the input sound, sr is the sample rate, window is the volume-averaging window in seconds</li>
  <li>avg_signal_volume(smp) - returns average signal volume.</li>
  <li>max_signal_volume(smp) - returns max signal volume (unused)</li>
  <li>attack_release(x, attack, release) - applies a linear attack and release to an array containing "1" or "0". Attack and release are given in samples, not seconds (in number of array elements). <b>Currently works very slowly. Optimization required.</b></li>
  <li>build_gate_filter_bank(sr, fmin = 10, fmax = 20000, nfilt = 10) - builds a bandpass filter bank to split the sound into different frequency bands. fmin and fmax are minimum/maximum frequencies, nfilt is the number of frequency bands.</li>
  <li>apply_gate_filter_bank(smp, filters) - splits the signal into multiple frequency bands using the filter bank provided in `filters`. Each filter is applied twice.</li>
</ul>
