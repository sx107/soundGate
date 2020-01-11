# soundGate
This is a simple python script to clear the noise out of audio files and remove the silence parts, mostly for voice recordings. This script automatically detects silence in audio files and determines the noise threshold based on it, in case if you need to batch-process a lot of files with different noise levels.

Uses librosa, scipy, numpy and matplotlib to plot itermediate results if needed.

<h3>Main functions</h3>
<ul>
  <li><b>find_signal</b>(smp, sr, nd_softness, nd_threshold, nd_timesmooth, nd_timethreshold, draw)<br>
    Returns an array with a length equal to the input sound length in samples (<i>len(smp)</i>) containing 1 for areas with voice, and 0 for silence areas. Parameters:
    <ul>
      <li>smp, sr - input sound sample and it's sample rate</li>
      <li>nd_softness - softness of the noise-voice gate. Lower values will make the gate more harsh, higher values will make the gate softer. 1.0 for standard softness.</li>
      <li>nd_threshold - threshold for the noise-voice gate</li>
      <li>nd_timesmooth - determines how much the signal areas will be enlargen in seconds. If set to 0.5, for example, then approx. 0.5 seconds of silence will be present in front and in the back of each voice area.</li>
      <li>nd_timethreshold - better leave it at 0.1, if too short areas containing voice are not picked up, lower it.</li>
      <li>draw - if set to true, this function will draw some intermediate results using matplotlib. See "itermediate results plots"</li>
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
  <li><b>process_sound</b>(ifn, channel, gate_filters, process_sr, output_sr, nt, no_silence, draw)</li>
</ul>

<h3>Utility functions</h3>
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
</ul>
