"""TurnPilot's chimes, in the same family as make_beeps.py: decaying sine beeps at musical pitches.

  lane change left    two beeps stepping down, E7 to C#7: the change comes from the left of the pitch pair
  lane change right   two beeps stepping up, C#7 to E7
  lane change abort   three short low beeps on A5
  too fast to turn    a falling minor third, G6 to E6, twice
  handback            three falling notes, G#6 E6 C#6, over the standard disengage pitch range

Run from this directory: python make_turnpilot_beeps.py
"""
import numpy as np
from scipy.io import wavfile

sr = 48000
max_int16 = 2**15 - 1

E7, CS7, GS6, G6, E6, CS6, A5 = 2637.02, 2217.46, 1661.22, 1567.98, 1318.51, 1108.73, 880.0


def harmonic_beep(freq, duration_seconds, decay=5.5e3):
  n = int(sr * duration_seconds)
  x = np.arange(n)
  return np.sin(2 * np.pi * freq * x / sr) * np.exp(-x / decay)


def sequence(*notes, gap=0.0, level=0.8):
  """notes are (freq, duration, decay); each starts after the previous one plus gap."""
  parts = []
  for freq, dur, decay in notes:
    parts.append(harmonic_beep(freq, dur, decay))
    parts.append(np.zeros(int(sr * gap)))
  out = np.concatenate(parts + [np.zeros(int(sr * 0.05))])
  return (max_int16 * level * out / np.max(np.abs(out))).astype(np.int16)


chimes = {
  'tp_lane_change_left.wav': sequence((E7, 0.16, 3.0e3), (CS7, 0.45, 5.5e3), gap=0.02),
  'tp_lane_change_right.wav': sequence((CS7, 0.16, 3.0e3), (E7, 0.45, 5.5e3), gap=0.02),
  'tp_lane_change_abort.wav': sequence((A5, 0.12, 2.0e3), (A5, 0.12, 2.0e3), (A5, 0.30, 4.0e3), gap=0.05),
  'tp_too_fast_turn.wav': sequence((G6, 0.18, 4.0e3), (E6, 0.35, 5.5e3), (G6, 0.18, 4.0e3), (E6, 0.45, 5.5e3), gap=0.04),
  'tp_handback.wav': sequence((GS6, 0.22, 5.0e3), (E6, 0.22, 5.0e3), (CS6, 0.6, 8.0e3), gap=0.03),
}

if __name__ == '__main__':
  for name, data in chimes.items():
    wavfile.write(name, sr, data)
    print(name, f'{len(data) / sr:.2f} s')
