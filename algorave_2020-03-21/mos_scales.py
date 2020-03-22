#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import math
import numpy as np

def same(a, b):
  return abs(a - b) < 0.001

try:
  fifth = float(sys.argv[1])
except Exception as e:
  fifth = 700

try:
  octave = float(sys.argv[2])
except Exception as e:
  octave = 1200

print(fifth, octave)

pitch = 0
pitches = [0]
for i in range(2, 59):
  pitch += fifth
  if pitch > octave:
    pitch -= octave
  pitches.append(pitch)
  if i < 5:
    continue

  all_pitches = pitches + [1200]
  sorted_pitches = sorted(all_pitches)
  intervals = []
  for i_idx in range(1, len(sorted_pitches)):
    intervals.append(sorted_pitches[i_idx] - sorted_pitches[i_idx - 1])

  # check that we have 2, not 3 intervals of different lengths
  unique_intervals = []
  for interval in intervals:
    found_same = False
    for ui in unique_intervals:
      if same(ui, interval):
        found_same = True
        break
    if found_same:
      continue
    unique_intervals.append(interval)
  unique_intervals.sort()
  if same(unique_intervals[0], 0):
    break
  if len(unique_intervals) > 2:
    continue

  s = ''
  for interval in intervals:
    if same(interval, unique_intervals[0]):
      s += 's'
    else:
      assert(same(interval, unique_intervals[1]))
      s += 'L'
  print(i, s)
  pitches_norm = np.array(sorted(pitches)) / 1200 * 12
  p_str = list(pitches_norm)
  print([float('%.2f' % p) for p in p_str])
  #non_empty = []
  #for p in p_str:
  #  if p:
  #    non_empty
  # print(i, intervals)

