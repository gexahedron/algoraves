#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import chart_studio as plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as express


def calc_umw(x, y):
    assert(abs(y - 0.5) > 1e-5)
    umw = 0
    thresh = 0.5 - 0.125 ** 0.5
    if y < 0.5:
        if y > thresh:
            umw = 3
    else:
        if y < 1 - thresh:
            umw = 6
        else:
            umw = 9
    min_x = 0.5 - abs(y - 0.5)
    max_x = 0.5 + abs(y - 0.5)
    perc = (x - min_x) / (max_x - min_x)
    if 1 / 3 < perc <= 2 / 3:
        umw += 1
    elif 2 / 3 < perc:
        umw += 2
    return umw


umw_colors = [express.colors.cyclical.Edge[c] for c in [
    15, 14, 13,
    11, 10,  9,
     7,  6,  5,
     3,  2,  1]
]


names = ['МксПрл', 'МксПрл➤МнПро', '', 'МнПро➤МксПрл', 'МнПро']
umwelt_names = [
    'maX-', 'maX=', 'maX+',
    'miX-', 'miX=', 'miX+',
    'maN-', 'maN=', 'maN+',
    'miN-', 'miN=', 'miN+',
]

data = []
slices = 5
temporal = 3


surf_slices = slices * 50
surf_temporal = temporal * 50
umw_surfs_xs = []
umw_surfs_ys = []
umw_surfs_zs = []
for i in range(len(umwelt_names)):
    umw_surfs_xs.append([])
    umw_surfs_ys.append([])
    umw_surfs_zs.append([])
screw_surf_xs = []
screw_surf_ys = []
screw_surf_zs = []
for i in range(surf_slices):
  y = i / (surf_slices - 1)
  angle = -y * math.pi / 2
  c, s = np.cos(angle), np.sin(angle)
  r = np.array([[c, -s], [s, c]])
  length = 0.5
  p0 = r.dot(np.array([-length, -length]))
  p1 = -p0
  p0 += np.array([0.5, 0.5])
  p1 += np.array([0.5, 0.5])
  x_shift = 0.5 - abs(y - 0.5)
  for t in range(surf_temporal):
    perc = t / (surf_temporal - 1)
    p = p0 * (1. - perc) + p1 * perc
    screw_surf_xs.append(p[0])
    screw_surf_ys.append(y)
    screw_surf_zs.append(p[1])
    x = x_shift + perc * (1 - 2 * x_shift)
    z = (1 - y) * x + y * (1 - x)
    umw = calc_umw(x, y)
    umw_surfs_xs[umw].append(x)
    umw_surfs_ys[umw].append(y)
    umw_surfs_zs[umw].append(z)

screw_surf = go.Mesh3d(
    x=screw_surf_xs,
    y=screw_surf_ys,
    z=screw_surf_zs,
    name='parade',
    colorscale='Hot',
    intensity=screw_surf_zs,
    opacity=0.3,
    showscale=False
)
data.append(screw_surf)

for surf_idx in range(len(umwelt_names)):
    surf = go.Mesh3d(
        x=umw_surfs_xs[surf_idx],
        y=umw_surfs_ys[surf_idx],
        z=umw_surfs_zs[surf_idx],
        name='parade',
        color=umw_colors[surf_idx],
        opacity=0.8,
        showscale=False
    )
    data.append(surf)


umwelts = []
umw_slice_colors = [express.colors.cyclical.Edge[c] for c in [
    14, 10, 8, 6, 2]
]
for i in range(slices):
  if i == 2:
      continue
  y = i / (slices - 1)
  ys = [y] * temporal
  xs = []
  zs = []
  x_shift = 0.5 - abs(y - 0.5)
  for t in range(temporal):
    perc = t / (temporal - 1)
    x = x_shift + perc * (1 - 2 * x_shift)
    z = (1 - y) * x + y * (1 - x)
    xs.append(x)
    zs.append(z)
    umwelts.append([x, y])
  xs = np.array(xs)
  ys = np.array(ys)
  zs = np.array(zs)
  color = int(255. * y)
  trace = go.Scatter3d(
      x=xs,
      y=ys,
      z=zs,
      mode='markers',
      marker=dict(
          size=12,
          color=umw_slice_colors[i],
          opacity=0.8
      ),
      name = names[i]
  )
  data.append(trace)
umwelts = np.array(umwelts)
