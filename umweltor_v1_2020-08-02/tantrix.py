#!/usr/bin/env python
# -*- coding: utf-8 -*-

# forked from (and heavily modified afterwards): https://github.com/zvikabh/tantrix
# please contact the original author for license


import numpy as np
import time
from copy import deepcopy

import cairo
from scipy.ndimage import morphology
from random import shuffle
from typing import List, Optional, Tuple


resolve_state = 0
# 0 - start
# 1 - resolved!
# 2 - didn't resolve
# 3 - in the process of resolving
# 4 - signal of stopping resolving

best_possible_board = None
best_possible_placed_pieces = []
best_total_shrink = 0
seed = 0


angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
hex_height = 100
hex_width = hex_height * np.sin(np.pi / 3)
hex_coords = np.asarray([np.sin(angles), np.cos(angles)]).T * hex_height / 2
hex_coords += [hex_width / 2, hex_height / 2]
hex_mid_coords = np.mean([hex_coords, np.roll(hex_coords, 1, axis=0)], axis=0)
hex_mid_coords = hex_mid_coords[[3, 2, 1, 0, 5, 4], :]
hex_shift_mat = np.asarray([[hex_width, hex_width / 2], [0, hex_coords[1, 1]]])


color_table = {
    'R': (0.9,   0, 0),
    'B': (  0,   0, 1),
    'Y': (0.9, 0.8, 0),
    'G': (  0, 0.8, 0)
}

def hexagon(ctx, p, color, do_fill):
    for pair in p:
        ctx.line_to(pair[0], pair[1])
    ctx.close_path()
    ctx.set_line_width(2)
    ctx.set_source_rgb(*color)
    if do_fill:
        ctx.stroke_preserve()
        ctx.set_source_rgb(*color)
        ctx.fill()
    else:
        ctx.stroke()


# function to convert to array
def surface_to_npim(surface):
    """ Transforms a Cairo surface into a numpy array. """
    im = +np.frombuffer(surface.get_data(), np.uint8)
    H,W = surface.get_height(), surface.get_width()
    im.shape = (H, W, 4) # for RGBA
    return im[:, :, :3] # bgr


class Piece(object):
  """Single Tantrix piece.

  Attrs:
    name: Arbitrary nickname.
    colors: String of length 6 indicating the color at each edge.
      e.g., piece #1 is '001212'.
    color_names: map from '012' to 'YBR'
    rotation: Current rotation, an int in range 0-5.
      A value of N indicates that colors should be shifted counterclockwise
      by N, e.g. rotation of 1 converts 'ABCDEF' to 'FABCDE'.
    coord: Tuple (row, col) indicating board position, or None if not placed
      on board.
    FIXME: rename colors, color_names
    FIXME: add umwelt, x, y
  """

  def __init__(self, name: str, colors: str, umwelt: int, x=-1.0, y=-1.0, color_names='YGR'):
    self.name = name
    self.colors = colors
    self.ins = dict()
    self.out = dict()
    for idx, c in enumerate(colors):
        if c not in self.ins:
            self.ins[c] = idx
        else:
            self.out[c] = idx
    self.color_names = {'0' : color_names[0], '1' : color_names[1], '2' : color_names[2]}
    self.rotation = 0
    self.coord = None
    self.umwelt = umwelt
    self.x = x
    self.y = y

  def __str__(self):
    colors_str = self.colors[self.rotation:] + self.colors[:self.rotation]
    color_names_str = ''.join([self.color_names[p] for p in colors_str])
    return '%s [%s] [%s] (%f, %f)' % (self.name, colors_str, color_names_str, self.x, self.y)

  def __repr__(self):
    colors_str = self.colors[self.rotation:] + self.colors[:self.rotation]
    color_names_str = ''.join([self.color_names[p] for p in colors_str])
    return '%s [%s] [%s] (%f, %f)' % (self.name, colors_str, color_names_str, self.x, self.y)

  def set_color_names(self, color_names):
    self.color_names = {'0' : color_names[0], '1' : color_names[1], '2' : color_names[2]}

  def get_color(self, direction: int):
    return self.colors[(self.rotation + direction) % 6]

  def get_color_name(self, direction: int):
    return self.color_names[self.colors[(self.rotation + direction) % 6]]

  def get_outgoing_dir(self, incoming_dir: int) -> int:
    color = self.get_color(incoming_dir)
    direction = (self.ins[color] - self.rotation) % 6
    if direction == incoming_dir:
      direction = (self.out[color] - self.rotation) % 6
    return direction


pieces = [
    [Piece('#1', '011022', 0), Piece('#2', '001122', 0)],
    [Piece('#3', '001212', 3)],
    [Piece('#4', '012021', 6)],
    [Piece('#5', '012012', 9)],
]
color_orders = [
    ['GGG'],
    ['YGG', 'GYG', 'GGY',
     'RGG', 'GRG', 'GGR',
     'GRR', 'RGR', 'RRG',
     'GYY', 'YGY', 'YYG'],
    ['YGR', 'YRG', 'RGY', 'RYG', 'GYR', 'GRY']
]

# FIXME: add this scheme as an option
# alternative color scheme:
# color_orders = [
#     ['YYY', 'RRR', 'GGG'],
#     ['YGG', 'GYG', 'GGY',
#      'YRR', 'RYR', 'RRY',
#      'RGG', 'GRG', 'GGR',
#      'RYY', 'YRY', 'YYR',
#      'GRR', 'RGR', 'RRG',
#      'GYY', 'YGY', 'YYG'],
#     ['YGR', 'YRG', 'RGY', 'RYG', 'GYR', 'GRY']
# ]


def draw_surface_from_grid(grid, selected_coord=None):
    page_dims = hex_shift_mat @ grid.shape[::-1] + [hex_width, hex_height]
    page_width, page_height = page_dims

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(page_width), int(page_height))
    ctx = cairo.Context(surface)

    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, page_width, page_height)
    ctx.fill()

    for y, row in enumerate(grid):
      for x, cell in enumerate(row):
        if not cell:
          continue
        back_color = (0, 0, 0)
        if selected_coord is not None:
            if (y, x) == selected_coord:
                back_color = (0.5, 0.5, 0.5)

        transform = np.matmul(hex_shift_mat, [[x], [y]])
        tx = transform[0] + 10
        ty = transform[1] + 10

        ctx.translate(tx, ty)
        hexagon(ctx, hex_coords, back_color, True)
        ctx.translate(-tx, -ty)

        # Colors
        completed_dirs = set()
        dir_to_color = [cell.get_color(i) for i in range(6)]
        for incoming_dir in range(6):
          if incoming_dir in completed_dirs:
              continue
          color = cell.get_color(incoming_dir)
          color_name = cell.get_color_name(incoming_dir)
          outgoing_dir = dir_to_color.index(color, incoming_dir + 1)
          hex_center = np.mean(hex_coords, axis=0)
          start = hex_mid_coords[incoming_dir, :]
          end = hex_mid_coords[outgoing_dir, :]
          weights = [1/2, 1/2]
          if abs(incoming_dir - outgoing_dir) in [1, 5]:
              weights = [5/8, 1/2]
          control1 = np.average([start, hex_center], weights=weights, axis=0)
          control2 = np.average([end, hex_center], weights=weights, axis=0)

          ctx.move_to(start[0] + tx, start[1] + ty)
          ctx.curve_to(
              control1[0] + tx, control1[1] + ty,
              control2[0] + tx, control2[1] + ty,
              end[0] + tx, end[1] + ty)
          ctx.set_source_rgb(*back_color)
          ctx.set_line_width(22)
          ctx.stroke_preserve()
          ctx.set_source_rgb(*color_table[color_name])
          ctx.set_line_width(16)
          ctx.stroke()

          completed_dirs.add(incoming_dir)
          completed_dirs.add(outgoing_dir)

        ctx.translate(tx, ty)
        hexagon(ctx, hex_coords, (1, 1, 1), False)
        ctx.translate(-tx, -ty)
    return surface


neighbor_delta = np.array([[-1, +1],
                           [ 0, +1],
                           [+1,  0],
                           [+1, -1],
                           [ 0, -1],
                           [-1,  0]])


class Board(object):
  """Board of Tantrix pieces.

  Attrs:
    grid: Array of currently placed Piece objects.
  """

  def __init__(self):
    self.grid = np.zeros((30, 30), dtype=np.object)

  def is_valid_placement_and_rotation(self, coord: Tuple[int, int],
                                      piece: Piece) -> bool:
    """Determines whether `piece` can be placed at `coord` with given rotation.
    """
    if self.grid[coord]:
      return False  # Coord is already taken.

    # Check validity at every edge.
    for direction in range(6):
      neighbor_coord = coord + neighbor_delta[direction, :]
      if (np.any(neighbor_coord < 0) or
          np.any(neighbor_coord >= self.grid.shape)):
        # Neighbor is out of bounds, so no objections in this direction.
        continue
      neighbor_coord = tuple(neighbor_coord)
      if not self.grid[neighbor_coord]:
        # Neighbor is unoccupied, so no objections in this direction.
        continue
      my_color = piece.get_color_name(direction)
      neighbor_color = self.grid[neighbor_coord].get_color_name(direction + 3)
      if my_color != neighbor_color:
        # print('Direction %d: My color %s does not match neighbor %s color %s'
        #       % (direction, my_color, self.grid[neighbor_coord],
        #          neighbor_color))
        return False
      # else:
      #   print('Direction %d: My color %s matches neighbor %s' % (
      #       direction, my_color, self.grid[neighbor_coord]))

    return True

  def get_next_coord(self,
                     piece: Piece,
                     incoming_dir: int,
                     start_coord: Tuple[int, int],
                     start_incoming_dir: int) -> Tuple[Tuple[int, int], int]:
    coord = piece.coord
    outgoing_dir = piece.get_outgoing_dir(incoming_dir)

    next_coord = tuple(coord + neighbor_delta[outgoing_dir, :])
    next_incoming_dir = (outgoing_dir + 3) % 6

    while self.grid[next_coord]:
        if next_coord == start_coord and next_incoming_dir == start_incoming_dir:
            return next_coord, next_incoming_dir
        next_piece = self.grid[next_coord]
        next_outgoing_dir = next_piece.get_outgoing_dir(next_incoming_dir)

        next_coord = tuple(next_coord + neighbor_delta[next_outgoing_dir, :])
        next_incoming_dir = (next_outgoing_dir + 3) % 6
    return next_coord, next_incoming_dir

  def get_placed_pieces(self,
                        start_coord: Tuple[int, int],
                        start_incoming_dir: int,
                        is_cycle: bool) -> Tuple[Tuple[int, int], int]:
    coord = start_coord
    start_piece = self.grid[start_coord]
    placed_pieces = [deepcopy(start_piece)]
    outgoing_dir = start_piece.get_outgoing_dir(start_incoming_dir)

    next_coord = tuple(coord + neighbor_delta[outgoing_dir, :])
    next_incoming_dir = (outgoing_dir + 3) % 6

    while True:
        if next_coord == start_coord and next_incoming_dir == start_incoming_dir:
            return placed_pieces
        next_piece = self.grid[next_coord]
        if not next_piece:
            assert(not is_cycle)
            return placed_pieces
        placed_pieces.append(deepcopy(next_piece))
        next_outgoing_dir = next_piece.get_outgoing_dir(next_incoming_dir)

        next_coord = tuple(next_coord + neighbor_delta[next_outgoing_dir, :])
        next_incoming_dir = (next_outgoing_dir + 3) % 6
    return placed_pieces


  def place_piece(self, coord: Tuple[int, int], piece: Piece) -> None:
    self.grid[coord] = piece
    piece.coord = coord

  def remove_piece(self, coord: Tuple[int, int]) -> None:
    self.grid[coord].coord = None
    self.grid[coord] = 0

  def has_holes(self) -> bool:
    piece_locations = (self.grid != 0)
    filled_locations = morphology.binary_fill_holes(piece_locations)
    return np.any(piece_locations != filled_locations)

  def compress(self, test: bool) -> None:
    row_shrink = 0
    column_shrink = 0
    total_shrink = 0
    """Shrink grid to smallest possible size."""
    for i in range(self.grid.shape[1]):
      if not np.any(self.grid[:, i:]):
        total_shrink += self.grid.shape[1] - 1 - i
        if not test:
          self.grid = self.grid[:, :i]
        break
    for i in range(self.grid.shape[1], 0, -1):
      if not np.any(self.grid[:, :i]):
        column_shrink = i
        total_shrink += i
        if not test:
          self.grid = self.grid[:, i:]
        break
    for i in range(self.grid.shape[0]):
      if not np.any(self.grid[i:, :]):
        total_shrink += self.grid.shape[0] - 1 - i
        if not test:
          self.grid = self.grid[:i, :]
        break
    for i in range(self.grid.shape[0], 0, -1):
      if not np.any(self.grid[:i, :]):
        row_shrink = i
        total_shrink += i
        if not test:
          self.grid = self.grid[i:, :]
        break
    if test:
        return total_shrink
    else:
        return row_shrink, column_shrink

def hex_distance(a, b):
    return max(
        abs(a[0] - b[0]),
        abs(a[1] - b[1]),
        abs((a[0] + a[1]) - (b[0] + b[1]))
    )


def find_cycle(unsorted_pieces: List[Piece], allow_holes: bool) -> Optional[Board]:
  """Solve a Tantrix puzzle.

  Args:
    pieces: List of pieces to use. Their locations and rotations will be
      updated.
    allow_holes: Whether holes are allowed in the solution.

  Returns:
    Shallow copy of list of pieces, along with their placement in the grid

  Raises:
    RuntimeError: If puzzle is unsolvable.
  """
  global best_possible_board, best_possible_placed_pieces, best_total_shrink, seed
  pieces = deepcopy(unsorted_pieces)
  pieces.sort(key=lambda p: -p.umwelt)
  board = Board()
  start_time = time.time()
  start_coord = (15, 15)
  start_incoming_dir = None
  # n_attempts = 0

  def recurse(coord: Tuple[int, int], piece_idx: int, incoming_dir: int) -> bool:
    global resolve_state
    if resolve_state == 4:
        return False

    # not the most correct heuristic, but let's try it:
    if hex_distance(coord, start_coord) > len(pieces) - piece_idx:# + some_addition
        return False

    board.place_piece(coord, pieces[piece_idx])

    # n_attempts += 1
    depth = piece_idx
    # print('%sPlaced %s' % ('  ' * depth, pieces[piece_idx]))

    next_coord, next_incoming_dir = board.get_next_coord(
        pieces[piece_idx], incoming_dir, start_coord, start_incoming_dir)
    # print('%sNext coord %s' % ('  ' * depth, next_coord))

    if next_coord == start_coord and start_incoming_dir == next_incoming_dir:
      if (piece_idx == len(pieces) - 1) and (allow_holes or not board.has_holes()):
        print('%sSolved!' % ('  ' * depth))
        print(piece_idx, board.grid[coord], coord, start_coord, hex_distance(coord, start_coord), len(pieces) - piece_idx)
        return True  # Solved!
      else:
        board.remove_piece(coord)
        return False  # Found a cycle, but not with all pieces
    elif piece_idx == len(pieces) - 1:
      global best_possible_board, best_possible_placed_pieces, best_total_shrink, seed
      placed_pieces = board.get_placed_pieces(start_coord, start_incoming_dir, False)
      total_shrink = board.compress(True)
      if total_shrink > best_total_shrink:
          best_total_shrink = total_shrink
          best_possible_board = deepcopy(board)
          row_shrink, column_shrink = best_possible_board.compress(False)
          for piece in placed_pieces:
              piece.coord = (piece.coord[0] - row_shrink, piece.coord[1] - column_shrink)
          best_possible_placed_pieces = placed_pieces
          seed = 0
      board.remove_piece(coord)
      return False  # Placed all pieces, but did not form a cycle.

    rotations = list(range(6))
    shuffle(rotations)
    for next_piece_idx in range(piece_idx + 1, len(pieces)):
      pieces[piece_idx + 1], pieces[next_piece_idx] = pieces[next_piece_idx], pieces[piece_idx + 1]
      next_piece = pieces[piece_idx + 1]
      for rotation in rotations:
        next_piece.rotation = rotation
        if board.is_valid_placement_and_rotation(next_coord, next_piece):
          if recurse(next_coord, piece_idx + 1, next_incoming_dir):
            print(piece_idx, board.grid[coord], coord, start_coord, hex_distance(coord, start_coord), len(pieces) - piece_idx)
            return True  # Solved
      # actually, this is unnecessary:
      pieces[piece_idx + 1], pieces[next_piece_idx] = pieces[next_piece_idx], pieces[piece_idx + 1]

    # Dead end. Undo changes to board.
    # print('%sRemoved %s' % ('  ' * depth, pieces[piece_idx]))
    board.remove_piece(coord)
    return False

  cycle_color = 'G'
  for i in ['0', '1', '2']:
      piece = pieces[0]
      if piece.color_names[i] != cycle_color:
          continue
      start_incoming_dir = (piece.ins[i] - piece.rotation) % 6
      if recurse(start_coord, 0, (piece.ins[i] - piece.rotation) % 6):
        end_time = time.time()
        print('== Solved in %.1f seconds ==' % (end_time - start_time))
        for piece in pieces:
          print('%s @%s' % (piece, piece.coord))
        placed_pieces = board.get_placed_pieces(start_coord, start_incoming_dir, True)
        row_shrink, column_shrink = board.compress(False)
        for piece in placed_pieces:
            piece.coord = (piece.coord[0] - row_shrink, piece.coord[1] - column_shrink)
        seed = 0
        return board, placed_pieces, 1

  print('Failed to solve.')
  return None, None, 2
