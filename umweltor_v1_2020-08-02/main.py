#!/usr/bin/env python
# -*- coding: utf-8 -*-

# FIXME: add link and acknowledgements to tantrix code
# tantrix gui
# app gui
# supercollider interaction
# tantrix state machine (?)
# разобраться - как избавиться от глобальных переменных,
# или как с ними работать из разных файлов
# через конфиг? синглтон?
# по-хорошему надо build файл придумать
# FIXME: "Showing current best path" как будто бы сломан и ничего не показывает
# FIXME: add_umwelt должен очищать текст "No resolution found!"
# FIXME: обновление мешка какое-то бажное
# FIXME: в посыле параметров синта где-то бага в вычислениях

import numpy as np
import json

import chart_studio as plotly
import plotly.graph_objs as go

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import cv2
from random import randint
import base64

from tantrix import *
from supercollider_interaction import *
from umweltor_gui import *


last_piece = None
play_board = None
play_pieces = None

added_pieces = []
# reminds of trefoil knot; solvable immediately
added_pieces.append(Piece('#0', '001122', 0, x=0.218353, y=0.060241, color_names='GGG'))
added_pieces.append(Piece('#1', '001122', 0, x=0.273375, y=0.100402, color_names='GGG'))
added_pieces.append(Piece('#2', '001122', 0, x=0.222809, y=0.048193, color_names='GGG'))
added_pieces.append(Piece('#3', '001122', 0, x=0.269561, y=0.060241, color_names='GGG'))
added_pieces.append(Piece('#4', '001122', 0, x=0.167774, y=0.076305, color_names='GGG'))
added_pieces.append(Piece('#5', '001122', 0, x=0.263490, y=0.056225, color_names='GGG'))
added_pieces.append(Piece('#6', '201201', 9, x=0.121749, y=0.943775, color_names='GGG'))


def get_piece_and_image(px, py):
    grid = np.zeros((1, 1), dtype=np.object)
    piece = None

    if px > 0 and py > 0:
        # FIXME: rewrite into Piece constructor
        nearest_umwelt = calc_umw(px, py)
        piece_num = int(nearest_umwelt / 3)
        piece_type = pieces[piece_num]
        piece_prototype = piece_type[randint(0, len(piece_type) - 1)]
        piece = Piece('piece', piece_prototype.colors, nearest_umwelt, x=px, y=py)
        piece.rotation = randint(0, 5)
        color_count = nearest_umwelt % 3
        color_order = color_orders[color_count]
        color_names = color_order[randint(0, len(color_order) - 1)]
        piece.set_color_names(color_names)
        grid[(0, 0)] = piece

    surface = draw_surface_from_grid(grid)
    img = surface_to_npim(surface)
    ret, png = cv2.imencode('.png', img)
    dash_image = png.tobytes()
    return dash_image, "data:image/png;base64," + base64.b64encode(dash_image).decode('ascii'), piece


def add_umwelt():
    global added_pieces, last_piece
    print('add umwelt')
    if last_piece is not None:
        last_piece.name = '#' + str(len(added_pieces))
        added_pieces.append(last_piece)
    last_piece = None

    grid = np.zeros((1, len(added_pieces)), dtype=np.object)
    for piece_idx, piece in enumerate(added_pieces):
        grid[(0, piece_idx)] = piece

    surface = draw_surface_from_grid(grid)
    img = surface_to_npim(surface)
    ret, png = cv2.imencode('.png', img)
    added_pieces_image = png.tobytes()
    return "data:image/png;base64," + base64.b64encode(added_pieces_image).decode('ascii')


def resolve_umwelts():
    print('resolve umwelt')
    global best_possible_board, best_possible_placed_pieces, best_total_shrink, seed
    best_possible_board = None
    best_possible_placed_pieces = []
    best_total_shrink = 0
    seed = 0

    global added_pieces, play_board, play_pieces, resolve_state
    print(added_pieces)
    play_board = None
    play_pieces = None
    resolve_state = 3
    play_board, play_pieces, local_resolve_state = find_cycle(added_pieces, allow_holes=True)
    if play_board is None:
        best_possible_board = None
        best_possible_placed_pieces = []
        best_total_shrink = 0
        seed = 0
        send_msg(-1, -1)
    resolve_state = local_resolve_state


def get_image_for(seed, play_board, play_pieces):
    piece = play_pieces[seed % len(play_pieces)]
    surface = draw_surface_from_grid(play_board.grid,
                                     selected_coord=piece.coord)
    img = surface_to_npim(surface)
    ret, png = cv2.imencode('.png', img)
    play_image = png.tobytes()
    return "data:image/png;base64," + base64.b64encode(play_image).decode('ascii'), (piece.x, piece.y)


def get_image(seed):
    global best_possible_board, best_possible_placed_pieces, play_board, play_pieces, resolve_state
    if resolve_state in [2, 4]:
        return no_image, (None, None)
    if play_board is None:
        if best_possible_board is None:
            return no_image, (None, None)
        else:
            return get_image_for(seed, best_possible_board, best_possible_placed_pieces)
    else:
        return get_image_for(seed, play_board, play_pieces)


added_pieces_image = add_umwelt()
no_image_image, no_image, no_piece = get_piece_and_image(-1, -1)



app = dash.Dash(__name__)

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene={
        'camera': {
            'eye': {"x": 1.4, "y": -1.2, "z": 1.14},
            'up' : {'x': 0, 'y': 0, 'z': 1}}
    }
)
fig = go.Figure(data=data, layout=layout)
fig.update_traces(hoverinfo='none')
fig.update_layout(
    width=600,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=4
    )
)

app.layout = html.Div(children=[
    html.Div(children=[
        html.Div(children=[
            html.Div(id='hidden-hover-div', style={'display': 'none'}),
            html.Div(id='hidden-resolve-div', style={'display': 'none'}),
            html.Div(id='hidden-stop-div', style={'display': 'none'}),
            html.P('selected umwelts:', id="chosen-text"),
            html.Img(id='tantrix-pieces',
                     src=added_pieces_image,
                     style={'height': '150px'})
        ], className="row"),

        html.Div(children=[
            html.Img(id='tantrix-current-piece',
                     src=no_image,
                     style={'width': '100px'}),
            html.Button('add umwelt', id='add-umwelt', n_clicks=0),
            html.Button('resolve umwelts', id='resolve-umwelts', n_clicks=0),
            html.Button('clear umwelts', id='clear-umwelts', n_clicks=0),
            html.Button('stop resolving', id='stop-resolving', n_clicks=0),
        ], className="row"),

        html.Div(children=[
            html.Div([
                dcc.Graph(
                    id='umweltor',
                    figure=fig,
                    config={
                        "displaylogo": False,
                        'displayModeBar': False,
                    }
                ),
            ], className="col-md-7"),
            html.Div(children=[
                html.Div(
                    [dcc.Interval(id="interval", interval=1000),
                     html.P('',
                       id="no-resolution-text",
                       style={'color': 'red'})]),
                html.Div(
                    [dcc.Interval(id="tantrix-interval", interval=1000),
                     html.Img(id='tantrix-solution',
                         src=no_image,
                         style={'width': '100%'})])
            ], className="col-md-5"),
        ], className="row"),
    ], className="col-md-12"),
], className="container-fluid")


@app.callback(Output('hidden-hover-div', 'children'),
              [Input('umweltor', 'hoverData')])
def display_hover_data(hoverData):
    if hoverData is not None:
        p = hoverData['points'][0]
        px, py = p['x'], p['y']
        send_msg(px, py)
    return json.dumps(hoverData, indent=2)


@app.callback(Output('tantrix-current-piece', 'src'),
              [Input('umweltor', 'clickData'),
               Input('add-umwelt', 'n_clicks'),])
def display_click_data(clickData, add_btn):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'add-umwelt' in changed_id:
        return no_image

    global last_piece
    last_piece = None
    img = no_image
    if clickData is not None and 'points' in clickData:
        p = clickData['points'][0]
        px, py = p['x'], p['y']
        if px < 0 or py < 0 or px > 1 or py > 1:
            return img
        if abs(py - 0.5) < 1e-6:
            return img
        if ((py <= px) and (px + py > 1)) or\
                ((py >= px) and (px + py < 1)):
            return img
        img_img, img, piece = get_piece_and_image(px, py)
        last_piece = piece
    return img


@app.callback(
    Output('tantrix-pieces', 'src'),
    [Input('add-umwelt', 'n_clicks'),
     Input('resolve-umwelts', 'n_clicks'),
     Input('clear-umwelts', 'n_clicks')])
def press_button_for_tantrix_pieces(add_btn, resolve_btn, clear_btn):
    global added_pieces, resolve_state
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'add-umwelt' in changed_id:
        added_pieces_image = add_umwelt()
        return added_pieces_image
    elif 'clear-umwelts' in changed_id:
        added_pieces = []
        return no_image
    else:
        raise dash.exceptions.PreventUpdate


@app.callback(Output('hidden-resolve-div', 'children'),
              [Input('resolve-umwelts', 'n_clicks')])
def press_resolve_umwelts(resolve_btn):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'resolve-umwelts' in changed_id:
        resolve_umwelts()
        raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output("no-resolution-text", "children"),
    [Input("interval", "n_intervals")])
def display_no_resolution(n):
    global resolve_state
    print('resolve_state:', resolve_state)
    if resolve_state == 3:
        return 'Showing current best path:'
    elif resolve_state == 2:
        return 'No resolution found!'
    elif resolve_state == 1:
        return 'Resolution:'
    else:
        return ''


@app.callback(
    Output('tantrix-solution', 'src'),
    [Input("tantrix-interval", "n_intervals")])
def get_tantrix_solution(n):
    global seed
    frame, point = get_image(seed=seed)
    if point[0] is not None:
        send_msg(point[0], point[1])
    seed += 1
    return frame


@app.callback(Output('hidden-stop-div', 'children'),
              [Input('stop-resolving', 'n_clicks')])
def stop_resolving(stop_btn):
    global resolve_state
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'stop-resolving' in changed_id:
        resolve_state = 4
        send_msg(-1, -1)
    raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)
