#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyOSC3


client = pyOSC3.OSCClient()
client.connect(('127.0.0.1', 57120))


def renormalize(px, py):
    if abs(py - 0.5) < 1e-5:
        return px
    x_shift = 0.5 - abs(py - 0.5)
    return (px - x_shift) /  (1 - 2 * x_shift)


def send_msg(px, py):
    try:
        msg = pyOSC3.OSCMessage()
        msg.setAddress("/umweltor")
        x = renormalize(px, py)
        msg.append(x)
        msg.append(py)
        print('sent', x, py, 'vs', px)
        client.send(msg)
    except Exception as e:
        pass
