from sc3.all import *
import random


@synthdef
def bplay(out = 0, buf = 0, rate = 1, amp = 0.5, pan = 0, pos = 0, rel=15,
          hpf=100, lpf=10000):
    sig = PlayBuf.ar(2, buf, BufRateScale.ir(buf) * rate,
                     1, BufDur.kr(buf) * pos * 48000,
                     done_action=2)
    env = EnvGen.ar(Env.linen(0.0, rel, 0), done_action=2)
    sig = sig * env
    sig = sig * amp
    sig = HPF.ar(sig, hpf)
    sig = LPF.ar(sig, lpf)
    sig = Pan2.ar(sig, pan)
    sig = Mix.ar(sig)
    Out.ar(out, sig)
