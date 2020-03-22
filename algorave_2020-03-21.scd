(
s.options.outDevice_("BlackHole 16ch")
)

("co34pt/Setup/Setup.scd").loadRelative

p.clock.tempo = 80/60;

(
~oh = Pbind(\instrument,\bplay,
	\buf,d["oh"][1],
	\dur,Pseq([0.5,Pseq([1],inf)],inf),
	\amp,0.8);
~oh.play;
)

(
/*l = Prewrite(1, // start with 1
        (    1:  [0.25,2],
            0.25: [3,3,2]/4,
            3/4: [0.25,1,0.125,0.125],
        ), 4);*/

l = Pbjorklund2(Pwhite(2,2),16)/4;

~k = Pbind(\instrument,\bplay,
	\buf,d["k"][2],
	\dur,l,
	\amp,2,
	\rate,Pseq([1,1.4],inf));
~k.play;

~k2 = Pbind(\instrument,\bplay,
	\buf,d["k"][2],
	\dur,l,
	\amp,2,
	\rate,Pseq([1,1.8],inf)*4);
~k2.play;
)

l = Pbjorklund2(Pwhite(2,2),16)/4;
~sn = Pbind(\instrument,\bplay,\buf,d["s"][1],\dur,l,\amp,0.3);
~sn.play;

// ~h = Pbind(\instrument,\bplay,\buf,d["ch"][1],\dur,l,\amp,0.5);
~h = Pbind(\instrument,\bplay,\buf,d["ch"][0],\dur,Pwrand([0.25,Pseq([0.125],4),Pseq([0.25]/3,3),Pseq([0.125]/2,4)],[0.3,0.5,0.09,0.01],inf),\amp,1,\rate,2);
~h.play;
)

(
l = Pbjorklund2(Pwhite(10,12),16)/4;
~t = Pbind(\instrument,\bplay,\buf,d["t"][0],\dur,l,\amp,0.4,\rate,4);
~t.play;
~t2 = Pbind(\instrument,\bplay,\buf,d["t"][0],\dur,l,\amp,0.5,\rate,2);
~t2.play;
)

(
~sn.stop;
~h.stop;
)
(
~k.stop;
~k2.stop;
)
(
~t.stop;
~t2.stop;
)

(
SynthDef(\bpfsaw, {
	arg atk=2, sus=0, rel=3, ts=1, c1=1, c2=(-1), freq=500, detune=0.02, pan=0, cfhzmin=0.1, cfhzmax=0.3, cfmin=500, cfmax=2000, rqmin=0.1, rqmax=0.2, lsf=200, ldb=0, amp=1, out=0;
	var sig, env;
	env = EnvGen.kr(Env([0,1,1,0],[atk,sus,rel],[c1,0,c2]),timeScale:ts,doneAction:2);
	sig = Saw.ar(freq*{LFNoise1.kr(0.5,detune).midiratio}!2);
	sig = BPF.ar(
		sig,
		{LFNoise1.kr(
			LFNoise1.kr(4).exprange(cfhzmin,cfhzmax)
		).exprange(cfmin,cfmax)}!2,
		{LFNoise1.kr(0.1).exprange(rqmin,rqmax)}!2
	);
	sig = BLowShelf.ar(sig,lsf,0.5,ldb);
	sig = Balance2.ar(sig[0],sig[1],pan);
	sig = sig*env*amp;
	Out.ar(out,sig);
}).add;
);

(
SynthDef(\sinfb, {
	arg freq = 440, atk = 0.01, sus = 0, rel = 1, fb = 0, amp = 0.3, out = 0, pan=0;
	var sig, env;
	env = EnvGen.ar(Env.linen(atk,sus,rel),1,1,0,1,2);
	sig = SinOscFB.ar(freq,fb,1) + SinOscFB.ar(freq * 1,fb,1);
	sig = sig*env;
	Out.ar(out,Pan2.ar(sig,pan,amp));
}).add;
);

(
SynthDef(\ring1,
	{
		arg f = 45, a = 9, d = 0.6, pan = 0, amp = 1, out = 0;
		var sig, env;
		env = Line.kr(0,1,d);
		env = FreeSelfWhenDone.kr(env);
		sig = Impulse.ar(0);
		sig = Ringz.ar(sig,f,d,a);
		Out.ar(out,Pan2.ar(sig.tanh,pan) * amp)
	};
).add;
)

(
~ring1 = Pbind(\instrument,\ring1,\f,Pkey(\freq),\scale,d[\scale],\buf,d["s"][1],\dur,l,\degree,Pwhite(-3,3),\octave,4,\d,0.38,\a,Pexprand(1,40),\amp,0.8,\pan,0);
~ring1.play;
)

(
var oct = 31; // notes in octave
var stretch;
var chords = [];
f = round(log(3/2)/log(2)*oct); // perfect fifth width
stretch = 1;
f = f * stretch;
f.postln;

// t == 12 by default
// choosing different tabs (MOS scales)
n = [0, 7 * f - 48, 2 * f - 12, 9 * f - 60, 4 * f - 24, 11 * f - 72, 6 * f - 36, 1 * f - 0, 8 * f - 48, 3 * f - 12, 10 * f - 60, 5 * f - 24];
if (oct == 5) {
	n = [0, 2 * f - 5, 4 * f - 10, 1 * f - 0, 3 * f - 5];
};
if (oct == 7) {
	n = [0, 2 * f - 7, 4 * f - 14, 6 * f - 21, 1 * f - 0, 3 * f - 7, 5 * f - 14];
};
if (oct == 19) {
	n = [0, 7 * f - 76, 14 * f - 152, 2 * f - 19, 9 * f - 95, 16 * f - 171, 4 * f - 38, 11 * f - 114, 18 * f - 190, 6 * f - 57, 13 * f - 133, 1 * f - 0, 8 * f - 76, 15 * f - 152, 3 * f - 19, 10 * f - 95, 17 * f - 171, 5 * f - 38, 12 * f - 114];
};
if (oct == 31) {
	n = [0, 19 * f - 341, 7 * f - 124, 26 * f - 465, 14 * f - 248, 2 * f - 31, 21 * f - 372, 9 * f - 155, 28 * f - 496, 16 * f - 279, 4 * f - 62, 23 * f - 403, 11 * f - 186, 30 * f - 527, 18 * f - 310, 6 * f - 93, 25 * f - 434, 13 * f - 217, 1 * f - 0, 20 * f - 341, 8 * f - 124, 27 * f - 465, 15 * f - 248, 3 * f - 31, 22 * f - 372, 10 * f - 155, 29 * f - 496, 17 * f - 279, 5 * f - 62, 24 * f - 403, 12 * f - 186];
};
if (oct == 43) {
	n = [0, 31 * f - 774, 19 * f - 473, 7 * f - 172, 38 * f - 946, 26 * f - 645, 14 * f - 344, 2 * f - 43, 33 * f - 817, 21 * f - 516, 9 * f - 215, 40 * f - 989, 28 * f - 688, 16 * f - 387, 4 * f - 86, 35 * f - 860, 23 * f - 559, 11 * f - 258, 42 * f - 1032, 30 * f - 731, 18 * f - 430, 6 * f - 129, 37 * f - 903, 25 * f - 602, 13 * f - 301, 1 * f - 0, 32 * f - 774, 20 * f - 473, 8 * f - 172, 39 * f - 946, 27 * f - 645, 15 * f - 344, 3 * f - 43, 34 * f - 817, 22 * f - 516, 10 * f - 215, 41 * f - 989, 29 * f - 688, 17 * f - 387, 5 * f - 86, 36 * f - 860, 24 * f - 559, 12 * f - 258];
};
if (oct == 55) {
	n = [0, 43 * f - 1375, 31 * f - 990, 19 * f - 605, 7 * f - 220, 50 * f - 1595, 38 * f - 1210, 26 * f - 825, 14 * f - 440, 2 * f - 55, 45 * f - 1430, 33 * f - 1045, 21 * f - 660, 9 * f - 275, 52 * f - 1650, 40 * f - 1265, 28 * f - 880, 16 * f - 495, 4 * f - 110, 47 * f - 1485, 35 * f - 1100, 23 * f - 715, 11 * f - 330, 54 * f - 1705, 42 * f - 1320, 30 * f - 935, 18 * f - 550, 6 * f - 165, 49 * f - 1540, 37 * f - 1155, 25 * f - 770, 13 * f - 385, 1 * f - 0, 44 * f - 1375, 32 * f - 990, 20 * f - 605, 8 * f - 220, 51 * f - 1595, 39 * f - 1210, 27 * f - 825, 15 * f - 440, 3 * f - 55, 46 * f - 1430, 34 * f - 1045, 22 * f - 660, 10 * f - 275, 53 * f - 1650, 41 * f - 1265, 29 * f - 880, 17 * f - 495, 5 * f - 110, 48 * f - 1485, 36 * f - 1100, 24 * f - 715, 12 * f - 330];
};

// todo: 43, 55

n = n / n.size * 12;
n.postln;

l = Pbjorklund2(Pwhite(2,4),16)/4;

d[\tuning] = Tuning.new(n);
d[\scale] = Scale.chromatic(d[\tuning]);


chords = round([0,0,[0,3,6],[0,3,7],[0,5,10],[0,4,11],[0,5,7,7.5]] / 12 * oct);
chords.postln;

~ring1 = Pbind(\instrument,\ring2,\f,Pkey(\freq),\scale,d[\scale],\buf,d["s"][1],\dur,l,\degree,Pwhite(-12,12),\octave,4,\d,0.38,\a,Pexprand(1,40),\amp,0.8,\pan,0,\rate,128);
~ring1.stop;
~ring2 = Pbind(\instrument,\ring2,\f,Pkey(\freq),\scale,d[\scale],\buf,d["s"][1],\dur,l,\degree,Pwhite(-12,12),\octave,4,\d,0.38,\a,Pexprand(1,40),\amp,0.5,\pan,1,\rel,0.5,\atk, 0.1);
~ring2.stop;
~ring3 = Pbind(\instrument,\ring2,\f,Pkey(\freq),\scale,d[\scale],\buf,d["s"][1],\dur,l,\degree,chords,\octave,Pwrand([5,7],[0.8,0.2],inf),\d,0.38,\a,Pexprand(10,400),\amp,1,\pan,-1,\rel,0.5,\atk, 0.1);
~ring3.stop;

~sinfb = Pbind(\instrument,\sinfb,
	\scale,d[\scale],
	\buf,d["s"][0],
	\dur,1/3,
	\degree,Place(chords, inf),
	\octave,Pwrand([4,5],[0.2,0.8],inf),
	\d,0.38,\a,Pexprand(10,400),\amp,1,\pan,Pwhite(-0.9, 0.9),\rel,0.2);
~sinfb.play;

~sinfb2 = Pbind(\instrument,\sinfb,
	\scale,d[\scale],
	\buf,d["s"][0],
	\dur,1/3,
	\degree,Place(chords, inf),
	\octave,Pwrand([5,6],[0.2,0.8],inf),
	\d,0.38,\a,Pexprand(10,400),\amp,1,\pan,Pwhite(-0.9, 0.9),\rel,0.2);
~sinfb2.play;


~sinfb3 = Pbind(\instrument,\sinfb,
	\scale,d[\scale],
	\buf,d["s"][0],
	\dur,1/3,
	\degree,Place(chords, inf),
	\octave,Pwrand([4,5],[0.5,0.5],inf),
	\d,0.38,\a,Pexprand(10,400),\amp,1,\pan,Pwhite(-0.9, 0.9),\rel,0.2);
~sinfb3.play;
)


(
~oh.stop(4);
~sinfb.stop(4);
~sinfb3.stop(4);~sinfb2.stop(4);
)