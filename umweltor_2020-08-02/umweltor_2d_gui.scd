// init
(
//increase number of buffers the server has access to for loading samples
s.options.numBuffers = 1024 * 16;
//increase the memory available to the server
s.options.memSize = 8192 * 64;
s.options.sampleRate = 48000;
// s.options.outDevice_("Soundflower (2ch)");

//boot the server
s.boot;
Task({
	3.wait;
	//activate StageLimiter - Part of the BatLib quark
	StageLimiter.activate;
	"Setup done!".postln;
}).start;

("umweltor_synth.scd").loadRelative
)



// gui
(
var half = 0.5;
var len = 600;
w = Window("umweltor", Rect(300, 300, len, len), false);

// (1) create a UserView
v = UserView(w, Rect(0, 0, len, len));

// (2) define a drawing function using Pen
v.drawFunc = {
    // Draw the frame
    Pen.strokeColor = Color.black;
	// Pen.addRect(Rect(0,0, v.bounds.width,v.bounds.height));
    Pen.stroke;
    // Draw the triangle
	Pen.fillColor = Color.red(1.6);
    Pen.moveTo(0 @ 0);
    Pen.lineTo(0 @ v.bounds.height);
    Pen.lineTo((v.bounds.width*half) @ (v.bounds.height*half));
    Pen.lineTo(0 @ 0);
    Pen.fill;
	// Draw another one
	Pen.fillColor = Color.blue(1.6);
    Pen.moveTo(v.bounds.width @ 0);
    Pen.lineTo(v.bounds.width @ v.bounds.height);
    Pen.lineTo((v.bounds.width*half) @ (v.bounds.height*half));
    Pen.lineTo(v.bounds.width @ 0);
    Pen.fill;

	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc(0 @ 0, 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(5, 0, 80, 30)).string_("max-\nотскок");

	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc(0 @ (v.bounds.height*half), 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(5, (v.bounds.height*half) - 15, 80, 30)).string_("max=\nупругость");

	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc(0 @ v.bounds.height, 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(5, v.bounds.height - 30, 120, 30)).string_("max+\nнепроницаемость");

	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc((v.bounds.width/3) @ (v.bounds.height/3), 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(v.bounds.width/3, v.bounds.height/3, 80, 30)).string_("тряска");
	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc((v.bounds.width/3) @ (v.bounds.height*half), 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(v.bounds.width/3, v.bounds.height*half, 80, 30)).string_("скольжение");
	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc((v.bounds.width/3) @ (v.bounds.height*2/3), 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(v.bounds.width/3, v.bounds.height*2/3, 80, 30)).string_("прилипание");

	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc((v.bounds.width*2/3) @ (v.bounds.height/3), 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(v.bounds.width*2/3, v.bounds.height/3, 120, 30)).string_("выталкивание");
	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc((v.bounds.width*2/3) @ (v.bounds.height*half), 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(v.bounds.width*2/3, v.bounds.height*half, 80, 30)).string_("погружение");
	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc((v.bounds.width*2/3) @ (v.bounds.height*2/3), 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(v.bounds.width*2/3, v.bounds.height*2/3, 80, 30)).string_("увязание");

	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc(v.bounds.width @ 0, 5, 0, 2 * pi);
	StaticText.new(v, Rect(v.bounds.width - 70, 0, 80, 30)).string_("     min-\nразрыв");
	Pen.fill;

	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc(v.bounds.width @ (v.bounds.height*half), 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(v.bounds.width - 70, (v.bounds.height*half) - 15, 80, 30)).string_("     min=\nцепляние");

	// Draw circle
	Pen.color = Color.yellow(1, 1);
	Pen.addArc(v.bounds.width @ v.bounds.height, 5, 0, 2 * pi);
	Pen.fill;
	StaticText.new(v, Rect(v.bounds.width - 70, v.bounds.height - 30, 80, 30)).string_("     min+\nзияние");
};

w.acceptsMouseOver = true;

v.mouseEnterAction = { |u, y, x|
	s.sendMsg('/s_new', \umweltor_synth, a = s.nextNodeID, 0, 1);
};

v.mouseOverAction = { |u, y, x|
	var ux = x / len;
	var uy = y / len;
	var x_shift = 0.5 - ((uy - 0.5).abs);
	if ((uy - 0.5).abs > 1e-5) {
		ux = (ux - x_shift) /  (1 - (2 * x_shift));
	};
	s.sendMsg('/n_set', a, \ux, ux, \uy, uy);
	(ux @ uy).postln;
};

v.mouseLeaveAction = { |u|
	s.sendMsg('/n_free', a);
};

w.front;
w.refresh;
)



// не вошедшее, но тоже хорошо:


// сыпучесть
{HPF.ar(WhiteNoise.ar, 1000) * LFNoise2.kr(0.6).squared.squared ! 2}.play



// сверчки
(
a = {
	var modulator, mod1, mod2, mod3;

	// repeat time is 0.7s: equates to 1.43 Hz.
	modulator = LFSaw.ar(1.43, 1, 0.5, 0.5);
	mod2 = (modulator * 40.6 * 2pi).cos.squared;
	mod3 = modulator * 3147;
	mod3 = (mod3 * 2pi).cos + ((mod3 * 2 * 2pi).cos * 0.3);
	mod1 = ((Wrap.ar(modulator.min(0.1714) * 5.84) - 0.5).squared * (-4) + 1) * (mod2 * mod3);
	mod1 = (mod1 * 0.1)!2;
}.play;
)
a.free;


// сверчки 2
(
b = {
	var trig, seq, demand, cricket;

	// instead of [metro], Impulse.kr is used here. Delta t = 17 ms equates to 58.82 Hz.
	trig = Impulse.kr(58.82);

	// the main idea of the following line was to use an approach
	// that uses the same coefficients as described in the pd patch
	seq = Dseq(Array.fill(41, {|i| if(i<7, {(i+2)/9},{0}) }),inf);
	demand = Demand.kr(trig,0,seq);

	// Implementation of the pd code for pulses including amplitude grow:
	// cricket = EnvGen.ar(Env.new([0, 1, 1, 0], [0.0001, 0.0001, 0]), trig) * demand;

	// 2nd implementation: pure data seemed to slightly disobey its own specifications,
	// so I analysed the waveform and came up with this:
	cricket = EnvGen.ar(Env.new([0, 1, 0], [4/44100, 0]), trig) * demand;


	cricket = OnePole.ar(cricket, exp(-2pi * (1000 * SampleDur.ir)));
	cricket = (
			// changed the Q factor of the first 3 BPFs to approximate farnells sound
			BPF.ar(cricket, 4500 + ((0..2)*50), 300.reciprocal, 100)).sum
			+ BPF.ar(cricket, 9000, 500.reciprocal, 42
	);
	cricket = ((cricket - OnePole.ar(cricket, exp(-2pi * (4000 * SampleDur.ir)))) * 0.5)!2;
}.play;
)
b.free;


// сверчки 3
(
c = {
	var sig, trig, seq, freq, mul, vals;

	trig = Impulse.kr(0.2);
	vals = [
		[0.5, 128],
		[8,6],
		[30,3]
	];
	freq = TChoose.kr(trig, vals);


	sig = WhiteNoise.ar;
	// The one pole filters in pure data and SC differ, so I changed the coefficents
	// a little. Also the  multiplication by 5 is not in the book, but helps to
	// approach the audible result of Farnells patch.
	sig = (sig - OnePole.ar(sig, exp(-2pi * (8000 * SampleDur.ir))));
	sig = (sig - OnePole.ar(sig, exp(-2pi * (8000 * SampleDur.ir))));
	sig = OnePole.ar(sig, exp(-2pi * (10000 * SampleDur.ir)));
	sig = OnePole.ar(sig, exp(-2pi * (10000 * SampleDur.ir)));
	sig = sig * 5;


	sig = BPF.ar(sig, [7500, 5500], 40.reciprocal).sum * SinOsc.ar(500);
	sig = sig * (1 / (SinOsc.ar( freq[0], 0, freq[1] ).squared + 1));
	sig = (sig - OnePole.ar(sig, exp(-2pi * (4000 * SampleDur.ir)))) * 4.dup;
}.play;
)
c.free;



// пузыри
(
SynthDef(\bubblebub, {	|out=0, t_trig=0, attack=0.01, decay=0.08, pitchcurvelen=0.1, freq=1000, doneAction=0, amp=0.1|
	var pitch, son;
	amp   = amp * EnvGen.ar(Env.perc(attack, decay).delay(0.003), t_trig, doneAction: doneAction);
	pitch = freq * EnvGen.ar(Env.new([0,0,1],[0,1]).exprange(1, 2.718), t_trig, timeScale: pitchcurvelen);
	son = SinOsc.ar(pitch);
	// high-pass to remove any lowpitched artifacts, scale amplitude
	son = HPF.ar(son, 500) * amp * 10 ! 2;
	Out.ar(out, son);
}).store
)
(
p = Pbind(
	\instrument, \bubblebub,
	\sizefactor, Pwhite(0.0,1,inf),
	\dur, Pgauss(0.3, 0.2),
	\freq,  Pkey(\sizefactor).linexp(0, 1, 1000, 3000),
	\amp ,  Pkey(\sizefactor).linlin(0, 1, 0.15, 0.04),
	\decay, Pkey(\sizefactor).linlin(0, 1, 0.05, 0.08),
	\doneAction, 2
).play
)
p.stop
