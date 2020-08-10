
(
SynthDef(\umweltor_synth, {|ux, uy|
	// meta
	var synths;

	// проникновение
	var loopTime = 8;
	var pop, pops, imp, modImp;
	var midNoise, hiNoise, loNoise;
	var minNoiseSrc, minNoiseRev, minNoise;

	// прилегание
	var waves_n = 7;
	var waves_amp, waves_freq, waves_edur;
	var waves_env;
	var waves_sig, waves_filt;
	var maxNoise;

	// проникновение
	// Harder pops
	pop = TDuty.ar(Dseq({(ux * ux + 0.1).bilinrand.abs} ! loopTime, inf));
	pop = Resonz.ar(
		Decay.ar(pop, TRand.ar(0.001, 0.1, pop)),
		Demand.ar(pop, 0, Dseq({7000.bilinrand.abs + 40} ! loopTime, inf)),
		mul:Demand.ar(pop, 0, Dseq({exprand(0.1, 0.87)} ! loopTime, inf)));
	pop = LeakDC.ar(pop, TRand.ar(0.9, 0.9995, pop));
	// Softer, highpass filtered impulses
	imp = HPF.ar(
		(1 - ux) * LPF.ar(Dust2.ar(8), Rand(8000, 9000)),
		6000.0,
		WhiteNoise.ar(0.7 * (1 - ux)));

	// Faster, modulated lowpass filtered impulses
	modImp = LPF.ar(
		(1 - ux) * Dust.ar(
			LFDNoise3.ar(1).exprange(100, 500),
			LFDNoise3.ar(10).exprange(0.01, 0.07)), Rand(8000, 9000));

	// Mid frequency modulated noise background
	midNoise = LPF.ar((1 - ux) * GrayNoise.ar(LFDNoise3.ar(60)) * WhiteNoise.ar(0.009), 12000.0);

	// High frequency modulated noise background
	hiNoise = HPF.ar((1 - ux) * LPF.ar(GrayNoise.ar(
		LFDNoise3.ar(60)) * WhiteNoise.ar(0.01), 14000.0), Rand(8000, 9000));

	minNoiseSrc = pop + imp + modImp + midNoise + hiNoise;
	minNoiseRev = Greyhole.ar(
		minNoiseSrc,
		\dtime.kr(0.2),
		\damp.kr(0.5),
		\size.kr(1),
		\diff.kr(0.707),
		\fb.kr(0.1),
		\modDepth.kr(0.01),
		\modFreq.kr(2)
	);
	minNoise = (minNoiseSrc * (1 - ux)) + (minNoiseRev * ux);

	// прилегание
	waves_amp = Array.fill(waves_n, {WhiteNoise.kr() * 0.5 + 0.4});
	waves_freq = Array.fill(waves_n, {exprand(1000.0, 3500.0)});
	waves_edur = Array.fill(waves_n, {Rand(3, 6)});
    waves_env = Array.fill(waves_n, { arg i;
		EnvGen.ar(
			Env.new(
				levels: [0, waves_amp.at(i), 0],
				times: [waves_edur.at(i) * ux, waves_edur.at(i) * (1 - ux)],
				curve: [2, -2],
				releaseNode: 2,
				loopNode: -1),
			gate: Trig.kr(Impulse.kr(Rand(1, 2)), dur: waves_edur.at(i) + 0.01)
		);
	});

	waves_sig = PinkNoise.ar(waves_env, 0, 0.1);
	maxNoise = Mix.ar(LPF.ar(waves_sig, waves_freq));

	synths = [
		Silent.ar,
		(uy * minNoise) +
		((1 - uy) * maxNoise) +
		(Ringz.ar(maxNoise, minNoise) * squared(max(0, min(uy, 0.6 - uy)))) +
		(Ringz.ar(minNoise, maxNoise) * squared(max(0, min(1 - uy, uy - 0.3))))
	];

	Out.ar(0, Select.ar(
		(
			(ux >= 0) *
			(ux <= 1) *
			(uy >= 0) *
			(uy <= 1) *
			((uy < 0.5) + (uy > 0.5))
		) > 0,
	synths));
}).add;
)
