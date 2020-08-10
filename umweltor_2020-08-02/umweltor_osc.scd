// init
(
//increase number of buffers the server has access to for loading samples
s.options.numBuffers = 1024 * 16;
//increase the memory available to the server
s.options.memSize = 8192 * 64;
s.options.sampleRate = 48000;
// check that mic is in 48000 hz mode
// s.options.inDevice_();
// s.options.outDevice_("ZoomAudioD");
// s.options.inDevice_("Soundflower (2ch)");
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

h = Synth(\umweltor_synth, [\ux, -1, \uy, -1]);

(
z = OSCFunc({ |msg, time, addr, port|
    var x, y;
    x = msg[1].asFloat;
	y = msg[2].asFloat;
    (x @ y).postln;
	h.set(\ux, x, \uy, y);
}, "/umweltor");
)
