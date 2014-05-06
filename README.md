SigSys Music Visualization
==========================
For our SigSys final project, we created a music visualizer with Python.

It analyzes a mono wav file and produces a video with a spectrum whose background pulses to the beat. We used `scipy.signal` and `numpy` for most of the signal processing.

To detect the beats we used the method described in [this paper](http://rotorbrain.com/foote/papers/icme2001.pdf).
