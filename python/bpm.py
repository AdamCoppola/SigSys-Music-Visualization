from numpy import *
import scipy.io.wavfile as wio
from scipy import signal as sig
from scipy import misc

import matplotlib.pyplot as plt
import numpy.linalg as la

# takes frames
# returns a 2D similarity matrix and the duration of the sample
def simMatrix(frames):
	rows = len(frames)

	sim = zeros((rows, rows))
	for i in xrange(0, rows):
		for j in xrange(0, rows):
			fi = frames[i]
			fj = frames[j]
			mag = (la.norm(fi)*la.norm(fj))

			if mag == 0.:
				sim[i][j] = 1
			else:
				sim[i][j] = dot(fi, fj)/(la.norm(fi)*la.norm(fj))

	# plt.matshow(sim, cmap='bone', origin='lower')

	# plt.xlabel('i')
	# plt.ylabel('j')
	# plt.title('Music Similarity Matrix')

	# plt.show()

	return sim

# Finds BPM from a given beat spec
def getBPM(beatSpec, sampleRate):
	BeatSpec = transform(beatSpec)

def transform (data):
	left,right = split(abs(fft.fft(data)),2)
	return add(left,right[::-1])

def autocorr(x):
    corr2d = sig.correlate2d(x, x, mode='same')

    corr = [std(c) for c in corr2d]

    return corr