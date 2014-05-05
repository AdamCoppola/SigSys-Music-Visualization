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

def acorr(mat):
	print "BEGAN AUTOCORRELATION"
	X = len(mat)
	sums = zeros(X)

	for x in xrange(0, X):
		print x
		for i in xrange(0, X):
			for j in xrange(0, len(mat[i])):
				sums[i] += mat[i][j]*mat[(i+x)%X][(j+x)%X]
	print "FINISHED AUTOCORRELATION"

	return sums


# Finds BPM from a given beat spec
def getBPM(beatSpec, sampleRate):
	BeatSpec = transform(beatSpec)

def transform (data):
	left,right = split(abs(fft.fft(data)),2)
	return add(left,right[::-1])