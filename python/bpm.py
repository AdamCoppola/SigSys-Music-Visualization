from numpy import *
import scipy.io.wavfile as wio
from scipy import signal as sig
from scipy import misc

import matplotlib.pyplot as plt
import numpy.linalg as la

# takes frames and returns a 2D similarity matrix
def simMatrix(frames):
	print shape(frames)
	rows = len(frames)

	sim = zeros((rows, rows))
	for i in xrange(0, rows):
		for j in xrange(0, rows):
			fi = frames[i]
			fj = frames[j]
			mag = (la.norm(fi)*la.norm(fj))

			if mag == 0.:
				sim[i][j] = 0
			else:
				sim[i][j] = 1-dot(fi, fj)/(la.norm(fi)*la.norm(fj))

	plt.matshow(sim, cmap='bone', origin='lower')

	plt.show()

	return sim