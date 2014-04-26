from numpy import *
import scipy.io.wavfile as wio
from scipy import signal, misc
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import time

def makeEven (num):
	if num % 2:
		return num-1
	else:
		return num

def process (data, window):
	windows = len(data)/window
	print 'WINDOWS'
	print windows
	spectrogram = empty(shape=(windows, window/2))
	for i in xrange(0, windows-2):
		spectrogram[i]=transform(data[i*window:(i+1)*window])
	return spectrogram

def transform (data):
	left,right = split(abs(fft.fft(data)),2)
	ys = add(left,right[::-1])
	return ys

rate, audio = wio.read('../wav/pompeii.wav')
print "Done reading"
laudio, raudio = zip(*audio)
print "Done zipping"

spec = process(laudio, makeEven(rate))

plt.figure(1)
plt.matshow(spec, 1)

plt.show()