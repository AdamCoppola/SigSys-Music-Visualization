from numpy import *
import scipy.io.wavfile as wio
from scipy import signal, misc
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import time

def updatePlot (handle, data):
	handle.set_ydata(data)
	return handle

def process (data, window):
	windows = len(data)/window
	for i in xrange(0, windows-2):
		transformed = transform(data[i*window:(i+1)*window])
		plotFFT(transformed)

def makeEven (num):
	if num % 2:
		return num-1
	else:
		return num

def transform (data):
	left,right = split(abs(fft.fft(data)),2)
	ys = add(left,right[::-1])
	return ys

def plotFFT (data):
	plt.plot(data, hold="on")
	plt.show()
	time.sleep(.1)
	plt.clf()

rate, audio = wio.read('../wav/pompeii.wav')
laudio, raudio = zip(*audio)

process(laudio, makeEven(rate))