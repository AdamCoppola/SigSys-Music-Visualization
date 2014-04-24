import numpy as np

import scipy.io.wavfile as wio

from scipy import signal, misc

import matplotlib.pyplot as plt

t = np.linspace(0, 5, 1000)
sigchirp = signal.chirp(t, 10, 3, 15, method='logarithmic')

plt.plot(t, sigchirp)

wio.write('chirp.wav', 40000, sigchirp)

plt.show()