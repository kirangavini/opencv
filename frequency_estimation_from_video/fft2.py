import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft

plt.style.use('seaborn-poster')

# sampling rate
sr = 3000
# sampling interval
ts = 1.0/sr
t = np.linspace(0,1,sr)

freq = 1
x = np.cos(2*np.pi*10*freq*t)
freq = 3
x += 2*np.sin(2*np.pi*10*freq*t)
freq = 7
x += np.sin(2*np.pi*10*freq*t)

X = fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

'''
fig, axs = plt.subplots(2,1)

axs[0,0].stem(freq , np.abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)
'''
plt.plot(t, x, 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()