import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.fft import fft, ifft


datafile = pd.read_csv("Translation_Feature/1_point.txt",header=None)
#cap = cv2.VideoCapture("Slomo_video2.mp4")
#cap = cv2.VideoCapture("Tuning Fork in Slow Motion.mp4")
fps = 2500
#fps = 6667
N = len(datafile.iloc[:,0])
n = np.arange(N)
T = N/fps
freq = n/T
fig, axs = plt.subplots(2,2)
axs[0,0].plot(n/fps,datafile.iloc[:,0],'r')
axs[0,0].set_xlabel('Time(s)')
axs[0,0].set_ylabel('Displacement(in pixel)')
axs[0,0].grid()
#axs[0,1].plot(freq,np.abs(fft(datafile.iloc[:,0])))
axs[0,1].stem(freq,np.abs(fft(datafile.iloc[:,0])),'b', markerfmt=" ", basefmt="-b")
axs[0,1].set_xlabel('Frequency(Hz)')
axs[0,1].set_xlim(0,500)
axs[0,1].set_ylabel('Amplitude')
axs[0,1].grid()
axs[1,0].plot(n/fps,datafile.iloc[:,1])
axs[1,0].set_xlabel('Time(s)')
axs[1,0].set_ylabel('Displacement(pixel)')
axs[1,0].grid()
axs[1,1].stem(freq,np.abs(fft(datafile.iloc[:,1])),'b', markerfmt=" ", basefmt="-b")
axs[1,1].grid()
axs[1,1].set_xlim(0,500)
axs[1,1].set_xlabel('Frequency(Hz)')
axs[1,1].set_ylabel('Amplitude')
plt.show()