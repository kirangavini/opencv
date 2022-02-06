from turtle import color
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,1,500)
y = np.sin(2*np.pi*t)
plt.plot(t,y)
plt.axhline(y=0,color='k')
plt.show()
