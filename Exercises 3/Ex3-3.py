# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:52:33 2019

@author: Asier
"""
import numpy as np
import matplotlib.pyplot as plt


f = 0.1 # frequency

# X-axis represents the time
time1 = np.arange(500)
time2 = np.arange(501, 601)
time3 = np.arange(602, 902)

# Y-axis represents amplitude of the wave
a1 = np.zeros(500)
a2 = np.cos(2 * np.pi * f * time2/2)
a3 = np.zeros(300)

# Concatenating  each axis
time = np.concatenate((time1, time2, time3))
A = np.concatenate((a1,a2,a3))

# Plotting
plt.plot(time, A)
plt.title('Noiselesss signal')
plt.xlim(0, 900)
plt.ylim(-1.0, 1.0)
plt.axis('tight')
plt.show





