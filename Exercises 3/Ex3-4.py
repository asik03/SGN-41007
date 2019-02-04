# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:52:33 2019

@author: Asier
"""
import numpy as np
import matplotlib.pyplot as plt


f = 0.1 # frequency
t = 900
var = 0.5 # variance of the noising signal

# Version with Gaussian noise and variance 0.5
#np.random.seed(12345678)


# X-axis represents the time
time1 = np.arange(500)
time2 = np.arange(501, 601)
time3 = np.arange(602, 902)

# Y-axis represents amplitude of the wave
a1 = np.zeros(500)
a2 = np.cos(2 * np.pi * f * time2/2)
a3 = np.zeros(300)

# Concatenating  each axis
time_a = np.concatenate((time1, time2, time3))
A = np.concatenate((a1,a2,a3))

# Plotting
plt.plot(time_a, A)
plt.title('Noiselesss signal')
plt.xlim(0, 900)
plt.ylim(-1.0, 1.0)
#plt.axis('tight')
plt.show


# X-axis represents the time
time_b = np.arange(t)
# Y-axis represents amplitude of the wave
y = np.zeros(900)
y_n = y + np.sqrt(var) * np.random.randn(y.size)

# Plotting
plt.plot(time_b, y_n)
plt.title('Noiselesss signal')
plt.xlim(0, 900)
plt.axis('auto')
plt.grid
plt.show

# Detection
h = np.exp(-2 *np.pi * 1j * f * A)
y = np.abs(np.convolve(h, y_n, 'same'))

# Final plot
fig, axis = plt.subplots(3, 1)
axis[0].plot(time_a, A)
axis[0].set_title('Noiseless Signal')
axis[1].plot(time_b, y_n)
axis[1].set_title('Noisy Signal')
axis[2].plot(time_a, y)
axis[2].set_title('Detection Result')
fig.tight_layout()
plt.show