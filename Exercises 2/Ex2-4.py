import numpy as np
import matplotlib.pyplot as plt

N = 100
f = 0.017

n = np.arange(N)

w = np.sqrt(0.25) * np.random.randn(100)
x = np.sin(2 * np.pi * f * n) + w

fig = plt.figure(figsize=[10, 5])

plt.plot(n, x, linewidth = 2, label = "Noisy Sinusoid")
plt.grid("on")
plt.xlabel("Time in $\mu$s")
plt.ylabel("Amplitude")
plt.title("An Example Plot")
plt.legend(loc = "upper left")
plt.show()


scores = []
frequencies = []
for f in np.linspace(0, 0.5, 1000):
    # Create vector e. Assume data is in x.
    n = np.arange(100)
    z = -2 * np.pi * 1j * f * n     # <compute -2*pi*i*f*n. Imaginary unit is 1j>
    e = np.exp(z)
    score = np.abs(np.dot(x, e))           # <compute abs of dot product of x and e>
    scores.append(score)
    frequencies.append(f)
fHat = frequencies[np.argmax(scores)]

print(fHat)