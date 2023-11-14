import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 1000).astype(float)

r = 50 * (1.0 + np.sin(t / 100.0))
g = 50 * (1.0 + np.cos(t / 100.0))
b = 50 * (1.0 + np.sin(t / 100.0+3.14))

plt.plot(t, r)
plt.plot(t,g)
plt.plot(t, b)

plt.show()