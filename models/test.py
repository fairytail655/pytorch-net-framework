import numpy as np
import matplotlib.pyplot as plt

v = 4
k = 0
x = np.arange(-10, 10, 0.01)
y = np.tanh(v*(x-k))
order = np.ones(x.shape, dtype=int)*2
z = v*(1 - np.power(y, 2))

plt.plot(x, z)
plt.show()