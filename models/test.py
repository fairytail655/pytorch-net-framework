import numpy as np
import matplotlib.pyplot as plt

n = 1
k = 1
x = np.arange(-10, 10, 0.01)
y = np.tanh(n*(x-k))

plt.plot(x, y)
plt.show()

print(y)