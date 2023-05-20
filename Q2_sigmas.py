import numpy as np
import matplotlib.pyplot as plt


def gauss(x, sigma=1, mu=0):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


g = gauss(0, sigma=1)
print(g)

X = np.arange(-10, 10.01, 0.01)

Y = np.array([gauss(x) for x in X])

plt.plot(X, Y)
plt.xlabel('x')
plt.ylabel('Normalized Gaussian Distribution')
#plt.legend(['Stirling', 'Lineal'])
plt.savefig('Gaussian.jpg')
plt.show()
