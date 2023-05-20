import numpy as np
import matplotlib.pyplot as plt


def factor(n):
    return np.math.factorial(n)


X = np.arange(1, 6, 1)

Y = np.array([factor(n) for n in X])

plt.plot(X, Y)
plt.xlabel('N')
plt.ylabel('N!')
# plt.savefig('N-factorial_vs_N.jpg')
plt.show()
