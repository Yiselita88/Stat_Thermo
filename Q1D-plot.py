import numpy as np
import matplotlib.pyplot as plt


def fact(n):
    return np.math.factorial(n)


def ln(x):
    return np.log(x)


# print(ln(fact(5)))

X = np.arange(1, 6, 1)

Y = np.array([ln(fact(x)) for x in X])

plt.plot(X, Y)
plt.xlabel('N')
plt.ylabel('ln(N!)')
# plt.savefig('lnN-fact_vs_N.jpg')
plt.show()
