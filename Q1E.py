import numpy as np
import matplotlib.pyplot as plt


def fact(N):
    return np.math.factorial(N)


print(fact(5))


def ln(x):
    return np.log(x)


print(ln(10))


def strlg(x):
    return x * ln(x) - x


print(strlg(2))

X = np.arange(1, 21, 1)

Y1 = np.array([strlg(x) for x in X])

#plt.plot(X, Y1)
# plt.xlabel('N')
# plt.ylabel('ln(N!)')
#plt.legend(['Stirling Approximation'])
# plt.savefig('Stirling.jpg')
# plt.show()


def ln_fact(x):
    return ln(fact(x))


# print(ln_fact(5))

Y2 = np.array([ln_fact(x) for x in X])

#plt.plot(X, Y1)
#plt.plot(X, Y2)
# plt.xlabel('N')
# plt.ylabel('ln(N!)')
#plt.legend(['Stirling Approximation', 'ln(N!) directly calculated'])
# plt.savefig('Stirling_vs_classical.jpg')
# plt.show()


def err(x):
    return (ln_fact(x) - strlg(x)) / ln_fact(x)


print(err(2))


Y3 = np.array([err(x) for x in X])
plt.plot(X, Y3)
#plt.plot(X, Y2)
plt.xlabel('N')
plt.ylabel('Errors')
#plt.legend(['Stirling', 'Lineal'])
# plt.savefig('errors.jpg')
plt.show()
