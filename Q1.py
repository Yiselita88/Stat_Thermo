import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# we're defining:
# 1) the potential function along with parameter k. 4184 is the conversion
# factor to go from kcal/mol to J/mol.
# 2) the Boltzmann factor for that potential function. 8.3145 is the gas
# constant, which was included to match the units
# 3) the partition function for the potential function. the comma after
# the "T" is to avoid errors with tuples
# 4) the Boltzmann distribution probability. This integral yields two numbers:
# the value of the integral and the error, that's why we included [0]:
# to indicate to print only the value of the integral


def V1(x, k=1):
    return k * (x - 5) ** 2 * 4184


def Boltzmann(x, T):
    return np.exp(-V1(x) / (8.3145 * T))


def Z(T):
    return scipy.integrate.quad(Boltzmann, 0, 10, args=(T,))


def P1(x, T):
    return Boltzmann(x, T) / Z(T)[0]


print(V1(3))

X = np.arange(0, 10, 0.01)

V = np.array([V1(x) for x in X])


plt.plot(X, V)
plt.xlabel('distance units')
plt.ylabel('Potential energy [J/mol]')
# plt.show()

T_parameters = [100, 300, 1000]
legend = []
for T in T_parameters:
    P = np.array([P1(x, T) for x in X])
    plt.plot(X, P)
    legend.append(str(T))
plt.xlabel('distance units')
plt.ylabel('Boltzmann Probability density distribution for V1')
plt.legend(legend)
# plt.show()


def int_P(T):
    def x_P1(x, T):
        return P1(x, T)
    return scipy.integrate.quad(x_P1, 0, 10, args=(T,))[0]


def av1_x(T):
    def x_P1(x, T):
        return x * P1(x, T)
    return scipy.integrate.quad(x_P1, 0, 10, args=(T,))[0]


def av1_x2(T):
    def x2_P1(x, T):
        return x ** 2 * P1(x, T)
    return scipy.integrate.quad(x2_P1, 0, 10, args=(T,))[0]


def av1_V1(T):
    def V1_P1(x, T):
        return V1(x) * P1(x, T)
    return scipy.integrate.quad(V1_P1, 0, 10, args=(T,))[0]


print(av1_x(100))
print(av1_x(300))
print(av1_x(1000))

T_range = np.arange(100, 1001, 100)
AVE_X = np.array([av1_x(T) for T in T_range])


plt.close()
plt.plot(T_range, AVE_X,)
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position [distance units]')
# plt.show()

print(av1_x2(100))
print(av1_x2(300))
print(av1_x2(1000))

AVE_X2 = np.array([av1_x2(T) for T in T_range])

plt.close()
plt.plot(T_range, AVE_X2, linestyle='--', marker='o', color='c')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position squared [distance units squared]')
plt.axhline(y=25.0, color='red', linestyle='dashed')
plt.ylim([20, 30])
plt.show()

print(av1_V1(100))
print(av1_V1(300))
print(av1_V1(1000))

AVE_V = np.array([av1_V1(T) for T in T_range])

plt.close()
plt.plot(T_range, AVE_V, linestyle='--', marker='o', color='b')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of potential energy [J/mol]')
# plt.show()


def int_P(T):
    def x_P1(x, T):
        return P1(x, T)
    return scipy.integrate.quad(x_P1, 0, 10, args=(T,))[0]
