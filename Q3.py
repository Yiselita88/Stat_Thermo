import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


def V2(x, D_e=10, alpha=0.31, x_0=5):
    return (D_e * (1 - np.exp(-alpha * (x - x_0))) ** 2) * 4184


def Boltzmann(x, T):
    return np.exp(-V2(x) / (8.3145 * T))


def Z(T):
    return scipy.integrate.quad(Boltzmann, 0, 10, args=(T,))


def P2(x, T):
    return Boltzmann(x, T) / Z(T)[0]


X = np.arange(0, 10, 0.001)


V = np.array([V2(x) for x in X])

print(V2(5))

plt.plot(X, V)
plt.xlabel('distance units')
plt.ylabel('Potential energy [J/mol]')
# plt.show()

T_parameters = [100, 300, 1000]
legend = []
for T in T_parameters:
    P = np.array([P2(x, T) for x in X])
    plt.plot(X, P)
    legend.append(str(T))
plt.xlabel('distance units')
plt.ylabel('Boltzmann Probability distribution for V2')
plt.legend(legend)
# plt.show()


def av2_x(T):
    def x_P2(x, T):
        return x * P2(x, T)
    return scipy.integrate.quad(x_P2, 0, 10, args=(T,))[0]


print(av2_x(100))
print(av2_x(300))
print(av2_x(1000))


T_range = np.arange(100, 1001, 100)

AVE2_X = np.array([av2_x(T) for T in T_range])

plt.close()
plt.plot(T_range, AVE2_X, linestyle='--', marker='o', color='m')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position [distance units]')
plt.axhline(y=5.0, color='red', linestyle='dashed')
plt.ylim([0, 10])
plt.show()


def av2_x2(T):
    def x2_P2(x, T):
        return x ** 2 * P2(x, T)
    return scipy.integrate.quad(x2_P2, 0, 10, args=(T,))[0]


print(av2_x2(100))
print(av2_x2(300))
print(av2_x2(1000))

T_range = np.arange(100, 1001, 100)

AVE2_X2 = np.array([av2_x2(T) for T in T_range])

plt.plot(T_range, AVE2_X2, linestyle='--', marker='o', color='c')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position squared [distance units squared]')
plt.axhline(y=25.0, color='red', linestyle='dashed')
# plt.show()


def av2_V2(T):
    def V2_P2(x, T):
        return V2(x) * P2(x, T)
    return scipy.integrate.quad(V2_P2, 0, 10, args=(T,))[0]


print(av2_V2(100))
print(av2_V2(300))
print(av2_V2(1000))


T_range = np.arange(100, 1001, 100)

AVE2_V2 = np.array([av2_V2(T) for T in T_range])

plt.plot(T_range, AVE2_V2, linestyle='--', marker='o', color='b')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of potential energy [J/mol]')
# plt.show()
