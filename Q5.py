import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


def V3(x, A=10, B=10, sigma_1=1, sigma_2=1):
    return (-A * np.exp(-((x + 3) ** 2 / (sigma_1 ** 2))) - B * np.exp(-((x - 3) ** 2 / (sigma_2 ** 2)))) * 4184


def Boltzmann(x, T):
    return np.exp(-V3(x) / (8.3145 * T))


def Z(T):
    return scipy.integrate.quad(Boltzmann, -10, 10, args=(T,))


def P3(x, T):
    return Boltzmann(x, T) / Z(T)[0]


X = np.arange(-10, 10, 0.01)


V = np.array([V3(x) for x in X])

print(V3(5))
print(Boltzmann(3, 100))

plt.plot(X, V)
plt.xlabel('distance units')
plt.ylabel('Potential energy [J/mol]')
plt.show()


T_parameters = [100, 300, 1000]
legend = []
for T in T_parameters:
    P = np.array([P3(x, T) for x in X])
    plt.plot(X, P)
    legend.append(str(T))
plt.xlabel('distance units')
plt.ylabel('Boltzmann Probability density distribution for V3')
plt.legend(legend)
plt.axhline(y=1.0, color='red', linestyle='dashed')
plt.show()


def av3_x(T):
    def x_P3(x, T):
        return x * P3(x, T)
    return scipy.integrate.quad(x_P3, -10, 10, args=(T,))[0]


# print(av3_x(100))
# print(av3_x(300))
# print(av3_x(1000))

T_range = np.arange(100, 1001, 100)

AVE3_X = np.array([av3_x(T) for T in T_range])

plt.plot(T_range, AVE3_X, linestyle='--', marker='o', color='m')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position [distance units]')
plt.show()


def av3_x2(T):
    def x2_P3(x, T):
        return x ** 2 * P3(x, T)
    return scipy.integrate.quad(x2_P3, -10, 10, args=(T,))[0]


print(av3_x2(100))
print(av3_x2(300))
print(av3_x2(1000))

T_range = np.arange(100, 1001, 100)

AVE3_X2 = np.array([av3_x2(T) for T in T_range])


plt.plot(T_range, AVE3_X2, linestyle='--', marker='o', color='c')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position squared [distance units squared]')
plt.show()


def av3_V3(T):
    def V3_P3(x, T):
        return V3(x) * P3(x, T)
    return scipy.integrate.quad(V3_P3, -10, 10, args=(T,))[0]


print(av3_V3(100))
print(av3_V3(300))
print(av3_V3(1000))


T_range = np.arange(100, 1001, 100)

AVE3_V3 = np.array([av3_V3(T) for T in T_range])


plt.plot(T_range, AVE3_V3, linestyle='--', marker='o', color='b')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of potential energy [J/mol]')
plt.show()


# Max

x_range = np.arange(-3, 3.001, 0.0001)
V3_range = np.array([V3(x) for x in x_range])
x_barrier = x_range[np.argmax(V3_range)]

print(x_barrier)


def P_left(T, x_barrier=x_barrier):
    return scipy.integrate.quad(P3, -10, x_barrier, args=(T,))[0]


def P_right(T, x_barrier=x_barrier):
    return scipy.integrate.quad(P3, x_barrier, 10, args=(T,))[0]


print(P_left(100))
print(P_left(300))
print(P_left(1000))

print(P_right(100))
print(P_right(300))
print(P_right(1000))
