import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import random
import scipy
import scipy.stats


def V3(x, A=10, B=10, sigma_1=1, sigma_2=1):
    return (-A * np.exp(-((x + 3) ** 2 / (sigma_1 ** 2))) - B * np.exp(-((x - 3) ** 2 / (sigma_2 ** 2)))) * 4184


def moves(x_0, x_min=-10, x_max=10, min_step=0.1, max_step=1, min_freq=0.8):
    pseudo_dx = np.random.uniform(0, 1)
    if pseudo_dx <= min_freq:
        dx = min_step
    else:
        dx = max_step
    rho = np.random.uniform(-1, 1)
    x_t = x_0 + rho * dx
    if x_t >= x_min and x_t <= x_max:
        return x_t
    else:
        while x_t < x_min or x_t > x_max:
            rho = np.random.uniform(-1, 1)
            x_t = x_0 + rho * dx
        return x_t


def av3_x(T):
    def x_P3(x, T):
        return x * P_hist.pdf(x)
    return scipy.integrate.quad(x_P3, x_min, x_max, args=(T,))[0]


def av3_x2(T):
    def x3_P3(x, T):
        return x ** 2 * P_hist.pdf(x)
    return scipy.integrate.quad(x3_P3, x_min, x_max, args=(T,))[0]


def av3_V3(T):
    def V3_P3(x, T):
        return V3(x) * P_hist.pdf(x)
    return scipy.integrate.quad(V3_P3, x_min, x_max, args=(T,))[0]


N = 1000
R = 8.3145
x_min = -10
x_max = 10
min_step = 0.1
max_step = 8
min_freq = 0.5
ignore = 0.1
accept_ratio = []
T_parameters = [100, 300, 1000]
AVE_X_list = []
AVE_X2_list = []
AVE_V_list = []
P_left = []
P_right = []


for T in T_parameters:

    accepted = []

    x_0 = random.uniform(x_min, x_max)

    accepted.append(x_0)

    counter = 1

    V0 = V3(x_0)

    for i in range(N):
        x_t = moves(x_0, x_min=x_min, x_max=x_max, min_step=min_step,
                    max_step=max_step, min_freq=min_freq)
        Vt = V3(x_t)

        if Vt <= V0:
            counter = counter + 1
            accepted.append(x_t)
            x_0 = x_t
            V0 = Vt

        else:
            rdm = np.random.uniform(0, 1)
            beta = np.exp(- (Vt - V0) / (R * T))
            if beta >= rdm:
                counter = counter + 1
                accepted.append(x_t)
                x_0 = x_t
                V0 = Vt
            else:
                accepted.append(x_0)
    accept_ratio.append(100 * counter / N)

    legend = [100, 300, 1000]
    histogram = np.histogram(
        accepted[int(ignore * N):], 100, range=(x_min, x_max))
    P_hist = scipy.stats.rv_histogram(histogram)
    x_values = np.arange(x_min, x_max + 0.1, 0.1)
    P_values = [P_hist.pdf(x) for x in x_values]
    plt.plot(x_values, P_values)
    legend.append(str(T))
    #print('Temperature', T)
    av_x = av3_x(T)
    # print(av_x)
    av_x2 = av3_x2(T)
    # print(av_x2)
    av_V = av3_V3(T)
    # print(av_V)
    AVE_X_list.append(av_x)
    AVE_X2_list.append(av_x2)
    AVE_V_list.append(av_V)
    P_left.append(scipy.integrate.quad(P_hist.pdf, x_min, 0, args=())[0])
    P_right.append(scipy.integrate.quad(P_hist.pdf, 0, x_max, args=())[0])


plt.xlabel('distance units')
plt.ylabel('Boltzmann Probability distribution density for V1')
plt.legend(legend)
plt.show()


plt.close()
legend = [100, 300, 1000]
N_range = np.arange(0, N + 1, 1)
plt.plot(N_range, accepted, linestyle='', marker='.')
plt.legend(legend)
plt.show()


plt.close()
plt.plot(T_parameters, accept_ratio, linestyle='', marker='o', color='r')
plt.xlabel('Temperature')
plt.ylabel('Percent of acceptance')
plt.show()

T_range = np.arange(100, 1001, 100)
AVE2_X = np.array([av3_x(T) for T in T_range])
AVE2_X2 = np.array([av3_x2(T) for T in T_range])
AVE2_V2 = np.array([av3_V3(T) for T in T_range])


plt.close()
plt.plot(T_parameters, AVE_X_list, linestyle='--', marker='o', color='m')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position [distance units]')
plt.ylim([-10, 10])
plt.show()


plt.close()
plt.plot(T_parameters, AVE_X2_list, linestyle='--', marker='o', color='c')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position squared')
plt.show()


plt.close()
plt.plot(T_parameters, AVE_V_list, linestyle='--', marker='o', color='b')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of potential energy [J/mol]')
plt.show()

print(P_left, P_right)
