import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import random
import scipy
import scipy.stats


# defining the potential function V1. 4184 = conversion factor from kcal/mol to J/mol
# we'll be using R (gas constant) instead of k_B to convert to SI units
def V1(x, k=1):
    return k * (x - 5) ** 2 * 4184


# defining the "moves" function. x_0 = initial argument from where to start the "moves"
# pseudo_dx: to generate a random number that can be compared with
# the frequency of selecting either a minimum or maximum step.
# the variables are described below.
def moves(x_0, x_min=0, x_max=10, min_step=0.1, max_step=1, min_freq=0.8):
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

# averages


def av1_x(T):
    def x_P1(x, T):
        return x * P_hist.pdf(x)
    return scipy.integrate.quad(x_P1, x_min, x_max, args=(T,))[0]


def av1_x2(T):
    def x2_P1(x, T):
        return x ** 2 * P_hist.pdf(x)
    return scipy.integrate.quad(x2_P1, x_min, x_max, args=(T,))[0]


def av1_V1(T):
    def V1_P1(x, T):
        return V1(x) * P_hist.pdf(x)
    return scipy.integrate.quad(V1_P1, x_min, x_max, args=(T,))[0]


# arguments (variables)
N = 1000000       # number of iterations
R = 8.3145       # constant gas to be use after we converted the units of energy. Units in J/K*mol
x_min = 0        # lower bound limit of range
x_max = 10       # upper bound limit of range
min_step = 0.1   #
max_step = 1
min_freq = 0.8
ignore = 0.1
accept_ratio = []
T_parameters = [100, 300, 1000]
AVE_X_list = []
AVE_X2_list = []
AVE_V_list = []


for T in T_parameters:

    accepted = []

# initial random value of position, accepted by default
    x_0 = random.uniform(x_min, x_max)

    accepted.append(x_0)

    counter = 1

    V0 = V1(x_0)

    for i in range(N):
        x_t = moves(x_0, x_min=x_min, x_max=x_max, min_step=min_step,
                    max_step=max_step, min_freq=min_freq)
        Vt = V1(x_t)

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


#histogram = np.histogram(accepted, 100, range=(x_min, x_max))
#P_hist = scipy.stats.rv_histogram(histogram)
#x_values = np.arange(x_min, x_max + 0.1, 0.1)
#P_values = [P_hist.pdf(x) for x in x_values]
#plt.plot(x_values, P_values)
# plt.show()


# all temperature together
    legend = [100, 300, 1000]
    histogram = np.histogram(
        accepted[int(ignore * N):], 100, range=(x_min, x_max))
    P_hist = scipy.stats.rv_histogram(histogram)
    x_values = np.arange(x_min, x_max + 0.1, 0.1)
    P_values = [P_hist.pdf(x) for x in x_values]
    plt.plot(x_values, P_values)
    legend.append(str(T))
    #print('Temperature', T)
    av_x = av1_x(T)
    # print(av_x)
    av_x2 = av1_x2(T)
    # print(av_x2)
    av_V = av1_V1(T)
    # print(av_V)
    AVE_X_list.append(av_x)
    AVE_X2_list.append(av_x2)
    AVE_V_list.append(av_V)


plt.xlabel('distance units')
plt.ylabel('Boltzmann Probability distribution density for V1')
plt.legend(legend)
# plt.show()


plt.close()
legend = [100, 300, 1000]
N_range = np.arange(0, N + 1, 1)
plt.plot(N_range, accepted, linestyle='', marker='.')
plt.legend(legend)
plt.ylim([0, 10])
# plt.show()


plt.close()
plt.plot(T_parameters, accept_ratio, linestyle='', marker='o', color='r')
plt.xlabel('Temperature')
plt.ylabel('Percent of acceptance')
# plt.show()

T_range = np.arange(100, 1001, 100)
AVE_X = np.array([av1_x(T) for T in T_range])
AVE_X2 = np.array([av1_x2(T) for T in T_range])
AVE_V = np.array([av1_V1(T) for T in T_range])


plt.close()
plt.plot(T_parameters, AVE_X_list, linestyle='--', marker='o', color='m')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position [distance units]')
plt.ylim([0, 10])
# plt.show()


plt.close()
plt.plot(T_parameters, AVE_X2_list, linestyle='--', marker='o', color='c')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of position squared')
plt.ylim([20, 30])
plt.axhline(y=25.0, color='red', linestyle='dashed')
# plt.show()


plt.close()
plt.plot(T_parameters, AVE_V_list, linestyle='--', marker='o', color='b')
plt.xlabel('Temperature [K]')
plt.ylabel('Average of potential energy [J/mol]')
plt.show()
