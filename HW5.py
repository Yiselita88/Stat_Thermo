import numpy as np


def Helmholtz(m, kB=1.38064852 / 10**23, N=6.0221409 * 10**23, h=6.62607004 / 10**34, pi=3.141592654, T=298.15):
    return (- N * kB * T * np.log(((2 * pi * m * kB * T / (h**2))**1.5) * kB * T * np.exp(1)))


A_Kr = Helmholtz(m=1.3914984 / 10 ** 25)
A_Ne = Helmholtz(m=3.3509177 / 10 ** 26)

print(A_Kr)
print(A_Ne)


def Energy(kB=1.38064852 / 10**23, N=6.0221409 * 10**23, T=298.15):
    return (N * kB * T * 3 / 2)


E = Energy()

print(E)


def C_v(kB=1.38064852 / 10**23, N=6.0221409 * 10**23):
    return (N * kB * 3 / 2)


C_v = C_v()

print(C_v)


def Entropy(A, E, T=298.15):
    return ((E - A) / T)


S_Kr = Entropy(A=A_Kr, E=E)
S_Ne = Entropy(A=A_Ne, E=E)

print(S_Kr)
print(S_Ne)


def Gibbs(A, kB=1.38064852 / 10**23, N=6.0221409 * 10**23, T=298.15):
    return (N * kB * T + A)


G_Kr = Gibbs(A=A_Kr)
G_Ne = Gibbs(A=A_Ne)

print(G_Kr)
print(G_Ne)


def Chem_Potential(m, kB=1.38064852 / 10**23, N=6.0221409 * 10**23, h=6.62607004 / 10**34, pi=3.141592654, T=298.15):
    return (- kB * T * np.log(((2 * pi * m * kB * T / (h**2))**1.5) * kB * T))


mu_Kr = Chem_Potential(m=1.3914984 / 10 ** 25)
mu_Ne = Chem_Potential(m=3.3509177 / 10 ** 26)

print(mu_Kr)
print(mu_Ne)


def degeneracy(S, kB=1.38064852 / 10**23):
    return(np.exp(S / kB))


#W = degeneracy(S=S_Ne)

# print(W)

# Q3 Thermal deBroglie


def deBroglie(m, T, kB=1.38064852 / 10**23, N=6.0221409 * 10**23, h=6.62607004 / 10**34, pi=3.141592654):
    return (h * 10**9 / np.sqrt(2 * pi * m * kB * T))


lambda_e1 = deBroglie(m=9.10938356 / 10**31, T=1)
lambda_e300 = deBroglie(m=9.10938356 / 10**31, T=300)
lambda_Ar1 = deBroglie(m=6.6335209 / 10**26, T=1)
lambda_Ar300 = deBroglie(m=6.6335209 / 10**26, T=300)

print(lambda_e1)
print(lambda_e300)
print(lambda_Ar1)
print(lambda_Ar300)
