import numpy as np
import matplotlib.pyplot as plt


#def stir(x):
#    return x*np.log(x) - x
#
#def lin(x):
#    return x
#
#X = np.arange(1, 21, 1)
#
#Y1 = np.array([stir(x) for x in X])
#Y2 = np.array([lin(x) for x in X])
#
#plt.plot(X, Y1)
##plt.plot(X, Y2)
#plt.xlabel('N')
#plt.ylabel('Values')
##plt.legend(['Stirling', 'Lineal'])
#plt.savefig('nombre.jpg')


def l(x, m=1, b=0):
    return m * x + b

X = np.arange(1, 21, 1)

m_values = [0.1, 1, 2]
legend = []
for m in m_values:
    Y = np.array([l(x, m=m) for x in X])
    plt.plot(X, Y)
    legend.append(str(m))
plt.xlabel('x')
plt.ylabel('y')  
plt.legend(legend)
plt.savefig('lineas.jpg')