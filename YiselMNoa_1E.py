import numpy as np
from math import e
ln = np.log

print(ln(e))

for n in range(1, 21):
    x = np.math.factorial(n)
    print(x)
    y = ln(np.math.factorial(n))
    print(y)
    s = n * ln(n) - n
    print(s)
    err = (y - s) / y
    if err < 0.01:
        print(y, s, err)
        break
