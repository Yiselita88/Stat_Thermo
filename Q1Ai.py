# method 1, using "for" loops and range function
def factorial(N):
    start = 1
    for k in range(N, 0, -1):
        start = start * k
    return start


print(factorial(5))
