# method 1, using "for" loops and range function
N = int(input("here goes the number:"))
start = 1
for k in range(N, 0, -1):
    start = start * k

print("factorial of", N, "is", start)
