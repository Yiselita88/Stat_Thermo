def factorial(N):
    if N == 1:
        result = 1
    else:
        result = N * factorial(N - 1)
    return result


#num = input("Aqui va el numero: ")
# print(factorial(int(num)))
print(factorial(4))
