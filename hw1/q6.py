import numpy as np


def func(n):
    if n == 0:
        return 2
    elif n == 1:
        return 1
    else:
        return func(n-1) + func(n-2)

result = func(32)
print(result)
