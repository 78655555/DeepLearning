import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
             [1.2, 104.0, 52.0, 8.0],
             [1.8, 135.0, 99.0, 0.9]])
print(A)

cal = A.sum(axis=0)  # axis = 0竖向相加 axis = 1 横向相加
print(cal)

# reshape : 来确保你的矩阵是你想要的
percentage = 100 * A / cal.reshape(1, 4)  # 1X4
print(percentage)




