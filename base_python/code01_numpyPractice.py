import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 3.0])
print(x)
print(x + y)
print(x * y)
print(type(x))

A = np.array([[1.0, 2.0], [3.0, 4.0]])
print(A)
print(A.shape)
print(A.dtype)
print(A * 10)

print(A[0])
print(A[0][1])
X = A.flatten()  # 将A转为1维数组
print(X.shape)
print(X)

print(X[np.array([0, 2])])  # 获取索引为0,2 的元素

print(X < 3)































