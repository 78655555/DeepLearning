import numpy as np

# 不要这么写 不要用秩为1的数组
a = np.random.rand(5)  # 生成五个随机数放在a数组中
print(a)  # 秩为1的数组
print(a.shape)
print(a.T)  # a的转置
print(np.dot(a, a.T))

# 正确写法
a = np.random.rand(5, 1)
print(a)
print(a.T)
print(np.dot(a, a.T))

# 不确定维数时:
assert (a.shape == (5, 1))
a.reshape(1, 5)

