import sys , os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


# 读入minist数据集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)







