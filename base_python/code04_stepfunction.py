import numpy as np
import matplotlib.pyplot as plt


def step_functipon(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_functipon(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y轴范围
plt.show()








