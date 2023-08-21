import matplotlib.pyplot as plt
import numpy as np


def Leaky_ReLU(x: np.ndarray, alpha=0.1) -> np.ndarray:
    return np.maximum(alpha * x, x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


if __name__ == '__main__':
    x = np.linspace(start=-10, stop=10, num=100)
    y = Leaky_ReLU(x)
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title('Leaky ReLU')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x, y)
    # plt.show()
    plt.savefig('Leaky_ReLU.png')
