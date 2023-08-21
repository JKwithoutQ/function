import matplotlib.pyplot as plt
import numpy as np


# tanh
# ReLU
# Leaky ReLU

def tanh(x: np.ndarray) -> np.ndarray:
    # np.tanh(x)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_grad(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


if __name__ == '__main__':
    x = np.linspace(start=-10, stop=10, num=100)
    y = tanh(x)
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title('tanh')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x, y)
    # plt.show()
    plt.savefig('tanh.png')
