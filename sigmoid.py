import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


if __name__ == '__main__':
    x = np.linspace(start=-10, stop=10, num=100)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title('sigmoid')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x, y)
    # plt.show()
    plt.savefig('sigmoid.png')
