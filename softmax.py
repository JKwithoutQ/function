import matplotlib.pyplot as plt
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


if __name__ == '__main__':
    x = np.linspace(start=-10, stop=10, num=100)
    y = softmax(x)
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_title('softmax')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x, y)
    # plt.show()
    plt.savefig('softmax.png')
