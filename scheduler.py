import matplotlib.pyplot as plt


def scheduler(iter_num, max_iter, gamma=10, power=0.75) -> float:
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    return decay


if __name__ == '__main__':
    max_iter = 10000
    lr0 = 0.1
    decay_lst = []
    lr_lst = [lr0]
    for step in range(max_iter):
        decay = scheduler(step, max_iter)
        decay_lst.append(decay)
        if step % 100 == 0:
            lr0 *= decay
            lr_lst.append(lr0)
    fig, ax = plt.subplots(1, 2)
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('decay')
    ax[0].plot(decay_lst, '*')
    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('lr')
    ax[1].plot(lr_lst, '*')
    plt.show()
