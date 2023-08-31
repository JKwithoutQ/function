from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sklearn import metrics


class Metric(object):

    def __init__(
            self,
            class_num: int,
            class_name: Union[List, str, None] = None,
            save_path: Union[str, None] = None
    ) -> None:
        if class_name is None:
            class_name = [_i for _i in range(class_num)]
        self.class_name = class_name
        self.class_num = class_num
        self.save_path = save_path
        self.prob = []
        self.lab = []
        columns = ['accuracy', 'recall', 'precision', 'f1-score']
        self.df = pd.DataFrame(index=self.class_name, columns=columns)

    def init(self):
        self.prob = []
        self.lab = []
        self.df = pd.DataFrame(index=self.class_name)

    def append(self, prob: Union[np.ndarray, torch.Tensor], lab: Union[np.ndarray, torch.Tensor]):
        prob = prob.cpu().numpy() if isinstance(prob, torch.Tensor) else prob
        lab = lab.cpu().numpy() if isinstance(lab, torch.Tensor) else lab
        # if lab.ndim == 1:
        #     lab = lab.reshape(-1, 1)
        self.prob.append(prob)
        self.lab.append(lab)

    def accuracy(self):
        print(np.array(self.prob).argmax(-1).reshape(-1))
        print(np.array(self.lab).reshape(-1))
        rst = metrics.accuracy_score(
            y_true=np.array(self.lab).reshape(-1),
            y_pred=np.array(self.prob).argmax(-1).reshape(-1)
        )
        print(self.df)
        return rst

    def classification_report(self):
        rst = metrics.classification_report(
            y_true=np.array(self.lab).reshape(-1),
            y_pred=np.array(self.prob).argmax(-1).reshape(-1),
            labels=np.array([_i for _i in range(10)]),
            target_names=[str(_i) for _i in range(10, 20)],
            output_dict=True
        )
        return rst

    def classification(self):
        print(self.lab)
        lab = np.array(self.lab).reshape(-1)
        prob = np.array(self.prob).argmax(-1).reshape(-1)
        print(lab, prob)
        for idx in range(self.class_num):
            _lab = np.where(lab == idx, 1, 0)
            _prob = np.where(prob == idx, 1, 0)
            true = np.logical_and(_lab, _prob)
            self.df.loc[idx, 'recall'] = np.divide(true.sum(), _lab.sum() + 1e-6)
            self.df.loc[idx, 'precision'] = np.divide(true.sum(), _prob.sum() + 1e-6)
            self.df.loc[idx, 'accuracy'] = true.mean()
        print(self.df)

    def save(self, path: str):


if __name__ == '__main__':
    num0 = np.random.random((5, 10))
    lab0 = np.random.randint(0, 10, 5)
    num1 = np.random.random((5, 10))
    lab1 = np.random.randint(0, 10, 5)
    m = Metric(10)
    m.append(num0, lab0)
    m.append(num1, lab1)
    t = m.classification()
    print(type(t))
    # df = pd.DataFrame(data=t)
    # print(df)
