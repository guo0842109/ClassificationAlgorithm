import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch

class SingleCoin(nn.Module):
    def __init__(self):
        super(SingleCoin, self).__init__()
        self.p = nn.Parameter(torch.tensor(np.random.rand(1)))

    def _prob(self, result):
        if result == 1:
            return self.p
        elif result == 0:
            return 1 - self.p
        else:
            raise Exception('硬币只有正反面，参数错误')

    def forward(self, observed):
        p_total = None
        for ob_i in observed:
            p = self._prob(ob_i)
            if p_total:
                p_total = p * p_total
            else:
                p_total = p
        return p_total

class DoubleCoins(nn.Module):
    def __init__(self):
        super(DoubleCoins, self).__init__()
        self.p = nn.Parameter(torch.tensor(np.random.rand(1)))

    def _prob(self, result):
        if result == 1:
            return self.p * self.p
        elif result == 0:
            return 1 - self.p * self.p
        else:
            raise Exception('硬币只有正反面，参数错误')

    def forward(self, observed):
        p_total = None
        for ob_i in observed:
            p = self._prob(ob_i)
            if p_total:
                p_total = p * p_total
            else:
                p_total = p
        return p_total

class PentaCoins(nn.Module):
    def __init__(self):
        super(PentaCoins, self).__init__()
        self.p = nn.Parameter(torch.tensor(np.random.rand(1)))

    def _prob(self, result):
        return torch.pow(self.p, result) * torch.pow(1 - self.p, 5 - result)

    def forward(self, observed):
        p_total = None
        for ob_i in observed:
            p = self._prob(ob_i)
            if p_total:
                p_total = p * p_total
            else:
                p_total = p
        return p_total

if __name__ == '__main__':
    # model = SingleCoin()
    # model = DoubleCoins()
    model = PentaCoins()
    # ob = [1,1,1,0,1,1,0] # for coin game 1
    # ob = [1, 1, 0, 1, 0, 0]
    ob = [3,2,4,1,2,3,4,5,1,3,2]
    optimizer = Adam(model.parameters())

    epochs = 10000
    for _ in range(epochs):
        pos = model.forward(ob)
        loss = -torch.log(pos)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('coin possibility:', model.p.data.item())