import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch

class Griddle(nn.Module):
    def __init__(self):
        super(Griddle, self).__init__()
        self.grid_4 = nn.Parameter(torch.tensor(np.random.rand(4)))
        self.grid_6 = nn.Parameter(torch.tensor(np.random.rand(6)))
        self.grid_8 = nn.Parameter(torch.tensor(np.random.rand(8)))

        self.initial_state = nn.Parameter(torch.tensor(np.random.rand(3)))# 最初的分布
        self.state_M = nn.Parameter(torch.tensor(np.random.rand(3, 3))) #状态迁移矩阵
        self.softmax = nn.Softmax(dim = 0)

    def _prob(self, ob_i, current_state):
        prob_4 = self.softmax(self.grid_4)[ob_i - 1] if ob_i - 1 < 4 else torch.zeros(1, dtype=torch.float64)
        prob_6 = self.softmax(self.grid_6)[ob_i - 1] if ob_i - 1 < 6 else torch.zeros(1, dtype=torch.float64)
        prob_8 = self.softmax(self.grid_8)[ob_i - 1] if ob_i - 1 < 8 else torch.zeros(1, dtype=torch.float64)
        # 0 代表 当前状态是4面骰子的概率
        # 1 代表 当前状态是6面骰子的概率
        # 2 代表 当前状态是8面骰子的概率
        return current_state[0] * prob_4 + current_state[1] * prob_6 + current_state[2] * prob_8

    def forward(self, ob_list):
        current_state = self.softmax(self.initial_state) #概率归一化
        pi_p = None
        for ob_i in ob_list:
            p = self._prob(ob_i, current_state)
            if pi_p is not None:
                pi_p = pi_p * p
            else:
                pi_p = p
            current_state = self.softmax(current_state @ self.state_M)
        return pi_p

if __name__ == '__main__':
    model = Griddle()
    ob_list = [8,4,2,5,6,1,3,4,5,6,4,4,3,7,5,1,6,3,2,4,4,6]

    epochs = 10000
    opt = Adam(model.parameters())
    for epoch_i in range(epochs):
        p = model.forward(ob_list)
        loss = -torch.log(p)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss.item())
        if epoch_i % 100 == 0:
            print('show model hidden parameters')
            print(model.softmax(model.grid_4).data.numpy())
            print(model.softmax(model.grid_6).data.numpy())
            print(model.softmax(model.grid_8).data.numpy())
            print(model.softmax(model.initial_state).data.numpy())
            print(model.softmax(model.state_M).data.numpy())