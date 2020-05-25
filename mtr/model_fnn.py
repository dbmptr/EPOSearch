# lenet base model for Pareto MTL
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import MSELoss


class RegressionTrain(torch.nn.Module):

    def __init__(self, model):
        super(RegressionTrain, self).__init__()

        self.model = model
        self.mse_loss = MSELoss()

    def forward(self, x, ts):
        n_tasks = self.model.n_tasks
        ys = self.model(x)
        task_loss = []
        for i in range(n_tasks):
            task_loss.append(self.mse_loss(ys[:, i], ts[:, i]))
        task_loss = torch.stack(task_loss)

        return task_loss

    def randomize(self):
        self.model.apply(weights_init)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.weight.data *= 0.1


class RegressionModel(torch.nn.Module):
    def __init__(self, n_feats, n_tasks):
        super(RegressionModel, self).__init__()
        self.n_tasks = n_tasks
        self.layers = nn.ModuleList()
        n_neurons = n_feats
        while n_neurons > n_tasks:
            self.layers.append(nn.Linear(n_neurons, int(n_neurons / 2)))
            n_neurons = int(n_neurons / 2)
        # if n_neurons != n_tasks:
        self.layers.append(nn.Linear(n_neurons, n_tasks))

        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), nn.Linear(50, 10))

    def forward(self, x, i=None):
        y = x
        for i in range(len(self.layers)):
            y_temp = self.layers[i](y)
            y = torch.tanh(y_temp) if i < len(self.layers) - 1 else y_temp

        return y
