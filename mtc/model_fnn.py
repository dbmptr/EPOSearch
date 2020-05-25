# lenet base model for Pareto MTL
import torch
import torch.nn as nn


class RegressionTrain(torch.nn.Module):

    def __init__(self, model):
        super(RegressionTrain, self).__init__()

        self.model = model
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, x, ts):
        # n_tasks = self.model.n_tasks
        ys = self.model(x)
        task_loss = self.loss(ys, ts).mean(dim=0)
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
            if i < len(self.layers) - 1:
                y = torch.tanh(y_temp)
            else:
                y = torch.sigmoid(y_temp)

        return y
