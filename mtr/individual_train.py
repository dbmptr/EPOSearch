import numpy as np
import os

import torch
import torch.utils.data
from torch.autograd import Variable

from model_fnn import RegressionModel, RegressionTrain

from time import time
import pickle


def train(dataset, base_model, niter, j):
    print("Preference only for {}".format(j))

    # LOAD DATASET
    # ------------
    if dataset == 'rf1':
        with open('data/rf1.pkl', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX).float()
    trainLabel = torch.from_numpy(trainLabel).float()
    testX = torch.from_numpy(testX).float()
    testLabel = torch.from_numpy(testLabel).float()
    n_tasks = testLabel.shape[1]
    n_feats = testX.shape[1]

    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set = torch.utils.data.TensorDataset(testX, testLabel)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))
    # ---------***---------

    # DEFINE MODEL
    # ---------------------
    model = RegressionTrain(RegressionModel(n_feats, n_tasks))
    # model.randomize()
    if torch.cuda.is_available():
        model.cuda()
    # ---------***---------

    # DEFINE OPTIMIZERS
    # -----------------
    # Choose different optimizers for different base model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.8)
    # ---------***---------


    # CONTAINERS FOR KEEPING TRACK OF PROGRESS
    # ----------------------------------------
    weights = []
    task_train_losses = []
    train_accs = []
    # ---------***---------

    # TRAIN
    # -----
    for t in range(niter):

        # scheduler.step()
        model.train()
        for (it, batch) in enumerate(train_loader):

            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # Update using only j th task
            optimizer.zero_grad()
            task_loss = model(X, ts)
            task_loss[j].backward()
            optimizer.step()

        # Calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                total_train_loss = []

                for (it, batch) in enumerate(test_loader):

                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()

                    valid_train_loss = model(X, ts)
                    total_train_loss.append(valid_train_loss)

                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim=0)

            # record and print
            if torch.cuda.is_available():

                task_train_losses.append(average_train_loss.data.cpu().numpy())

                print('{}/{}: train_loss={}'.format(
                    t + 1, niter, task_train_losses[-1]))

    # torch.save(model.model.state_dict(),
    #            f'./saved_model/{dataset}_{base_model}_niter_{niter}.pickle')

    result = {"training_losses": task_train_losses}

    return result


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def run(dataset='rf1', base_model='fnn', niter=100):
    """
    run Pareto MTL
    """
    n_tasks = 8
    start_time = time()
    results = dict()
    for j in range(n_tasks):
        s_t = time()
        res = train(dataset, base_model, niter, j)
        results[j] = {"r": j, "res": res}
        print(f"**** Time taken for {dataset}_{j} = {time() - s_t}")

    results_file = os.path.join("results",
                                f"indiv_{dataset}_{base_model}_{niter}.pkl")
    pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


run(dataset='rf1', niter=100)
