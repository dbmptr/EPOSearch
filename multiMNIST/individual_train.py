import numpy as np
import os

import torch
import torch.utils.data
from torch.autograd import Variable

from model_lenet import RegressionModel, RegressionTrain
from model_resnet import MnistResNet, RegressionTrainResNet

from time import time
import pickle


def train(dataset, base_model, niter, j):
    preference = np.array([1. - j, j])
    n_tasks = 2
    print("Preference Vector = {}".format(preference))

    # LOAD DATASET
    # ------------
    # MultiMNIST: multi_mnist.pickle
    if dataset == 'mnist':
        with open('data/multi_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        with open('data/multi_fashion.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_and_mnist':
        with open('data/multi_fashion_and_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
    testLabel = torch.from_numpy(testLabel).long()

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
    if base_model == 'lenet':
        model = RegressionTrain(RegressionModel(n_tasks), preference)
    if base_model == 'resnet18':
        model = RegressionTrainResNet(MnistResNet(n_tasks), preference)

    if torch.cuda.is_available():
        model.cuda()
    # ---------***---------

    # DEFINE OPTIMIZERS
    # -----------------
    # Choose different optimizers for different base model
    if base_model == 'lenet':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.0)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.5)

    if base_model == 'resnet18':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20], gamma=0.1)

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

        scheduler.step()
        n_manual_adjusts = 0
        model.train()
        for (it, batch) in enumerate(train_loader):

            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # Update using only j th task
            optimizer.zero_grad()
            task_j_loss = model(X, ts, j)
            task_j_loss.backward()
            optimizer.step()

        if n_manual_adjusts > 0:
            print(f"\t # manual tweek={n_manual_adjusts}")
        # Calculate and record performance
        if t == 0 or (t + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                total_train_loss = []
                train_acc = []

                correct1_train = 0
                correct2_train = 0

                for (it, batch) in enumerate(test_loader):

                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()

                    valid_train_loss = model(X, ts)
                    total_train_loss.append(valid_train_loss)
                    output1 = model.model(X).max(2, keepdim=True)[1][:, 0]
                    output2 = model.model(X).max(2, keepdim=True)[1][:, 1]
                    correct1_train += output1.eq(ts[:, 0].view_as(output1)).sum().item()
                    correct2_train += output2.eq(ts[:, 1].view_as(output2)).sum().item()

                train_acc = np.stack([1.0 * correct1_train / len(train_loader.dataset),
                                      1.0 * correct2_train / len(train_loader.dataset)])

                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim=0)

            # record and print
            if torch.cuda.is_available():

                task_train_losses.append(average_train_loss.data.cpu().numpy())
                train_accs.append(train_acc)

                print('{}/{}: train_loss={}, train_acc={}'.format(
                    t + 1, niter, task_train_losses[-1], train_accs[-1]))

                # weights.append(weight_vec.cpu().numpy())
                # print('{}/{}: weights={}, train_loss={}, train_acc={}'.format(
                #     t + 1, niter, weights[-1], task_train_losses[-1], train_accs[-1]))
    # torch.save(model.model.state_dict(), './saved_model/%s_%s_niter_%d.pickle' %
    #            (dataset, base_model, niter, npref))
    torch.save(model.model.state_dict(),
               f'./saved_model/{dataset}_{base_model}_niter_{niter}.pickle')

    result = {"training_losses": task_train_losses,
              "training_accuracies": train_accs}

    return result


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def run(dataset='mnist', base_model='lenet', niter=100):
    """
    run Pareto MTL
    """
    start_time = time()
    results = dict()
    out_file_prefix = f"indiv_{dataset}_{base_model}_{niter}"
    for j in range(2):
        s_t = time()
        res = train(dataset, base_model, niter, j)
        results[j] = {"r": np.array([1 - j, j]), "res": res}
        print(f"**** Time taken for {dataset}_{j} = {time() - s_t}")

    results_file = os.path.join("results", out_file_prefix + ".pkl")
    pickle.dump(results, open(results_file, "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


run(dataset='mnist', base_model='lenet', niter=100)
run(dataset='fashion', base_model='lenet', niter=100)
run(dataset='fashion_and_mnist', base_model='lenet', niter=100)

# run(dataset = 'mnist', base_model = 'resnet18', niter = 20, npref = 5)
# run(dataset = 'fashion', base_model = 'resnet18', niter = 20, npref = 5)
# run(dataset = 'fashion_and_mnist', base_model = 'resnet18', niter = 20, npref = 5)
