import numpy as np
import os

import torch
import torch.utils.data
from torch.autograd import Variable

from model_lenet import RegressionModel, RegressionTrain
from model_resnet import MnistResNet, RegressionTrainResNet

from epo_lp import EPO_LP
from time import time
import pickle


def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable


def train(dataset, base_model, niter, preference):

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
    # model.randomize()
    if torch.cuda.is_available():
        model.cuda()
    # ---------***---------

    # DEFINE OPTIMIZERS
    # -----------------
    # Choose different optimizers for different base model
    if base_model == 'lenet':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.8)

    if base_model == 'resnet18':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20], gamma=0.1)

    # Instantia EPO Linear Program Solver
    _, n_params = getNumParams(model.parameters())
    print(f"# params={n_params}")
    epo_lp = EPO_LP(m=n_tasks, n=n_params, r=preference)
    # ---------***---------

    # CONTAINERS FOR KEEPING TRACK OF PROGRESS
    # ----------------------------------------
    task_train_losses = []
    train_accs = []
    # ---------***---------

    # TRAIN
    # -----
    for t in range(niter):
        # scheduler.step()

        n_manual_adjusts = 0
        descent = 0.
        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # Obtain losses and gradients
            grads = {}
            losses = []
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts)
                losses.append(task_loss[i].data.cpu().numpy())
                task_loss[i].backward()

                # One can use scalable method proposed in the MOO-MTL paper 
                # for large scale problem; but we use the gradient
                # of all parameters in this experiment.
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            G = torch.stack(grads_list)
            GG = G @ G.T
            losses = np.stack(losses)

            try:
                # Calculate the alphas from the LP solver
                alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
                if epo_lp.last_move == "dom":
                    descent += 1
            except Exception as e:
                print(e)
                alpha = None
            if alpha is None:   # A patch for the issue in cvxpy
                alpha = preference / preference.sum()
                n_manual_adjusts += 1

            if torch.cuda.is_available:
                alpha = n_tasks * torch.from_numpy(alpha).cuda()
            else:
                alpha = n_tasks * torch.from_numpy(alpha)
            # Optimization step
            optimizer.zero_grad()
            task_losses = model(X, ts)
            weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
            weighted_loss.backward()
            optimizer.step()

        print(f"\tdescent={descent/len(train_loader)}")
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


def run(dataset='mnist', base_model='lenet', niter=100, npref=5):
    """
    run Pareto MTL
    """
    start_time = time()
    init_weight = np.array([0.5, 0.5])
    preferences = circle_points(npref, min_angle=0.0001*np.pi/2, max_angle=0.9999*np.pi/2)  # preference
    results = dict()
    out_file_prefix = f"epo_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    for i, pref in enumerate(preferences[::-1]):
        s_t = time()
        res = train(dataset, base_model, niter, pref)
        results[i] = {"r": pref, "res": res}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        pickle.dump(results, open(results_file, "wb"))

    print(f"**** Time taken for {dataset} = {time() - start_time}")


run(dataset='mnist', base_model='lenet', niter=100, npref=5)
run(dataset='fashion', base_model='lenet', niter=100, npref=5)
run(dataset='fashion_and_mnist', base_model='lenet', niter=100, npref=5)

# run(dataset = 'mnist', base_model = 'resnet18', niter = 20, npref = 5)
# run(dataset = 'fashion', base_model = 'resnet18', niter = 20, npref = 5)
# run(dataset = 'fashion_and_mnist', base_model = 'resnet18', niter = 20, npref = 5)
