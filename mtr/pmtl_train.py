import numpy as np
import os

import torch
import torch.utils.data
from torch.autograd import Variable

from model_fnn import RegressionModel, RegressionTrain

from min_norm_solvers import MinNormSolver
from time import time
import pickle


def get_d_paretomtl_init(grads,value,weights,i):
    """ 
    calculate the gradient direction for ParetoMTL initialization 
    """
    
    flag = False
    nobj = value.shape
   
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
   
    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        vec =  torch.matmul(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    # weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
    # weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
    # weight = torch.stack([weight0,weight1])

    new_weights = []
    for t in range(len(value)):
        new_weights.append(torch.sum(torch.stack([sol[j] * w[idx][j ,t] for j in torch.arange(0, torch.sum(idx))])))
    
    return flag, torch.stack(new_weights)


def get_d_paretomtl(grads,value,weights,i):
    """ calculate the gradient direction for ParetoMTL """
    
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
    

    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()


    vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    # weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    # weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    # weight = torch.stack([weight0,weight1])

    new_weights = []
    for t in range(len(value)):
        new_weights.append(sol[t] + torch.sum(torch.stack([sol[j] * w[idx][j ,t] for j in torch.arange(0, torch.sum(idx))])))
    
    return torch.stack(new_weights)


def circle_points_(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles
def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]



def train(dataset, base_model, niter, npref, rvecs, pref_idx):

    # generate #npref preference vectors
    ref_vec = torch.tensor(rvecs).cuda().float()
    
    # load dataset 
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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.8)
    
    
    # store infomation during optimization
    weights = []
    task_train_losses = []
    train_accs = []
    
    
    # print the current preference vector
    print('Preference Vector ({}/{}):'.format(pref_idx + 1, npref))
    print(ref_vec[pref_idx].cpu().numpy())

    # run at most 2 epochs to find the initial solution
    # stop early once a feasible solution is found 
    # usually can be found with a few steps
    for t in range(2):
      
        model.train()
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            grads = {}
            losses_vec = []
            
            
            # obtain and store the gradient value
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
                
                grads[i] = []
                
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
            
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            flag, weight_vec = get_d_paretomtl_init(grads,losses_vec,ref_vec,pref_idx)
            
            # early stop once a feasible solution is obtained
            if flag == True:
                print("fealsible solution is obtained.")
                break
            # print(f'len(weight_vec)={len(weight_vec)}')
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()
                
        else:
        # continue if no feasible solution is found
            continue
        # break the loop once a feasible solutions is found
        break
                
        

    # run niter epochs of ParetoMTL 
    for t in range(niter):
        
        # scheduler.step()
      
        model.train()
        for (it, batch) in enumerate(train_loader):
            
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # obtain and store the gradient 
            grads = {}
            losses_vec = []
            
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(X, ts) 
                losses_vec.append(task_loss[i].data)
                
                task_loss[i].backward()
            
                # can use scalable method proposed in the MOO-MTL paper for large scale problem
                # but we keep use the gradient of all parameters in this experiment              
                grads[i] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

                
                
            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            weight_vec = get_d_paretomtl(grads,losses_vec,ref_vec,pref_idx)
            
            # normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            normalize_coeff = 1. / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            
            # optimization step
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = model(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            
            loss_total.backward()
            optimizer.step()


        # calculate and record performance
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
    

def run(dataset = 'rf1',base_model = 'fnn', niter = 100, npref = 5):
    """
    run Pareto MTL
    """
    n_tasks = 8
    start_time = time()
    preferences = np.abs(np.random.randn(npref, n_tasks))
    preferences /= preferences.sum(axis=1, keepdims=True)
    results = dict()
    out_file_prefix = f"pmtl_{dataset}_{base_model}_{niter}_{npref}_from_0-"
    for i, pref in enumerate(preferences):
        s_t = time()
        pref_idx = i 
        res = train(dataset, base_model, niter, npref, preferences, pref_idx)
        results[i] = {"r": pref, "res": res}
        print(f"**** Time taken for {dataset}_{i} = {time() - s_t}")
        results_file = os.path.join("results", out_file_prefix + f"{i}.pkl")
        pickle.dump(results, open(results_file, "wb"))
    pickle.dump(results, open(f"pmtl_{dataset}_{base_model}_{niter}_{npref}.pkl", "wb"))
    print(f"**** Time taken for {dataset} = {time() - start_time}")


run(dataset='rf1', niter=100, npref=10)

