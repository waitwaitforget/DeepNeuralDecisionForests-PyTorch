import sys
import torch
import torch.nn as nn
from   torch.autograd import Variable
import numpy as np
from   torch.nn.parameter import Parameter
import torch.optim as optim

from   data_util import load_mnist

################ Definition #########################
DEPTH = 4  # Depth of a tree
N_LEAF = 2 ** (DEPTH + 1)  # Number of leaf node
N_LABEL = 10  # Number of classes
N_TREE = 1  # Number of trees (ensemble)
N_BATCH = 128  # Number of data points per mini-batch

cuda = 1

class DeepNeuralDecisionForest(nn.Module):
    def __init__(self, w4_e, w_d_e, p_keep_conv, p_keep_hidden):
        super(DeepNeuralDecisionForest, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self.conv.add_module('relu1', nn.ReLU())
        self.conv.add_module('pool1', nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('drop1', nn.Dropout(p_keep_conv))
        self.conv.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.conv.add_module('relu2', nn.ReLU())
        self.conv.add_module('pool2', nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('drop2', nn.Dropout(p_keep_conv))
        self.conv.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.conv.add_module('relu3', nn.ReLU())
        self.conv.add_module('pool3', nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('drop3', nn.Dropout(p_keep_conv))

        self.w4_e = []
        self.w_d_e = []
        for w4, w_d in zip(w4_e, w_d_e):
            self.w4_e.append(Parameter(w4))
            self.w_d_e.append(Parameter(w_d))
        self.p_keep_hidden  = p_keep_hidden
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p_keep_hidden)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv.forward(x)
        feat = feat.view(-1, 1152)
        decision_p_e = []

        for w4, w_d in zip(self.w4_e, self.w_d_e):
            l4 = self.relu(torch.mm(feat, w4))
            l4 = self.dropout(l4)

            decision_p = self.sigmoid(torch.mm(l4, w_d))
            decision_p_e.append(decision_p)
        return decision_p_e

def init_weights(shape):
    m = torch.randn(shape) * 0.01
    if cuda:
        m = m.cuda()
    return m

def init_prob_weights(shape, minval=-5, maxval=5):
    m = torch.Tensor(shape[0], shape[1]).uniform_(minval, maxval)
    if cuda:
        m = m.cuda()
    return m

def compute_mu(flat_decision_p_e, nsamples, nleaves):
    batch_0_indices = torch.range(0, nsamples * nleaves - 1, nleaves).unsqueeze(1).repeat(1, nleaves).long()

    in_repeat = nleaves / 2
    out_repeat = nsamples

    batch_complement_indices = torch.LongTensor(
        np.array([[0] * in_repeat, [nsamples * nleaves] * in_repeat] * out_repeat).reshape(nsamples, nleaves))

    # First define the routing probabilistics d for root nodes
    mu_e = []
    indices_var = Variable((batch_0_indices + batch_complement_indices).view(-1))
    if cuda:
        indices_var = indices_var.cuda()

    # iterate over each tree
    for i, flat_decision_p in enumerate(flat_decision_p_e):
        mu = torch.gather(flat_decision_p, 0, indices_var).view(nsamples, nleaves)
        mu_e.append(mu)

    # from the scond layer to the last layer, we make the decison nodes
    for d in xrange(1, DEPTH + 1):
        indices = torch.range(2 ** d, 2 ** (d + 1) - 1) - 1
        tile_indices = indices.unsqueeze(1).repeat(1, 2 ** (DEPTH - d + 1)).view(1, -1)
        batch_indices = batch_0_indices + tile_indices.repeat(nsamples, 1).long()

        in_repeat = in_repeat / 2
        out_repeat = out_repeat * 2

        # Again define the indices that picks d and 1-d for the nodes
        batch_complement_indices = torch.LongTensor(
            np.array([[0] * in_repeat, [nsamples * nleaves] * in_repeat] * out_repeat).reshape(nsamples, nleaves))

        mu_e_update = []
        indices_var = Variable((batch_indices + batch_complement_indices).view(-1))
        if cuda:
            indices_var = indices_var.cuda()

        for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
            mu = torch.mul(mu, torch.gather(flat_decision_p, 0, indices_var).view(
                nsamples, nleaves))
            mu_e_update.append(mu)

            mu_e = mu_e_update
    return mu_e

def compute_mu2(input):
    lastOffset = 0
    nextOffset = N_TREE
    lastTensor = input[:,0:N_TREE]
    for i in range(DEPTH+1):
        lastWidth = (1<<i) * N_TREE
        lastOffset, midOffset, nextOffset = nextOffset, nextOffset + lastWidth, nextOffset + lastWidth * 2
        leftTensor = input[:, lastOffset:midOffset]
        rightTensor= input[:, midOffset:nextOffset]

        leftProduct = lastTensor * leftTensor
        rightProduct = (1 - lastTensor) * rightTensor

        lastTensor = torch.cat((leftProduct, rightProduct), 1)
    return lastTensor  

# Define p(y|x)
def compute_py_x(mu_e, leaf_p_e):
    py_x_e = []
    nlabels= leaf_p_e[0].size(1)
    nsamples = mu_e[0].size(0)
    for mu, leaf_p in zip(mu_e, leaf_p_e):
        py_x_tree = mu.unsqueeze(2).repeat(1, 1, nlabels).mul(leaf_p.unsqueeze(0).repeat(nsamples, 1, 1)).mean(1)
        py_x_e.append(py_x_tree)

    py_x_e = torch.cat(py_x_e,1)
    py_x = py_x_e.mean(1).squeeze()
    return py_x

# Update pi( leaf_node probabilities)
def update_leaf_p(mu_e, pi_e, py_x, Y_val, pi_update):
    pi_update_e = []
    Y_val = Y_val.float()
    nlabels = Y_val.size(1)
    nleaves = pi_e[0].size(0)
    if cuda:
        pi_update = pi_update.cuda()

    for mu, pi in zip(mu_e, pi_e):
        mu = mu.unsqueeze(2).expand(mu.size(0),mu.size(1),nlabels)
        Y_val  = Y_val.unsqueeze(1).expand(Y_val.size(0), nleaves, Y_val.size(1))
        pi = pi.unsqueeze(0).expand(Y_val.size(0), pi.size(0), pi.size(1))
        # compute nominator and denominator
        common = pi * mu

        nominator = common * Y_val
        denominator = common.sum(1).expand_as(nominator)
        denominator = denominator + denominator.eq(0).float()

        pi_update = (nominator / denominator).sum(0).squeeze()
        
        # normalize pi
        sum_pi_over_y = pi_update.sum(1)
        all_0_y = sum_pi_over_y.eq(0).expand_as(pi_update).float()
        norm_pi_body = (pi_update + (1.0/ N_LABEL * all_0_y )) / (sum_pi_over_y.expand_as(pi_update) + all_0_y)
        pi_update_e.append(norm_pi_body.clone())
        pi_update.zero_()
    return pi_update_e

# training process
def train(model, loss, optimizer, X_val, Y_val, leaf_p_e):
    X_val = Variable(X_val)
    Y_val = Variable(Y_val)
    nsamples = X_val.size(0)
    nleaves = leaf_p_e[0].size(0)
    nlabels = leaf_p_e[0].size(1)
    optimizer.zero_grad()

    decision_p_e = model.forward(X_val)

    mu_e = []

    for decision_p in decision_p_e:
        mu = compute_mu2(decision_p)
        mu_e.append(mu)
    # compute mu
    #mu_e = compute_mu(flat_decision_p_e, nsamples, nleaves)
    #mu_e = [Variable(torch.randn(128,16), requires_grad=True)]
    # compute py_x
    
    py_x = compute_py_x(mu_e, leaf_p_e)
    # compute loss
    
    output = -loss.forward(py_x, Y_val)

    output.backward()

    optimizer.step()

    return output.data[0]


# testing process
def predict(model, X_val, leaf_p_e, mode = 1):
    X_val = Variable(X_val)
    decision_p_e = model.forward(X_val)
    flat_decision_p_e = []
    nsamples = X_val.size(0)
    nleaves = leaf_p_e[0].size(0)

    mu_e = []

    for decision_p in decision_p_e:
        mu = compute_mu2(decision_p)
        mu_e.append(mu)
    # compute py_x
    py_x = compute_py_x(mu_e, leaf_p_e)
    if mode == 1:
        return py_x.data.cpu().numpy().argmax(axis=1)
    elif mode == 2:
        return py_x, mu_e

def batch_forward(model, X_val, leaf_p_e):
    X_val = Variable(X_val)
    decision_p_e = model.forward(X_val)
    nsamples = X_val.size(0)
    nleaves = leaf_p_e[0].size(0)
    mu_e = []

    for decision_p in decision_p_e:
        mu = compute_mu2(decision_p)
        mu_e.append(mu)
    #mu_e = [Variable(torch.randn(128,16), requires_grad=True)]
    # compute py_x
    py_x = compute_py_x(mu_e, leaf_p_e)
    return mu_e, py_x

# update leaf_p
def full_forward(model, X_val, leaf_p_e):
    
    n_examples = len(X_val)
    num_batches = n_examples / N_BATCH
    total_mu_e = []
    #total_py_x_e = []
    for k in range(num_batches):
        start, end = k * N_BATCH, (k + 1) * N_BATCH
        mu_e, py_x = batch_forward(model, X_val[start:end], leaf_p_e)
        
        if k == 0:
            for mu in mu_e:
                total_mu_e.append(mu)
            total_py_x_e = py_x
        else:
            for i in xrange(len(total_mu_e)):
                total_mu_e[i] = torch.cat([total_mu_e[i], mu_e[i]])
            total_py_x_e = torch.cat([total_py_x_e, py_x])
        
    return total_mu_e, total_py_x_e

def main():
    # forest weights
    w4_ensemble = []
    w_d_ensemble = []
    leaf_p_e = []
    softmax = nn.Softmax()
    # parameter initialization
    print('# parameter initialization')
    for i in range(N_TREE):
        w4_ensemble.append(init_weights([128 * 3 * 3, 625]))
        w_d_ensemble.append(init_prob_weights([625, 100], -1, 1))
        pi = init_prob_weights([N_LEAF, N_LABEL], 0, 1)
        pi = softmax.forward(Variable(pi))
        if cuda:
            pi = pi.cuda()
        leaf_p_e.append(pi)

    # network hyperparameters
    p_conv_keep = 0.5
    p_full_keep = 0.4
    model = DeepNeuralDecisionForest(w4_e = w4_ensemble, w_d_e = w_d_ensemble, p_keep_conv = p_conv_keep, p_keep_hidden = p_full_keep)

    if cuda:
        model.cuda()
    ################ Load dataset #######################
    print('# data loading')
    trX, teX, trY_onehot, teY = load_mnist(onehot=True)
    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)
    trY = trY_onehot.argmax(axis=1)
    teY = teY.argmax(axis=1)

    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).long()
    trY_onehot = torch.from_numpy(trY_onehot).long()

    n_examples = len(trX)

    if cuda:
        trX = trX.cuda()
        teX = teX.cuda()
        trY = trY.cuda()
        trY_onehot = trY_onehot.cuda()

    optimizer = optim.RMSprop(model.parameters(), lr=1e-2, alpha=0.99, weight_decay=0)
    batch_size = N_BATCH
    print('# begin training')
    loss = nn.NLLLoss(size_average = True)
    pi_update = torch.zeros(leaf_p_e[0].size())
    
    for i in range(50):
        cost = 0.
        num_batches = n_examples / batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, trX[start:end], trY[start:end], leaf_p_e)
            
        # Define cost and optimization method
        predY = predict(model, teX, leaf_p_e, 1)
        print predY[:10]
        print("Epoch %d, cost = %f, acc = %.2f%%"
              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY)))
        mu_e, py_x = full_forward(model, trX, leaf_p_e)
        mu_data_e = []
        for mu in mu_e:
            mu_data_e.append(mu.data)
        leaf_data_e = []
        for leaf in leaf_p_e:
            leaf_data_e.append(leaf.data)
        print('Epoch %d, updating leaf probabilities!'%(i+1))
        
        updated_pi = update_leaf_p(mu_data_e, leaf_data_e, py_x.data, trY_onehot, pi_update)
        leaf_p_e = []
        for leaf in updated_pi:
            leaf_p_e.append(Variable(leaf))
        
        del mu_e
        del py_x
        

if __name__=='__main__':
    main()
