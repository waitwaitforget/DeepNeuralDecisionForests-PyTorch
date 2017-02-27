import gzip
import os
from os import path
import urllib
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms

DATASET_DIR = 'datasets/'


def download_file(url, local_path):
    dir_path = path.dirname(local_path)
    if not path.exists(dir_path):
        print("Creating the directory '%s' ..." % dir_path)
        os.makedirs(dir_path)

    print("Downloading from '%s' ..." % url)
    urllib.URLopener().retrieve(url, local_path)


def download_mnist(local_path):
    url_root = "http://yann.lecun.com/exdb/mnist/"
    for f_name in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        f_path = os.path.join(local_path, f_name)
        if not path.exists(f_path):
            download_file(url_root + f_name, f_path)


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def load_mnist(ntrain=60000, ntest=10000, onehot=True):
    data_dir = os.path.join(DATASET_DIR, 'mnist/')
    if not path.exists(data_dir):
        download_mnist(data_dir)

    with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trY = loaded[8:].reshape((60000))

    with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teY = loaded[8:].reshape((10000))

    trX /= 255.
    teX /= 255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY

def load_cifar100(root):
    train = dset.CIFAR100(root,train=True,transform=transforms.ToTensor())
    train_data = train.train_data.astype(np.float32)/255
    train_labels = train.train_labels
    test = dset.CIFAR100(root=root,train=False,transform=transforms.ToTensor())
    test_data  = test.test_data.astype(np.float32)/255
    test_labels = test.test_labels
    return train_data, train_labels, test_data, test_labels

def split_datasets(train_data, train_labels, test_data, test_labels, num_newclass = 10):
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    n_class = train_labels.max()+1
    if n_class <= num_newclass:
        print "original datasets class number < num_newclass"
        return  None
    if train_labels.ndim == 1:
        train_labels = one_hot(train_labels,n_class)
        test_labels  = one_hot(test_labels, n_class)

    for i in range(num_newclass):
        idx = np.random.randint(0,n_class-i,1)
        index1 = np.where(train_labels[:,idx]==1)
        index2 = np.where(test_labels[:,idx] == 1)
        if i==0:
            new_traindata = train_data[index1[0]]
            new_trainlabels = np.ones((len(index1[0]),1))*(i+1)
            new_testdata = test_data[index2[0]]
            new_testlabels = np.ones((len(index2[0]),1))*(i+1)
        else:

            new_traindata = np.vstack((new_traindata,train_data[index1[0]]))
            new_trainlabels = np.vstack((new_trainlabels, np.ones((len(index1[0]),1))*(i+1)))
            new_testdata  = np.vstack((new_testdata,test_data[index2[0]]))
            new_testlabels = np.vstack((new_testlabels, np.ones((len(index2[0]),1))*(i+1)))

        train_data  = np.delete(train_data,index1[0],axis=0)
        train_labels = np.delete(train_labels,index1[0], axis=0)
        train_labels = np.delete(train_labels, idx, axis=1)

        test_data = np.delete(test_data,index2[0], axis=0)
        test_labels = np.delete(test_labels, index2[0], axis=0)
        test_labels = np.delete(test_labels, idx, axis=1)

    old_task = {}
    old_task['train_data']      = train_data
    old_task['train_labels']    = train_labels
    old_task['test_data']       = test_data
    old_task['test_labels']     = test_labels
    new_task  ={}
    new_task['train_data']      = new_traindata
    new_task['train_labels']    = one_hot(new_trainlabels.astype('int32')-1,num_newclass)
    #new_task['train_labels']    = one_hot(new_trainlabels,num_newclass)
    new_task['test_data']       = new_testdata
    new_task['test_labels']     = one_hot(new_testlabels.astype('int32')-1,num_newclass)
    return old_task, new_task



