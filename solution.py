import zipfile
import numpy as np
import pandas as pd
import gpflow
import random
from gpflow.test_util import notebook_niter, is_continuous_integration
from gpflow.training import NatGradOptimizer, AdamOptimizer, XiSqrtMeanVar
from scipy import sparse
 
from sklearn.decomposition import TruncatedSVD

def read_train_zip_file(filepath):
    zfile = zipfile.ZipFile(filepath)
    read_x = {}
    read_y = {}
    for finfo in zfile.infolist():
        name = finfo.filename[12:-2]
        if name == '':
            continue
        x_or_y = finfo.filename[-1]
        ifile = zfile.open(finfo)
        rfile = zfile.read(finfo).decode('utf-8')
        line_list = ifile.readlines()
        if x_or_y == 'x':
            read_x[name] = []
            l = rfile.split("\n")
            l = [i.split(' ')for i in l]
            l = l[:-1]
            #print(l)
            nl = np.array(l)[:,:-1]
            
            cur = {}
            for i in range(nl.shape[0]):
                if int(nl[i][0])-1 in cur:
                    cur[int(nl[i][0])-1].append(int(nl[i][1]) - 1)
                else:
                    cur[int(nl[i][0])-1] = [int(nl[i][1]) - 1]
            for x in cur:
                read_x[name].append((x,cur[x])) 
        else:
            read_y[name] = []
            l = rfile.split("\n")
            l = [i.split(' ')for i in l]
            l = np.array(l[:-1])
            for i,x in enumerate(l):
                read_y[name].append((i,int(x)))
    return read_x, read_y


def read_test_zip_file(filepath):
    zfile = zipfile.ZipFile(filepath)
    read_x = {}
    for finfo in zfile.infolist():
        name = finfo.filename[20:-2]
        if name == '':
            continue
        x_or_y = finfo.filename[-1]
        ifile = zfile.open(finfo)
        rfile = zfile.read(finfo).decode('utf-8')
        line_list = ifile.readlines()
        read_x[name] = []
        l = rfile.split("\n")
        l = [i.split(' ')for i in l]
        l = l[:-1]
        #print(l)
        nl = np.array(l)[:,:-1]
            
        cur = {}
        for i in range(nl.shape[0]):
            if int(nl[i][0])-1 in cur:
                cur[int(nl[i][0])-1].append(int(nl[i][1]) - 1)
            else:
                cur[int(nl[i][0])-1] = [int(nl[i][1]) - 1]
        for x in cur:
            read_x[name].append((x,cur[x]))
                
    return read_x


class Logger(gpflow.actions.Action):
    def __init__(self, model):
        self.model = model
        self.logf = []

    def run(self, ctx):
        if (ctx.iteration % 10) == 0:
            likelihood = - ctx.session.run(self.model.likelihood_tensor)
            self.logf.append(likelihood)

if __name__ == '__main__':
    x, y = read_train_zip_file("./conll_train.zip")
    train_x = []
    train_y = []
    for k in x:
        for j in zip(x[k],y[k]):
            train_x.append(np.array(j[0][1]))
            train_y.append(np.array(j[1][1]))

    y_train = np.zeros([len(train_y),1])
    x_train = sparse.lil_matrix((y_train.shape[0],2035522))
    for i in range(x_train.shape[0]):
        x_train[i, train_x[i]] = 1
        y_train[i][0] = train_y[i]

    
    x_test = x_train[200000:,:]
    y_test = y_train[200000:,:]
    
    my_svd = TruncatedSVD(n_components=50, algorithm='arpack')
    my_svd.fit(x_train[:200000,:])
    mat_trans_x = my_svd.transform(x_train[final_index])

    Xtransformed = mat_trans_x
    Ytransformed = y_train[:200000,:]
    m = gpflow.models.SVGP(
    Xtransformed, Ytransformed, kern=gpflow.kernels.RBF(input_dim = 50),
    likelihood=gpflow.likelihoods.MultiClass(23),
    minibatch_size= 1000,
    Z=Xtransformed[::100].copy(), num_latent=23, whiten=True, q_diag=True)

    adam = gpflow.train.AdamOptimizer().make_optimize_action(m)
    logger = Logger(m)
    actions = [adam, logger]
    loop = gpflow.actions.Loop(actions, stop=5000)()
    m.anchor(m.enquire_session())
    
    '''
    x_test = read_test_zip_file("./conll_test_features.zip")
    sort_x_test = sorted(x_test.items(), key = lambda x: int(x[0]))

    sentences_index = [-1]

    test_sentences = None
    for s, sentence in enumerate(sort_x_test):
        cur_x = sparse.lil_matrix((len(sentence[1]),2035522))
        for i in range(len(sentence[1])):
            cur_x[i, sentence[1][i][1]] = 1
        if test_sentences == None:
            test_sentences = cur_x
        else:
            test_sentences = sparse.vstack((test_sentences, cur_x))
        sentences_index.append(sentences_index[-1] + len(sort_x_test[s][1]))
    transformed_test_sentence  = my_svd.transform(test_sentences)
    final_output = np.log(m.predict_y(transformed_test_sentence)[0])
    output_string = []
    for i in range(len(final_output)):
        results = list(map(str, final_output[i].tolist()))
        output_string.append(",".join(results))
        output_string.append("\n")
        if i in set(sentences_index[1:]):
            output_string.append("\n")
    with open('predictions.txt', 'w+') as f:
        for w in output_string:
            f.write(w)
    '''
    re = 0
    predicted = m.predict_y(x_test)[0]
    for i in range(y_test.shape[0]):
        if comp_y[i] == np.argmax(r[i]):
            c+=1
    print(c/y_test.shape[0])