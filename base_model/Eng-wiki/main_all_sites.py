from lib import models_siamese, utils,graph
import numpy as np
import os
import time
from scipy import sparse
import random
#GPU 控制
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

print('start prepairing data')
x_x0,x_x1,x_y0,x_y1,c_x0,c_x1,c_y0,c_y1=utils.prepair_data()

# make pos and neg examples
train_index=utils.make_index(40000,40000,0)
np.save('./data/train_index.npy',train_index)
train_index=np.load('./data/train_index.npy')
test_index=utils.make_index(10000,10000,1)
np.save('./data/test_index.npy',test_index)
test_index=np.load('./data/test_index.npy')
train_index=utils.index_shuffle(train_index)
test_index=utils.index_shuffle(test_index)
x0_train=x_x0[train_index[0],:,:]
x1_train=x_x0[train_index[1],:,:]
y0_train=x_y0[train_index[0]]
y1_train=x_y1[train_index[1]]
y_train= np.ones([len(train_index[0])])
y_train[x_y0[train_index[0]]!=x_y1[train_index[1]]]=0
x0_test=c_x0[test_index[0],:,:]
x1_test=c_x0[test_index[1],:,:]
y0_test=c_y0[test_index[0]]
y1_test=c_y1[test_index[1]]
y_test= np.ones([len(test_index[0])])
y_test[c_y0[test_index[0]]!=c_y1[test_index[1]]]=0

print('start build graph')
# Calculate Laplacians
g0=sparse.csr_matrix(utils.build_tx_graph()).astype(np.float32)

graphs0 = []
for i in range(3):
    graphs0.append(g0)
L0 = [graph.laplacian(A, normalized=True) for A in graphs0]
L1 = x_x1.shape[1]

# Graph Conv-net
f0,f1,features,K=1,1,1,3
params = dict()
params['num_epochs']     = 50
params['batch_size']     = 256
params['eval_frequency'] = int(100)
# params['eval_frequency'] = int(x1_train.shape[0] / (params['batch_size'] * 4))

# Building blocks.
params['filter']         = 'chebyshev5'
params['brelu']          = 'b2relu'
params['pool']           = 'apool1'

# Architecture.
params['F0']              = [1,1]   # Number of graph convolutional filters.
params['F1']              = [1,1]   # Number of graph convolutional filters.
params['K']              = [K,K]   # Polynomial orders.
params['p']              = [1,1]    # Pooling sizes.
params['M']              = [1]    # Output dimensionality of fully connected layers.
params['input_features0'] = f0
params['input_features1'] = f1
params['lamda']          = 0.35
params['mu']             = 0.8
# Optimization.
params['regularization'] = 5e-3
params['dropout']        = 0.6
params['learning_rate']  = 1e-4
params['decay_rate']     = 0.95
params['momentum']       = 0
params['decay_steps']    = int(x1_train.shape[0] / params['batch_size'])

params['dir_name']       = 'siamese_' + time.strftime("%Y_%m_%d_%H_%M") + '_state'

# Save logs to folder
path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(path, 'logs', params['dir_name'])
os.makedirs(log_path)
# print(params)
print('start run model')
# Run model
model = models_siamese.siamese_cgcnn_cor(L0,L1, **params)

# utils.out_tsne(c_x0,x_x1,c_y0,x_y1,0,model)
# utils.out_tsne(c_x1,x_x0,c_y1,x_y0,1,model)

#utils.out_map(c_x0,x_x1,c_y0,x_y1,0,model)
#utils.out_map(c_x1,x_x0,c_y1,x_y0,1,model)
#utils.out_img(c_x0,x_x1,c_y0,x_y1,model)

# train code
accuracy, loss, t_step, scores_summary = model.fit(x0_train, x1_train,y_train, x0_test, x1_test,y_test)

