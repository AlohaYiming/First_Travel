# Copyright (c) 2017 Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
# Copyright (c) 2016 Michaël Defferrard
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


from lib import graph
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import scipy.sparse
import numpy as np
import os, time, collections, shutil
all_score = []
all_data1 = []
all_data2 = []
scores = []
trindex = np.load('trindex.npy')
clslist = []
for line in open('./data/trainset_txt_img_cat.list'):
    cls = line.split('\t')[2]
    clslist.append(int(cls))
# Siamese model

class base_model(object):

    def __init__(self):
        self.regularizers = []
        seed = 123
        np.random.seed(seed)
        tf.set_random_seed(seed)

    # High-level interface which runs the constructed computational graph.

    def predict(self,data1,data2,labels=None,label2=1,sess=None):
        loss = 0
        size = data1.shape[0]
        predictions = np.empty(size)
        sess = self._get_session(sess)

        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])

            batch_data1 = np.zeros((self.batch_size, data1.shape[1], data1.shape[2]))
            batch_data2 = np.zeros((self.batch_size, data2.shape[1], data2.shape[2]))

            tmp_data1 = data1[begin:end, :, :]
            tmp_data2 = data2[begin:end, :, :]
            if type(tmp_data1) is not np.ndarray:
                tmp_data1 = tmp_data1.toarray()  # convert sparse matrices
            if type(tmp_data2) is not np.ndarray:
                tmp_data2 = tmp_data2.toarray()  # convert sparse matrices
            batch_data1[:end - begin] = tmp_data1
            batch_data2[:end - begin] = tmp_data2

            feed_dict = {self.ph_data1: batch_data1, self.ph_data2:batch_data2,self.ph_dropout: 1,self.ph_label2:label2}

            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros(self.batch_size)
                batch_labels[:end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run([self.op_prediction,self.op_loss], feed_dict)
                # print('emb',embedding_loss)
                #print('batch_pred',batch_pred)
                if np.isnan(batch_loss)==False:
                    loss += batch_loss

            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)

            predictions[begin:end] = batch_pred[:end - begin]
            # print(predictions[begin:end])

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            print('none loss',loss)
            return predictions

    def search(self,x0,x1,type=0):
        sess = tf.Session(graph=self.graph)
        path = os.path.join(self._get_path('checkpoints'), 'model')
        #path = "/home1/yul/xym/Eng-wiki/checkpoints/siamese_2018_09_25_16_37_state/model/"
        
        #print(path)
        ckpt = tf.train.get_checkpoint_state('./model/')
        #print(ckpt)
        self.op_saver.restore(sess, ckpt.model_checkpoint_path)
        
        # learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)
        if type==0:
            feed_dict={self.ph_data1:x0,self.ph_data2:x1,self.ph_dropout:self.dropout}
        else: 
            feed_dict={self.ph_data1:x1,self.ph_data2:x0,self.ph_dropout:self.dropout}
        op_logits = sess.run([self.op_prediction],feed_dict)
        # np.save('./tsne/feat-'+str(id)+'.npy',np.array(gcn_features))
        # print('gcn',np.array(gcn_features).shape)
        #print('x0:',x0)
        #print('x1:',x1)
        scores.append(op_logits)
        return op_logits
    
    def evaluate(self, data1,data2 ,labels,label2, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N
            N: number of signals (samples)
        """
        t_process, t_wall = time.process_time(), time.time()
        scores, loss = self.predict(data1,data2 ,labels,label2, sess)
        fpr, tpr,thresholds = roc_curve(labels, scores)
        # print(scores)
        roc_auc = auc(fpr, tpr)
        string = 'samples: {:d}, AUC : {:.2f}, loss: {:.4e}'.format(len(labels), roc_auc, loss)

        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall)
        return string, roc_auc, loss, scores

    def fit(self, train_data1,train_data2, train_labels, val_data1,val_data2, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))

        # Training.
        accuracies = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data1.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data1,batch_data2, batch_labels = train_data1[idx, :, :],train_data2[idx,:,:],train_labels[idx]
            if type(batch_data1) is not np.ndarray:
                batch_data1 = batch_data1.toarray()  # convert sparse matrices
            if type(batch_data2) is not np.ndarray:
                batch_data2 = batch_data2.toarray()  # convert sparse matrices
            feed_dict = {self.ph_data1: batch_data1,self.ph_data2:batch_data2, self.ph_labels: batch_labels, self.ph_dropout: self.dropout}
            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))

                string, auc, loss, scores_summary = self.evaluate(train_data1,train_data2, train_labels, sess)
                print('  training {}'.format(string))

                string, auc, loss, scores_summary = self.evaluate(val_data1,val_data2, val_labels, sess)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))

                accuracies.append(auc)
                losses.append(loss)

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/auc', simple_value=auc)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(accuracies), np.mean(accuracies[-10:])))
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return accuracies, losses, t_step, scores_summary

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.

    def build_graph(self, M_0,M_1):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data1 = tf.placeholder(tf.float32, (self.batch_size, M_0, self.input_features0), 'data1')
                self.ph_data2 = tf.placeholder(tf.float32, (self.batch_size, M_1, self.input_features1), 'data2')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_y1 = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_y2 = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')

            # Model.
            op_logits = self.inference(self.ph_data1,self.ph_data2, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_y1,self.ph_y2,self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()


            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()

    def inference(self, data1,data2, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data1,data2, dropout)
        return logits

    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        """Return the predicted classes."""
        with tf.name_scope('prediction'):
            # prediction = tf.nn.softmax(logits)
            prediction = logits
            return prediction

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        print('logits:',logits)
        print('labels:',labels)
        print('regularization:',regularization)
        with tf.name_scope('loss'):
            with tf.name_scope('hinge_loss'):
                labels = tf.cast(labels, tf.float32)
                zeros = tf.zeros(labels.get_shape())
                output = tf.ones(labels.get_shape()) - tf.multiply(labels, logits)
                hinge_loss = tf.where(tf.greater(output, zeros), output, zeros)
                hinge_loss = tf.reduce_mean(hinge_loss)
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = hinge_loss + regularization
            print('logits:',logits)
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/hinge_loss', hinge_loss)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([hinge_loss, regularization, loss])
                tf.summary.scalar('loss/avg/hinge_loss', averages.average(hinge_loss))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average

    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
                tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape,regularization=True):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)

        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        #initial = tf.constant_initializer(0.1)
        initial = tf.constant_initializer(0.0)

        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
            tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.

    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, L0,L1, F0,F1, K, p, M, filter='chebyshev5', brelu='b1relu', pool='mpool1',
                num_epochs=20, learning_rate=0.01, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=50, eval_frequency=200,
                dir_name=''):
        super().__init__()

        # Verify the consistency w.r.t. the number of layers.
        #assert len(L) >= len(F) == len(K) == len(p)
        assert len(F0) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L0) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.

        # Keep the useful Laplacians only. May be zero.
        M_0 = L0[0].shape[0]
        j = 0
        self.L0 = []
        for pp in p:
            self.L0.append(L0[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L0 = self.L0

        M_1 = L1

        # # Print information about NN architecture.
        # Ngconv = len(p)
        # Nfc = len(M)
        # print('NN architecture')
        # print('  input: M_0 = {}'.format(M_0))
        # for i in range(Ngconv):
        #     print('  layer {0}: cgconv{0}'.format(i+1))
        #     print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
        #             i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
        #     F_last = F[i-1] if i > 0 else self.input_features
        #     print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
        #             i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
        #     if brelu == 'b1relu':
        #         print('    biases: F_{} = {}'.format(i+1, F[i]))
        #     elif brelu == 'b2relu':
        #         print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
        #                 i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))
        # for i in range(Nfc):
        #     name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
        #     print('  layer {}: {}'.format(Ngconv+i+1, name))
        #     print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
        #     M_last = M[i-1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
        #     print('    weights: M_{} * M_{} = {} * {} = {}'.format(
        #             Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
        #     print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))

        # Store attributes and bind operations.
        self.L0,self.L1,self.F0, self.F1,self.K, self.p, self.M = L0,L1, F0,F1 ,K, p, M
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)

        # Build the computational graph.
        self.build_graph(M_0,M_1)
    def chebyshev2(self, x, L, Fout, K):
        """
        Filtering with Chebyshev interpolation
        Implementation: numpy.

        Data: x of size N x M x F
            N: number of signals
            M: number of vertices
            F: number of features per signal per vertex
        """
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        # Transform to Chebyshev basis
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        def chebyshev(x):
            return graph.chebyshev(L, x, K)
        x = tf.py_func(chebyshev, [x], [tf.float32])[0]  # K x M x Fin*N
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature.
        W = self._weight_variable([Fin*K,Fout], 'weight',regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def chebyshev5(self, x, L, Fout, K):
        print('using chebyshev5-1')
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat(0, [x, x_])  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K,Fout],'weight', regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x,relu=True):
        """Bias and ReLU (if relu=True). One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        x = x + b
        return tf.nn.relu(x) if relu else x

    def b2relu(self, x,relu=True):
        """Bias and ReLU (if relu=True). One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)],regularization=False)
        x = x + b
        return tf.nn.relu(x) if relu else x

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout,relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout],regularization=True)
        b = self._bias_variable([Mout],regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, dropout):
        # Graph convolutional layers.
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])

        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M,'add')
                x = tf.nn.dropout(x, dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], 'add',relu=False)
        return x


class siamese_cgcnn_cor(cgcnn):
    """
    Siamese Graph CNN which uses the Chebyshev approximation.
    Following the graph convolutional layers, the inner product of each node's features
    from the pair of graphs is used as input of the fully connected layer.
    Ktena et al., MICCAI 2017

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """

    def  __init__(self, L0,L1, F0,F1, K, p, M, input_features0,input_features1, lamda, mu,filter='chebyshev5', brelu='b1relu', pool='mpool1',
                num_epochs=20, learning_rate=0.01, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=50, eval_frequency=200,
                dir_name=''):

        self.input_features0 = input_features0
        self.input_features1 = input_features1
        self.lamda = lamda
        self.mu = mu
        super().__init__(L0,L1, F0,F1, K, p, M, filter, brelu, pool, num_epochs, learning_rate, decay_rate,
                         decay_steps, momentum, regularization, dropout, batch_size, eval_frequency, dir_name)

    def search2(x0,x1):
        sess2 = tf.Session(graph=self.graph)
        path = '/home1/yul/xym/ap/Eng-wiki/lib/../checkpoints/siamese_2018_10_04_16_06_state/model'
        ckpt = tf.train.get_checkpoint_state('./model/')
        self.op_saver.restore(sess2, ckpt.model_checkpoint_path)
        feed_dict={self.ph_data1:x0,self.ph_data2:x1,self.ph_dropout:self.dropout}
        op_logits=sess2.run([self.op_prediction],feed_dict)
        return op_logits

    def loss(self, logits, labels,regularization,label2):
        """Adds to the inference model the layers required to generate loss."""
        print('loss fanction start')
        print('logitssize',logits)
        print('labelssize',labels)
        print('regularizationsize',regularization)
        with tf.name_scope('loss'):
            with tf.name_scope('var_loss'):
                labels = tf.cast(labels, tf.float32)
                label2 = tf.cast(1-label2,tf.float32)
                shape = labels.get_shape()

                same_class = tf.boolean_mask(logits, tf.equal(labels, tf.ones(shape)))
                diff_class = tf.boolean_mask(logits, tf.not_equal(labels, tf.ones(shape)))
                same_mean, same_var = tf.nn.moments(same_class, [0])
                diff_mean, diff_var = tf.nn.moments(diff_class, [0])
                var_loss = same_var + diff_var
                print('-----------------------------------------',labels,shape)
            with tf.name_scope('mean_loss'):
                mean_loss = self.lamda * tf.where(tf.greater(self.mu - (same_mean - diff_mean), 0),
                                                  self.mu - (same_mean - diff_mean), 0)

            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)

            #mn = tf.reduce_mean(label2)
            #print('-----------------mn:',label2)
            loss =label2*(1) * var_loss + (1)*mean_loss + regularization
            
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([var_loss, mean_loss, regularization, loss])
                tf.summary.scalar('loss/avg/var_loss', averages.average(var_loss))
                tf.summary.scalar('loss/avg/mean_loss', averages.average(mean_loss))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def getdegree(idx,degree):
        #print('idx:',len(idx),idx)
        #print('degree:',len(degree),degree)
        degree2 = degree
        for i in range(256):
            a = trindex[0][idx[i]]
            b = trindex[1][idx[i]]
            #prinit("a,b:",clslist[a],clslist[b],degree[0][i])
            if clslist[a] == clslist[b]:
                if degree[0][i] < -0.54:
                    degree2[0][i] = 0.1
                elif degree[0][i] > -0.54 and degree[0][i] < -0.42:
                    degree2[0][i] = 0.2
                elif degree[0][i] > -0.42 and degree[0][i]< -0.36:
                    degree2[0][i] = 0.3
                elif degree[0][i] > -0.36 and degree[0][i]< -0.30:
                    degree2[0][i] = 0.4
                elif degree[0][i] > -0.30 and degree[0][i] < -0.25:
                    degree2[0][i] = 0.5
                elif degree[0][i] > -0.25 and degree[0][i]< -0.20:
                    degree2[0][i] = 0.6
                elif degree[0][i] > -0.20 and degree[0][i]< -0.16:
                    degree2[0][i] = 0.7
                elif degree[0][i] > -0.16 and degree[0][i] < 0:
                    degree2[0][i] = 0.8
                elif degree[0][i] > 0:
                    degree2[0][i] = 0.9
            else: degree2[0][i] = 0
            #print('degree2:',degree2)
            return degree2



    def fit(self, train_data1,train_data2,train_labels, val_data1, val_data2,val_labels):
        print('sm-fit')
        t_process, t_wall = time.process_time(), time.time()
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)
        # ckpt=tf.train.get_checkpoint_state('./checkpoints/')
        # self.op_saver.restore(sess,ckpt.model_checkpoint_path)
        
        # Training.
        aucs = []
        losses = []
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data1.shape[0] / self.batch_size)
       
        for step in range(1, num_steps + 1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data1.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            batch_data1, batch_data2,batch_labels = train_data1[idx, :, :], train_data2[idx,:,:],train_labels[idx]
            if step == num_steps:
                for i in idx:
                    all_data1.append(i)
                    all_data2.append(i)
            if type(batch_data1) is not np.ndarray:
                batch_data1 = batch_data1.toarray()  # convert sparse matrices
            if type(batch_data2) is not np.ndarray:
                batch_data2 = batch_data2.toarray()  # convert sparse matrices
            
            label22 =np.array(base_model.search(self,batch_data1,batch_data2,0))
            degree = siamese_cgcnn_cor.getdegree(idx,label22)
            #print('degree:',degree)
            nsum = 0
            for i in range(256):
                nsum += degree[0][i]
            label2 = nsum/256.0
            print('label2',label2)
            
            feed_dict = {self.ph_data1: batch_data1, self.ph_data2:batch_data2,self.ph_labels: batch_labels,
                         self.ph_dropout: self.dropout,self.ph_label2:label2}
            #
            #print(label2)
            prediction,learning_rate,loss_average = sess.run([self.op_prediction,self.op_train,self.op_loss_average], feed_dict)
            if step == num_steps:
                for i in prediction:
                    all_score.append(i)
            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data1.shape[0]
                # np.save('./Tsne/feat'+str(step)+'-'+str(step+self.batch_size)+'.npy')
                label2 = 1
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, auc, loss, scores_summary = self.evaluate(train_data1,train_data2,train_labels,label2, sess)
                print('  training {}'.format(string))

                string, auc, loss, scores_summary = self.evaluate(val_data1,val_data2,val_labels,label2, sess)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time() - t_process, time.time() - t_wall))

                aucs.append(auc)
                losses.append(loss)

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/auc', simple_value=auc)
                summary.value.add(tag='validation/loss', simple_value=loss)
                writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation accuracy: peak = {:.2f}, mean = {:.2f}'.format(max(aucs), np.mean(aucs[-10:])))
        np.save('data1.npy',all_data1)
        #np.save('data2.npy',all_data2) 
        np.save('score.npy',all_score)
        writer.close()
        sess.close()

        t_step = (time.time() - t_wall) / num_steps
        return aucs, losses, t_step, scores_summary

    def build_graph(self, M_0,M_1):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data1 = tf.placeholder(tf.float32, (self.batch_size, M_0,self.input_features0), 'data1')
                self.ph_data2 = tf.placeholder(tf.float32, (self.batch_size, M_0, self.input_features0), 'data2')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size), 'labels')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.ph_label2 = tf.placeholder(tf.float32,(),'label2')

            # Model.
            op_logits  = self.inference(self.ph_data1,self.ph_data2,self.ph_dropout)

            self.op_loss, self.op_loss_average = self.loss(op_logits,self.ph_labels,self.regularization,self.ph_label2)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                                          self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)

        self.graph.finalize()

    def chebyshev5(self, x, L, Fout, K, regularization=False):
        print('using chebyshev5-2')
        print(x.get_shape())
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_],0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=regularization)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def corr_layer(self, x1, x2):
        N1, M1, F1 = x1.get_shape()
        N2, M2, F2 = x1.get_shape()
        x1 = tf.reshape(x1, [int(N1 * M1), int(F1)])
        x2 = tf.reshape(x2, [int(N2 * M2), int(F2)])
        # multiply ->  yuan su xiang cheng
        # reduce_sum -> an weidu ya bian
        corr = tf.reduce_sum(tf.multiply(x1, x2), 1, keep_dims=True)
        # print(type(corr))
        res = tf.reshape(corr, [int(N1), int(M1), 1])
        return res

    # define pool
    def toppool(self, x, n):
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N * M), int(F)])
        x2 = tf.reduce_sum(x, 1, keep_dims=True)
        x2 = tf.squeeze(tf.reshape(x2, [int(N), int(M), 1]))
        newarray = [tf.nn.top_k(x2[i,:],n).indices for i in range(0,N)]
        # newarray = tf.reshape(newarray,[int(N)*n])
        # x = tf.zeros([int(N*n),F],dtype='float32')
        x = tf.reshape(tf.gather(x,newarray),[int(N),n,int(F)])
        # print(x.get_shape())
        return x

    def build_model(self, g):
        # Graph convolutional layers.
        gcn_features = []
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i + 1)):
                with tf.name_scope('filter'):
                    g = self.filter(g, self.L0[i],self.F0[i], self.K[i])
                    gcn_features.append(g)
                # if i == 0:
                #     with tf.name_scope('attention'):
                #         g = self.attention(g)
                with tf.name_scope('bias_relu'):
                    g = self.brelu(g)
                with tf.name_scope('pooling'):
                    g = self.pool(g, self.p[i])
                    # g = self.toppool(g, 50)
        # g->shape(200,110,64)
        return g,gcn_features
    def cnn_img(self,x):
        #第一个卷积层（224——>112)
        conv1=tf.layers.conv2d(
              inputs=x,
              filters=32,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        #第二个卷积层(112->56)
        conv2=tf.layers.conv2d(
              inputs=pool1,
              filters=64,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        #第三个卷积层(56->28)
        conv3=tf.layers.conv2d(
              inputs=pool2,
              filters=128,
              kernel_size=[3, 3],
              padding="same",
              activation=tf.nn.relu,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        #第四个卷积层(28->14)
        conv4=tf.layers.conv2d(
              inputs=pool3,
              filters=128,
              kernel_size=[3, 3],
              padding="same",
              activation=tf.nn.relu,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        #第五个卷积层(14->7)
        conv5=tf.layers.conv2d(
              inputs=pool4,
              filters=128,
              kernel_size=[3, 3],
              padding="same",
              activation=tf.nn.relu,
              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool5=tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
        re1 = tf.reshape(pool5, [-1, 7 * 7 * 128])
        return re1

    def attention(self,g):
        v,q,k = g,g,g
        kt = tf.transpose(k,perm=[0,2,1])
        print('kt:',kt.get_shape())
        shape = tf.cast(v.get_shape()[2],tf.float32)
        att_k = tf.nn.softmax(tf.div(tf.matmul(q,kt),tf.sqrt(shape)))
        print('att_k:',att_k.get_shape())
        out_g = tf.matmul(att_k,v)
        print('out_g',out_g.get_shape())
        return out_g

    def lstm_txt(self,g):
        lstmCell = tf.contrib.rnn.BasicLSTMCell(64)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        value, what = tf.nn.dynamic_rnn(lstmCell, g, dtype=tf.float32)
        print('Lstm-out-shape',value.get_shape())
        return value,what


    def cnn_txt(self,g):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters = 128
        word_length = int(g.get_shape()[1])
        embedding_size = int(g.get_shape()[2])
        filter_sizes = [3,4,5]
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(tf.expand_dims(g,3),W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,ksize=[1, word_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print('pool',pooled.get_shape())
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        output = tf.reshape(h_pool, [-1, num_filters_total])
        output = tf.expand_dims(output,2)
        print('cnn-out',output.get_shape())
        return output,output

        
    def _inference(self, x_0, x_1,dropout):

        depth = 10 # 10分类
        # Share weights between the two models of the pair
        with tf.variable_scope("siamese",reuse=tf.AUTO_REUSE) as scope:
            m_0,gcn_features0 = (self.build_model(x_0))

            m_1,gcn_features1 = (self.build_model(x_1))

        N1,M1,F1 = m_0.get_shape()
        N2,M2,F2 = m_1.get_shape()
        model_0 = tf.reshape(m_0,[int(N1),int(M1*F1)])
        model_1 = tf.reshape(m_1,[int(N2),int(M2*F2)])
        fc_layers = 1024
        with tf.variable_scope('txt_fc'):
            model_0 = self.fc(model_0,fc_layers,relu=False)
        with tf.variable_scope('img_fc0'):
            model_1 = self.fc(model_1,fc_layers,relu=False)
        img_out = model_1

        with tf.variable_scope('txt_classify'):
            pred_0 = tf.nn.softmax(self.fc(model_0,depth,relu=False))
            py_0 = tf.one_hot(tf.argmax(pred_0,1),depth)
        with tf.variable_scope('img_classify'):
            pred_1 = tf.nn.softmax(self.fc(model_1,depth,relu=False))
            py_1 = tf.one_hot(tf.argmax(pred_1,1),depth)
        # dot
        x = tf.multiply(model_0,model_1)
        
        # Logits linear layer
        with tf.variable_scope('logits'):
            x = tf.nn.dropout(x, dropout)
            x = self.fc(x, self.M[-1],relu=False)

        return tf.squeeze(x) # tf.sigmoid(x)

    def inference(self, data1,data2,dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        logits = self._inference(data1, data2,dropout)
        # logits=tf.nn.softmax(logit)
        return logits 
