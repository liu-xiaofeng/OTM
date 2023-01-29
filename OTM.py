import lasagne
import theano
import sys
import numpy as np
import theano.tensor as T
from lasagne import layers
from lasagne import regularization
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from data_aug_online import data_aug
from get_network import get_alex_net_small
from emd_loss import emd_l2, cost_matrix_loss, get_cost_vectors, update_cost_matrix
from constants import APS, PS, fea_len, classn, WeightDecay, BatchSize, Epochs, MinProb, XenWei, CmWei
from batch_iter import iterate_minibatches
from data_loader import load_data
from eval import diag_error

cv_i = int(sys.argv[1]);
TrainImgeFolder = ['.../train_fold_{}.txt'.format(cv_i)];
TestImgeFolder = ['.../test_fold_{}.txt'.format(cv_i)];

# set the learning rate for the first several epochs
PreLearningRate = theano.shared(np.array(10**-2.5, dtype=theano.config.floatX));
# set the learning rate
LearningRate = theano.shared(np.array(2*10**-3, dtype=theano.config.floatX));
# number of epochs using the PreLearningRate as the learning rate
PreEpochs = 240;

def custom_loss_function(cnn):
    input_var = T.tensor4('inputs');
    target_var = T.matrix('targets');
    target_xen_var = T.ivector('targets_xen');
    cost_vectors_var = T.matrix('cost_vec');

    # Reset the input
    input_layer_index = map(lambda pair : pair[0], cnn.layers).index('input');
    first_layer = cnn.get_all_layers()[input_layer_index + 1];
    input_layer = layers.InputLayer(shape = (None, 3, PS, PS), input_var = input_var);
    first_layer.input_layer = input_layer;

    # Get the features
    feas_layer_index = map(lambda pair : pair[0], cnn.layers).index('drop_fc7');
    network = cnn.get_all_layers()[feas_layer_index];
    feas_deter = lasagne.layers.get_output(network, deterministic = True);

    # Get the output
    output_layer_index = map(lambda pair : pair[0], cnn.layers).index('output');
    network = cnn.get_all_layers()[output_layer_index];
    output = lasagne.layers.get_output(network);

    # Build the loss
    loss = lasagne.objectives.categorical_crossentropy(output+MinProb, target_var).mean() * XenWei + \
           cost_matrix_loss(output, target_var, cost_vectors_var).mean() * CmWei + \
           WeightDecay * regularization.regularize_network_params(layer=network, penalty=regularization.l2);

    loss_xen = lasagne.objectives.categorical_crossentropy(output, target_xen_var).mean() + \
           WeightDecay * regularization.regularize_network_params(layer=network, penalty=regularization.l2);

    # Build update function
    params = layers.get_all_params(network, trainable = True);
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = LearningRate, momentum = 0.98);
    updates_xen = lasagne.updates.nesterov_momentum(loss_xen, params, learning_rate = PreLearningRate, momentum = 0.98);

    # Get deterministic output
    output_deter = lasagne.layers.get_output(network, deterministic = True);

    # Build functions
    val_fn = theano.function([input_var], [feas_deter, output_deter]);
    train_fn = theano.function([input_var, target_var, cost_vectors_var], loss, updates = updates);
    train_xen_fn = theano.function([input_var, target_xen_var], loss_xen, updates = updates_xen);

    return network, train_fn, train_xen_fn, val_fn;

def val_fn_epoch(val_fn, Set):
    X = Set[0].copy();
    y = Set[1].copy();
    f = np.zeros((X.shape[0], fea_len), dtype=np.float32);
    p = np.zeros((X.shape[0], classn), dtype=np.float32);

    pn = 0;
    for batch in iterate_minibatches(X, y, BatchSize, shuffle = False):
        inptx, _ = batch;
        feas1, pred1 = val_fn(data_aug(inptx[:, :, :: 1, :], deterministic=True));
        feas2, pred2 = val_fn(data_aug(inptx[:, :, ::-1, :], deterministic=True));
        f[pn : pn+inptx.shape[0], :] = (feas1 + feas2) / 2.0;
        p[pn : pn+inptx.shape[0], :] = (pred1 + pred2) / 2.0;
        pn += inptx.shape[0];
    max_p = np.expand_dims(np.argmax(p, axis=1), axis=1);
    max_y = np.expand_dims(np.argmax(y, axis=1), axis=1);
    return max_p, max_y, p, y, f;

#############################################################
# main

sys.setrecursionlimit(10000);
np.set_printoptions(precision=3, edgeitems=5, linewidth=200, suppress=True);

# load dataset
Train = load_data(TrainImgeFolder);
Test = load_data(TestImgeFolder);

# load network, custom loss functions
cnn = get_alex_net_small();
network, train_fn, train_xen_fn, val_fn = custom_loss_function(cnn);

# train and validate
print("TrLoss\t\tTrAcc0\t\tTrAcc1\t\tTeAcc0\t\tTeAcc1\t\tEpochs");
for epoch in range(Epochs):
    trn_err = 0;
    trn_btn = 0;
    for batch in iterate_minibatches(Train[0], Train[1], BatchSize, shuffle = True):
        # load a mini-batch
        inputs, targets = batch;
        # data augmentation
        inputs = data_aug(inputs.copy());
        if epoch < PreEpochs:
            # train the first several epochs with the cross-entropy loss
            trn_err += train_xen_fn(inputs, np.argmax(targets, axis=1).astype(np.int32));
        else:
            trn_err += train_fn(inputs, targets, get_cost_vectors(targets, cost_matrix));
        trn_btn += 1;

    # print validation results
    p_trn, t_trn, prob_p_trn, prob_t_trn, f_trn = val_fn_epoch(val_fn, Train);
    p_tst, t_tst, prob_p_tst, prob_t_tst, f_tst = val_fn_epoch(val_fn, Test);
    trn_acc0, trn_acc1 = diag_error(p_trn, t_trn);
    tst_acc0, tst_acc1 = diag_error(p_tst, t_tst);
    print("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}/{}".format(
        trn_err/trn_btn, trn_acc0, trn_acc1, tst_acc0, tst_acc1, epoch, Epochs));

    # decrease the learning rate
    LearningRate.set_value(np.float32(0.995*LearningRate.get_value()));
    PreLearningRate.set_value(np.float32(0.995*PreLearningRate.get_value()));

    if epoch >= 3:
        cost_matrix = update_cost_matrix(f_trn, t_trn, 0.5, 1);

