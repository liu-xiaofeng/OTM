import theano.tensor as T
import theano.tensor.nnet
import numpy as np
import scipy.stats as st
from constants import classn, fea_len

# The EMD^2 loss for ordered-classes
def emd_l2(predictions, targets):
    for s in range(1, classn):
        predictions = T.inc_subtensor(predictions[:, s], predictions[:, s-1]);
        targets = T.inc_subtensor(targets[:, s], targets[:, s-1]);
    return (predictions - targets)**2;

# The EMD^2 loss for orderless-classes
# use the get_cost_vectors function to get cost_vectors
def cost_matrix_loss(predictions, targets, cost_vectors):
    return T.sqr(predictions - targets) * cost_vectors;

# this function returns the cost_vectors for cost_matrix_loss
# use update_cost_matrix to get cost_matrix (cm)
def get_cost_vectors(y, cm):
    max_y = np.argmax(y, axis=1);
    cost_vectors = np.zeros(shape=(y.shape[0], classn), dtype=np.float32);
    for i in range(y.shape[0]):
        cost_vectors[i] = cm[max_y[i]];
    return cost_vectors;

def update_cost_matrix(feas, labs, mu, omega):
    cm = np.zeros((classn, classn), dtype=np.float32);

    feas = feas / np.sum(feas, axis=1, keepdims=True);
    centroids = np.zeros((classn, fea_len), dtype=np.float32);
    for clab in range(classn):
        centroids[clab, :] = np.mean(feas[np.squeeze(labs)==clab, :], axis=0);

    for ci in range(classn):
        for cj in range(classn):
            cm[ci, cj] = np.linalg.norm(centroids[ci, :] - centroids[cj, :]);

    for ci in range(len(cm)):
        cm[ci] = np.array([(st.percentileofscore(cm[ci], a, 'rank')/100)**omega for a in cm[ci]]);

    cm = (cm + cm.transpose())/2.0;
    cm = (cm - mu);
    for ci in range(classn):
        cm[ci, ci] = 1.0;

    #print cm;
    return cm;

