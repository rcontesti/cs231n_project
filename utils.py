from sklearn.metrics import roc_curve,auc, roc_auc_score
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.autograd import Variable



"""
def proxy_loss(probs, labels):
    """
    probs [Batch Size] x [Number of Labels]
    labels[Bathc Size]
    """
    positive_probs=probs.gather(1, labels.view(-1,1))
    #TODO: this is not right
    negative_probs=1- positive_probs

    #One vs. All
    mean_positive=torch.mean(positive_probs, axis=1)
    #Per Class Average:


    #mean_negative=
    return 1-mean_positive*mean_positive
    
"""

def roc_auc_loss_function(labels, logits, prob_weights):

    """
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    weights: Either `None` or a `Tensor` with shape broadcastable to `logits`.

    Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise loss.
    """


    #original_shape = labels.get_shape().as_list()
    original_shape=labels.shape

    #Check the shape and format of these variables
    print(logits.dtype(), logits.shape)
    print(labels.dtype(), labels.shape)
    
    logits_difference=logits.unsqueeze(0)-logits.unsqueeze(1)
    labels_difference=labels.unsqueeze(0)-labels.unsqueeze(1) 
    weight_product=prob_weights.unsqueeze(0)-prob_weights.unsqueeze(1)

    signed_logits_difference=labels_difference*logits_difference

    raw_loss=weighted_sigmoid_cross_entropy_with_logits\
                                    (labels=torch.ones_like(signed_logits_difference),\
                                     logits=signed_logits_difference\
                                    )
    
    weighted_loss=weight_product*raw_loss
    


    loss=(torch.abs(labels_difference)*weight_loss).mean(0)*0.5
    loss=loss.reshape(original_shape)

    return loss


class Roc_Auc_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Roc_Auc_Loss,self).__init__()
    
    def forward(self, labels, logits, weights):
        return roc_auc_loss_function(labels, logits, prob_weights)
    

def weighted_sigmoid_cross_entropy_with_logits(labels, logits, positive_weights=1.0, negative_weights=1.0):

    softplus_term=torch.max(-logits, 0)+torch.log(1.0+torch.exp(-torch.abs(logits)))

    weight_dependent_factor = (negative_weights + (positive_weights - negative_weights) * labels)
    return ( negative_weights * (logits - labels * logits) + weight_dependent_factor * softplus_term)


def evaluate_model(scores, y, classes=[0,1,2,3,4]):
    # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    """
    y :       array, shape = [n_samples] or [n_samples, n_classes]
    scores : array, shape = [n_samples] or [n_samples, n_classes]
    """

    #convert to numpy
    scores=scores.data.numpy()
    y=     y.scores.data.numpy()



    #ROC scores
    roc_auc_scores_classes=roc_auc_score(y, scores)
    #ROC aggregate 'micro' score
    roc_auc_score_micro=roc_auc_score(y_true, y_score, average="micro")

    return roc_auc_scores_classes, roc_auc_score_micro
