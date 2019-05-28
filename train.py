import tensorflow as tf
from  tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import sys
import numpy as np
import os as os
import loss_layers
import models
from  util_loaders import *


#tf.logging.set_verbosity(tf.logging.ERROR)

#Config

path_to_models="../models/"
model_name="example"
path_to_model_dir=path_to_models+model_name
path_to_data="../project_data/"

#clean_models_log_files(path_to_model_dir)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def train_input_fn():
    #It has to be done without arguments
    #https://stackoverflow.com/questions/49140164/tensorflow-error-unsupported-callable?rq=1
    return input_fn(filenames=[path_to_data+"train.tfrecords"])

def eval_input_fn():
    return input_fn(filenames=[path_to_data+"eval.tfrecords"])


def model_fn(features, labels, mode, params):    
    #TODO: training=(mode == tf.estimator.ModeKeys.TRAIN) for dropout at testinf
    
    num_classes=5

    input_ = tf.identity(features["image"], name="input_tensor")   
    
    x = tf.reshape(input_, [-1, 299, 299, 1])    

    
    logits, y_pred, y_pred_cls=models.example_model(x, num_classes)
    #https://github.com/keras-team/keras-applications
    tf.keras.applications
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
        
    else:
              
        one_hot_labels=tf.one_hot(labels, num_classes)
        loss, other_outputs=loss_layers.roc_auc_loss(one_hot_labels,logits)

        loss = tf.reduce_mean(loss)
       
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            ,"auc": tf.metrics.auc(labels, y_pred_cls)
            #,"precision_at_0": tf.metrics.precision_at_k(labels, y_pred_cls, k=1, class_id=np.int32(0)) 
                   }

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec


model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": 1e-4},
                               model_dir=path_to_model_dir)


#Train Loop
count = 0
while (count < 2):    
        
    model.train(input_fn=train_input_fn, steps=10)
    
    result = model.evaluate(input_fn=eval_input_fn)
    
    
    predictor = model.predict(input_fn=eval_input_fn)
    
    #https://stackoverflow.com/questions/45912684/in-tensorflow-how-can-i-read-my-predictions-from-a-generator?rq=1
    predictions_dict = next(predictor)
    print("predictions",predictions_dict)
    
    print(result)
    print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
    sys.stdout.flush()
    count = count + 1

