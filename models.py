import tensorflow as tf
from  tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import sys
import numpy as np
import os as os


#tf.logging.set_verbosity(tf.logging.ERROR)


def example_model(x, num_classes):
    
    #MODEL LAYERS inputs_>>logits
    """
    # tf.layers.dense (N,Input_shape)--> (N, Output_shape)  https://www.tensorflow.org/api_docs/python/tf/layers/dense
    """
    x = tf.identity(x, name="input_tensor_after")
    x = Conv2D(32, (3, 3),padding='same', name='layer_conv1', activation=tf.nn.relu)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Conv2D(32, (3, 3),padding='same', name='layer_conv1', activation=tf.nn.relu)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(name='layer_fc1',units=128, activation=tf.nn.relu)(x)  
    x = Dropout(rate=0.5, noise_shape=None, seed=None)(x)
    x = Dense(name='layer_fc2',units=num_classes, activation=tf.nn.relu)(x)

    logits=x


    # PREDICTIONS

    y_pred     = tf.nn.softmax(logits=logits)
    y_pred     = tf.identity(y_pred, name="output_pred")

    y_pred_cls = tf.argmax  (y_pred, axis=1)
    y_pred_cls = tf.identity(y_pred_cls, name="output_cls")
    
    return logits, y_pred, y_pred_cls