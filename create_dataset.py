from random import shuffle
import glob
import sys
import cv2
import numpy as np
from numpy import genfromtxt
#import skimage.io as io
import tensorflow as tf
import csv


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def createDataRecord(out_filename, addrs, labels):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])

        label = labels[i]

        if img is None:
            continue

        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

def csv_to_list(csv_file):
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return your_list

if __name__=__main__:
    #Paths
    path_to_data='../project_data/'
    path_to_data_images_train=path_to_data+'Mamm_Images_Train'
    path_to_data_images_eval=path_to_data+'Mamm_Images_Eval'
    path_to_data_images_test=path_to_data+'Mamm_Images_Test'
    path_to_models='../models/'

    images_train_path=path_to_data_images_train+'/*.jpg'
    images_eval_path=path_to_data_images_eval+'/*.jpg'
    images_test_path=path_to_data_images_test+'/*.jpg'

    addrs_train = glob.glob(cimages_train_path)
    addrs_eval = glob.glob(cimages_eval_path)
    addrs_test = glob.glob(cimages_test_path)

    labels_train=csv_to_list(path_to_data+'train_complex_labels.csv')
    labels_eval =csv_to_list(path_to_data+'train_complex_labels.csv')
    #test_labels

    c = list(zip(addrs_train, labels_train))
    shuffle(c)
    addrs, labels = zip(*c)
    createDataRecord(path_to_data+'train.tfrecords', train_addrs, train_labels)

    c = list(zip(addrs_eval, labels_eval))
    shuffle(c)
    addrs, labels = zip(*c)
    createDataRecord(path_to_data+'eval.tfrecords', eval_addrs, eval_labels)
