import tensorflow as tf
import numpy as np
import scipy.misc
from PIL import Image
import requests
import datetime
import time
import os
import matplotlib.pyplot as plt
import math
from random import shuffle
import argparse
import sys
from skimage.measure import compare_ssim as ssim
import main2
parser = argparse.ArgumentParser(description='Process some integers.')



#vector or processed
MODEL = "qr"

def classic_model(features):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1 ,64,64,1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        padding="same",
        kernel_size=[11, 11],
        activation=tf.nn.relu)

    # Convolutional Layer #3
    output = tf.layers.conv2d(
        inputs=conv2,
        filters=1,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)
    return output 

def qr_model(features):
    """Model function for CNN."""
    # Input Layer
    features = tf.expand_dims(features,0)
    features = tf.expand_dims(features,3)
    input_layer = tf.reshape(features, [-1 ,63,63,1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=63,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        padding="same",
        kernel_size=[11, 11],
        activation=tf.nn.relu)

    # Convolutional Layer #3
    output = tf.layers.conv2d(
        inputs=conv2,
        filters=1,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)
    return output 

def vector_model(features):
    # Input Layer
    input_layer = tf.reshape(features, [-1 ,64,64,1])
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=16,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #3
    output = tf.layers.conv2d(
        inputs=conv3,
        filters=1,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)

    return output

def test_model(features):

    if MODEL == "vector":
        output=vector_model(features)
    elif MODEL == "classic":
        output=classic_model(features)
    elif MODEL == "qr":
        output=qr_model(features)
    return output

def main(img,per):        
        data_path = "C:\\Users\\eloy\\Desktop\\Code\\Python\\dp_one\\experiments\\experiment("+str(per)+")\\model-"+str(per)+"-2550.ckpt"
        nx,ny=img.shape
        r_input = tf.placeholder(tf.float32, None)
        out = test_model(r_input)
        saver = tf.train.Saver()
        matrix_vector = []
        masks_vector = []
        intensity_vector = []
        np.random.seed(1)
        for _ in range(0, int(per*64*64)):
            random_matrix = (np.random.rand(nx, ny) < 0.5) * np.float32(1)
            masks_vector.append(random_matrix)
            matrix_vector.append(random_matrix.flatten())
        Tmatrix_vector=np.linalg.pinv(np.matrix(matrix_vector))
        for j in masks_vector:
            intensity_vector.append(np.sum(j * img))
        res = np.reshape(
        np.matmul(Tmatrix_vector,intensity_vector), (63, 63))

    
        with tf.Session() as sess:
            saver.restore(sess,data_path)
            output = sess.run(
                        [out],
                        feed_dict={
                            r_input: res
                        })
        
        return output
def deepreconstruct(img,per):
    output= main(img,per)
    return output
if __name__ == "__main__":
 
    i = main2.Image()
    im = i.return_image(size=64).astype("uint8")
    percent = np.arange(0.1, 1.1, 0.1)
    for per in percent:
        output = deepreconstruct(im,per)
        print(ssim(im,output))
