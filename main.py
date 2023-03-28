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

parser = argparse.ArgumentParser(description='Process some integers.')



#vector or processed
MODEL = "classic"
MODE = "train"

def classic_model(features, labels):
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

def qr_model(features, labels):
    """Model function for CNN."""
    # Input Layer
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

def vector_model(features,labels):
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

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=1,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)
    
    output_f = tf.layers.dense(tf.layers.flatten(conv3),4096)
    output = tf.reshape(output_f,[-1,64,64,1])
    
    return output

def test_model(features, labels):

    if MODEL == "vector":
        output=vector_model(features,labels)
    elif MODEL == "classic":
        output=classic_model(features,labels)
    elif MODEL == "qr":
        output=qr_model(features,labels)
    loss=tf.abs(tf_ssim(labels, output)-1.0)
    
    

    return output,loss

def train_model(features, labels):
    
    if MODEL == "vector":
        output=vector_model(features,labels)
    elif MODEL == "classic":
        output=classic_model(features,labels)
    elif MODEL == "qr":
        output=qr_model(features,labels)
    """
    loss = tf.losses.mean_squared_error(labels, output)
    
    loss=tf.abs(tf_ssim(labels, output)-1.0)
    """
    loss = 0.0001*tf.losses.mean_squared_error(labels, output) + 100*tf.abs(tf_ssim(labels, output)-1.0)
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001,global_step,
                                           1000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(
        loss=loss)
    
    tf.summary.scalar("learning_rate", learning_rate)

    lossG=tf.abs(tf_ssim(labels, output)-1.0)

    return train_op, lossG, output,labels


def main(argunused):
    
    per = np.arange(0.1, 1.1, 0.1)
    for percent in per:
        print(percent)
        if MODE == 'train':
            #ts = time.time()
            #st = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y_%H.%M.%S')
            experiment_dir = "C:\\Users\\eloy\\Desktop\\Code\\Python\\dp_one\\experiments\\" +"experiment("+str(percent)+")\\"
            #data_path = "C:\\Users\\eloy\\Desktop\\Code\\Python\\dp_one\\experiments\\experiment(10000)\\model-6259000.ckpt"
            tf.reset_default_graph()
            r_input = tf.placeholder(tf.float32, None)
            labels = tf.placeholder(tf.float32, None)
        
            train_op, loss,out,original = train_model(r_input, labels)
        
            saver = tf.train.Saver()
            tf.summary.scalar("loss", loss)
            tf.summary.image("input",r_input)
            tf.summary.image("original",original)
            tf.summary.image("reconstructed",out)
        
            summary_op = tf.summary.merge_all()
           
            with tf.Session() as sess:
                #saver.restore(sess,data_path)
                writer = tf.summary.FileWriter(experiment_dir, sess.graph)
                sess.run(tf.global_variables_initializer())
                ge = 0
                number_of_batches = 1167
                i = 0
                for _ in range(0,5):
                    x = list(range(1,number_of_batches+1))
                    shuffle(x)
                    batch_size=1
                    for ind in x:
                        if MODEL=="vector":
                            r = np.load("images_2/res_1000_" + str(ind) + ".npy")
                            l = np.load("images_2/im_" + str(ind) + ".npy")
                        elif MODEL=="classic":
                            r = np.load("data_set/intensities_" + str(ind) + ".npy")
                            l = np.load("data_set/im_" + str(ind) + ".npy")
                            num = int(len(r[0,:,0])*percent)
                            batch_res = np.zeros([num, 64, 64, 1], dtype="float32")
                            for ip in range(0,256):
                                batch_res[ip, :, :, 0]=process_data_set(percent,r[ip,:,0])
                        elif MODEL=="qr":
                            r = np.load("images_qr/res_1000_" + str(ind) + ".npy")
                            l = np.load("images_qr/im_63_" + str(ind) + ".npy")
                        
                        x2 = list(range(0,int(256/batch_size)))
                        shuffle(x2)
                        for k in x2:
                            if MODEL == "vector":
                                rp = r[(batch_size)*k:(batch_size)*k+batch_size,:,:,:]
                            elif MODEL == "classic":
                                rp = batch_res[(batch_size)*k:(batch_size)*k+batch_size,:,:,:]
                            elif MODEL == "qr":
                                rp = r[(batch_size)*k:(batch_size)*k+batch_size,:,:,:]
                            lp = l[(batch_size)*k:(batch_size)*k+batch_size,:,:,:]
                            _, loss_value, summary,output = sess.run(
                                [train_op, loss, summary_op,out],
                                feed_dict={
                                    r_input: rp,
                                    labels: lp
                                })
                            if i % 255 == 0 and i != 0:
                                writer.add_summary(summary, global_step=ge)
                                saver.save(sess, experiment_dir + "model-"+str(percent)+"-" + str(i) + ".ckpt")
                                print(i)
                                print("LOSS:" + str(loss_value))
                            ge = ge+1
                            i = i + 1
                global matrix_vector
                global masks_vector
                global Tmatrix_vector
                del(matrix_vector)
                del(masks_vector)
                del(Tmatrix_vector)
                sess.close()

        elif MODE == "test":
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            data_path = "C:\\Users\\eloy\\Desktop\\Code\\Python\\dp_one\\experiments\\13-12-2017_13.05.05\\model-179850.ckpt"
            
            r_input = tf.placeholder(tf.float32, None)
            labels = tf.placeholder(tf.float32, None)
            out,loss = test_model(r_input, labels)
            saver = tf.train.Saver()

            if MODEL=="vector":
                print(1)
            elif MODEL == "classic":
                print(2)
            elif MODEL == "qr":
                rT = np.load("images_qr/res_1000_" + str(0) + ".npy") 
                lT = np.load("images_qr/im_63_" + str(0) + ".npy")
                rpT = rT[0:10,:,:,:]
                lpT = lT[0:10,:,:,:]

                r = np.load("images_qr/res_1000_" + str(2800) + ".npy") 
                l = np.load("images_qr/im_63_" + str(2800) + ".npy")
                rp = r[0:255,:,:,:]
                lp = l[0:255,:,:,:]
            with tf.Session() as sess:
        
                saver.restore(sess,data_path)
                outputT, loss_value, = sess.run(
                            [out, loss],
                            feed_dict={
                                r_input: rpT,
                                labels: lpT
                            })
                print(loss_value)
                output, loss_value, = sess.run(
                            [out, loss],
                            feed_dict={
                                r_input: rp,
                                labels: lp
                            })
                print(loss_value)
                plt.subplot(4,2,1)
                plt.imshow(lpT[0,:,:,0],"gray")      
                plt.subplot(4,2,2)
                plt.imshow(outputT[0,:,:,0],"gray")
                plt.subplot(4,2,3)           
                plt.imshow(lp[1,:,:,0],"gray")
                plt.subplot(4,2,4)      
                plt.imshow(output[1,:,:,0],"gray")
                plt.subplot(4,2,5)           
                plt.imshow(lp[2,:,:,0],"gray")
                plt.subplot(4,2,6)      
                plt.imshow(output[2,:,:,0],"gray")
                plt.subplot(4,2,7)           
                plt.imshow(lp[10,:,:,0],"gray")
                plt.subplot(4,2,8)      
                plt.imshow(output[10,:,:,0],"gray")
                plt.show()
    

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def psnr(target, ref, scale):
	#assume RGB image
	target_data = np.array(target)
	target_data = target_data[scale:-scale, scale:-scale]

	ref_data = np.array(ref)
	ref_data = ref_data[scale:-scale, scale:-scale]
	
	diff = ref_data - target_data
	diff = diff.flatten('C')
	rmse = math.sqrt( np.mean(diff ** 2.) )
	return 20*math.log10(1.0/rmse)



def process_data_set(percent,intensity_vector):
    global matrix_vector
    global masks_vector
    global Tmatrix_vector
    num = int(len(intensity_vector)*percent)
    intensity_vector= intensity_vector[:num]
    try:
        matrix_vector
        masks_vector
        Tmatrix_vector
    except:
        matrix_vector = []
        masks_vector = []
        np.random.seed(1)
        for _ in range(0, num):
            random_matrix = (np.random.rand(64, 64) < 0.5) * np.float32(1)
            masks_vector.append(random_matrix)
            matrix_vector.append(random_matrix.flatten())
        Tmatrix_vector=np.linalg.pinv(np.matrix(matrix_vector))
    result = np.reshape(
                np.matmul(Tmatrix_vector,intensity_vector), (64, 64))
    return result
if __name__ == "__main__":
    tf.app.run()
