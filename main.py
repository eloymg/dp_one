import tensorflow as tf
import numpy as np
import scipy.misc
from PIL import Image
import requests


def cnn_model_fn(features, labels):
    """Model function for CNN."""
    # Input Layer
    #input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])
    input_layer = tf.reshape(features, [-1, 64, 64, 1])

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
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #3
    output = tf.layers.conv2d(
        inputs=conv2,
        filters=1,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)

    #if mode == tf.estimator.ModeKeys.PREDICT:
    #  return tf.estimator.EstimatorSpec(mode=mode, predictions=output)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, output)

    # Configure the Training Op (for TRAIN mode)
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #if mode == "train":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
    #return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    #eval_metric_ops = {
    #   "accuracy": tf.metrics.accuracy(
    #      labels=labels, predictions=predictions["classes"])}
    #return tf.estimator.EstimatorSpec(
    #   mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return train_op, loss


"""
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
"""


def main(argunused):
    """
    Arguments:
        raw_input numpy tensor of size dim x dim 

    """
    fh = open("fall11_urls.txt")
    #r_input,labels = generate_batch(1,fh,masks_vector,matrix_vector)
    r_input = tf.placeholder(tf.float32, None)
    labels = tf.placeholder(tf.float32,None)
    train_op, loss = cnn_model_fn(r_input, labels)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        while (i < 100):
            r, l = generate_batch(200, fh)
            _, loss_value = sess.run(
                [train_op, loss], feed_dict={
                    r_input: r,
                    labels: l
                })
            print("LOSS:" + str(loss_value))
            i = i + 1


#     # Create the Estimator
#     classifier = tf.estimator.Estimator(
#       model_fn=cnn_model_fn, model_dir="C:/tmp/test_convnet_model")

#   # Train the model
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": expanded_input},
#         y=labels,
#         batch_size=1,
#         num_epochs=5,
#         shuffle=True)
#     classifier.train(input_fn=train_input_fn, steps=100)

global matrix_vector
global masks_vector


def generate_batch(batch_size, fh):
    batch_im = np.zeros([batch_size, 64, 64, 1], dtype="float32")
    batch_res = np.zeros([batch_size, 64, 64, 1], dtype="float32")
    try:
        matrix_vector
        masks_vector
    except:
        matrix_vector = []
        masks_vector = []
        np.random.seed(1)
        for _ in range(0, 1000):
            random_matrix = (np.random.rand(64, 64) < 0.5) * np.float32(1)
            masks_vector.append(random_matrix)
            matrix_vector.append(random_matrix.flatten())
    for i in range(0, batch_size):
        intensity_vector = []
        im = []
        res = []
        count = 0
        while len(im) == 0 and len(res) == 0:
            url = fh.readline().split()[1]
            try:
                im = np.asarray(
                    Image.open(
                        requests.get(url, stream=True, timeout=0.5).raw))
                im = scipy.misc.imresize(im, (64, 64))[:, :, 0]
                print(i)
            except:
                im = []
                res = []
                continue
            for j in masks_vector:
                intensity_vector.append(np.sum(j * im))
            im = im.astype('float32')

            res = np.reshape(
                np.matmul(intensity_vector, matrix_vector), (64, 64))
            count += 1
        batch_im[i, :, :, 0] = im
        batch_im = batch_im.astype('float32')
        batch_res[i, :, :, 0] = res
        batch_res = batch_res.astype('float32')
    return batch_im, batch_res


if __name__ == "__main__":
    tf.app.run()
