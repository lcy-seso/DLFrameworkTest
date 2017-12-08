#coding=utf-8
import numpy as np
import tensorflow as tf
import pdb

from load_MNIST import load_MNIST


def data_iterator(dataset="training", path="data", batch_size=128):
    batch_idx = 0
    lbl, img = load_MNIST(dataset, path)
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(lbl))
        np.random.shuffle(idxs)
        shuf_features = img[idxs]
        shuf_labels = lbl[idxs]
        for batch_idx in range(0, len(lbl), batch_size):
            images_batch = shuf_features[batch_idx:
                                         batch_idx + batch_size] / 255.
            images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:
                                       batch_idx + batch_size].astype("int32")
            yield images_batch, labels_batch


def main():
    # define the network topology.
    # placeholder to recieve data.
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(
        tf.int32, shape=[
            None,
        ])

    # Define the learnable parameters.
    with tf.name_scope("hidden"):
        W = tf.Variable(tf.zeros([784, 10]), name="weights")
        b = tf.Variable(tf.zeros([10]), name="bias")
        y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_op = tf.train.AdamOptimizer().minimize(cross_entropy)

    # define the initializer.
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    train_reader = data_iterator()
    test_lbl, test_img = load_MNIST("testing")
    for _ in range(1000):
        images_batch, labels_batch = train_reader.next()
        _, loss_val = sess.run(
            [train_op, cross_entropy],
            feed_dict={
                x: images_batch,
                y_: labels_batch.astype("int32")
            })
        print("Cur Cost : %f" % loss_val)

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: test_img, y_: test_lbl}))


if __name__ == "__main__":
    main()
