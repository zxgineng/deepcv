# from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim


class Graph:
    def __init__(self, mode):
        self.mode = mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

    def build(self, inputs):
        cnn_layer = self.build_cnn(inputs)
        net = tf.reshape()


    def build_cnn(self,inputs):

        with tf.variable_scope('cnn_layer',[inputs]):

            net = slim.conv2d(inputs, 64, 3)
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 128, 3)
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 256, 3)
            net = slim.conv2d(net, 256, 3)
            net = slim.max_pool2d(net, [2, 1], [2, 1])
            net = slim.conv2d(net, 512, 3)
            net = slim.batch_norm(net, is_training=self.is_training)
            net = slim.conv2d(net, 512, 3)
            net = slim.batch_norm(net, is_training=self.is_training)
            net = slim.max_pool2d(net, [2, 1], [2, 1])
            net = slim.conv2d(net, 512, 2, padding='VALID')

            return net


        # return vcoords_logits, side_logits,scores_softmax

if __name__ == '__main__':
    graph = Graph('train')
    x = tf.placeholder(tf.float32,[None,32,None,1])
    graph.build(x)
