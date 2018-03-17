import tensorflow as tf
import os
from tensorflow.contrib import slim
from model.model_utils import *
import numpy as np

class PNet():
    def __init__(self, input, target=None):
        self.input = input
        self.target = target
        self.build_model()
        self.init_saver()
        if target:
            self.global_step = tf.Variable(0, trainable=False)
            self.label = target[0]
            self.bbox_target = target[1]
            self.cal_loss()
            self.optimizer()


    def build_model(self):
        with tf.variable_scope('PNet'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=prelu,
                                weights_initializer=slim.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding='valid'):
                net = slim.conv2d(self.input, 10, 3, stride=1, scope='conv1')
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1', padding='SAME')
                net = slim.conv2d(net, num_outputs=16, kernel_size=[3, 3], stride=1, scope='conv2')
                net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=1, scope='conv3')
                # batch*1*1*2
                self.cls_pred = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1, scope='conv4_1',
                                            activation_fn=tf.nn.softmax)
                # batch*1*1*4
                self.bbox_pred = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1, scope='conv4_2',
                                             activation_fn=None)


    def cal_loss(self):
        cls_prob = tf.squeeze(self.cls_pred,[1, 2], name='cls_prob')
        cls_loss = cls_ohem(cls_prob, self.label)
        tf.losses.add_loss(cls_loss)
        bbox_pred = tf.squeeze(self.bbox_pred, [1, 2], name='bbox_pred')
        bbox_loss = bbox_ohem(bbox_pred, self.bbox_target, self.label)
        tf.losses.add_loss(bbox_loss)
        self.accuracy = cal_accuracy(cls_prob, self.label)
        self.loss = tf.losses.get_total_loss()


    def optimizer(self):
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss, self.global_step)


    def load_latest(self, sess, check_point_dir):
        latest_checkpoint = tf.train.latest_checkpoint(check_point_dir)
        if latest_checkpoint:
            print("Loading latest model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")


    def load(self, sess, check_point_file):
        print("Loading model checkpoint {} ...".format(check_point_file))
        self.saver.restore(sess, check_point_file)
        print("Model loaded")


    def save(self, sess, dir, name):
        print("Saving model...")
        self.saver.save(sess, os.path.join(dir, name), self.global_step)
        print("Model saved")


    def init_saver(self):
        self.saver = tf.train.Saver(var_list=tf.global_variables('PNet'))

