import os
from tensorflow.contrib import slim
from model.model_utils import *
import numpy as np

class ONet():
    def __init__(self, input, target=None):
        self.input = input
        self.target = target
        self.build_model()
        self.init_saver()
        if target:
            self.global_step = tf.Variable(0, trainable=False)
            self.label = target[0]
            self.bbox_target = target[1]
            self.landmark_target = target[2]
            self.cal_loss()
            self.optimizer()


    def build_model(self):
        with tf.variable_scope('ONet'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=prelu,
                                weights_initializer=slim.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                padding='valid'):
                net = slim.conv2d(self.input, num_outputs=32, kernel_size=[3, 3], stride=1, scope="conv1")
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
                net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv2")
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
                net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv3")
                net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
                net = slim.conv2d(net, num_outputs=128, kernel_size=[2, 2], stride=1, scope="conv4")
                fc_flatten = slim.flatten(net)
                fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope="fc1",activation_fn=prelu)
                # batch*2
                self.cls_pred = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc", activation_fn=tf.nn.softmax)
                # batch*4
                self.bbox_pred = slim.fully_connected(fc1, num_outputs=4, scope="bbox_fc", activation_fn=None)
                # batch*10
                self.landmark_pred = slim.fully_connected(fc1, num_outputs=10, scope="landmark_fc", activation_fn=None)



    def cal_loss(self):
        cls_loss = cls_ohem(self.cls_pred, self.label)
        tf.losses.add_loss(cls_loss)
        bbox_loss = bbox_ohem(self.bbox_pred, self.bbox_target, self.label)
        tf.losses.add_loss(bbox_loss)
        landmark_loss = landmark_ohem(self.landmark_pred, self.landmark_target, self.label)
        tf.losses.add_loss(landmark_loss)
        self.accuracy = cal_accuracy(self.cls_pred, self.label)
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
        self.saver = tf.train.Saver(var_list=tf.global_variables('ONet'))

