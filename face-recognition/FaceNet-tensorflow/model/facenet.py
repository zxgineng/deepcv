import tensorflow as tf
import os
from model.inception_resnet_v1 import inference

class Facenet():
    def __init__(self, inputs,args=None):
        self.inputs = inputs
        self.args = args
        self.build_model()
        self.init_saver()
        if args.is_training:
            self.global_step = tf.train.create_global_step()
            self.cal_loss()
            self.optimizer()

    def build_model(self):
        self.keep_probability = tf.placeholder(tf.float32)
        prelogits, _ = inference(self.inputs,self.keep_probability,self.args.is_training,self.args.embedding_size)
        self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        self.anchor, self.positive, self.negative = tf.unstack(tf.reshape(self.embeddings, [-1, 3, self.args.embedding_size]), 3, 1)

    def cal_loss(self):
        with tf.variable_scope('triplet_loss'):
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(self.anchor, self.positive)), 1)
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(self.anchor, self.negative)), 1)
            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), self.args.alpha)
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
            tf.losses.add_loss(loss)
            self.loss =tf.losses.get_total_loss()

    def optimizer(self):
        self.initial_learning_rate = tf.placeholder(tf.float32)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step,
                                                   self.args.learning_rate_decay_epochs * self.args.epoch_size,
                                                   self.args.learning_rate_decay_factor, staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss, self.global_step)

    def load_latest(self, sess, check_point_dir):
        latest_checkpoint = tf.train.latest_checkpoint(check_point_dir)
        if latest_checkpoint:
            print("Loading latest model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        else:
            print('No checkpoint!')

    def load(self, sess, check_point_file):
        print("Loading model checkpoint {} ...".format(check_point_file))
        self.saver.restore(sess, check_point_file)
        print("Model loaded")

    def save(self, sess,dir,name):
        print("Saving model...")
        self.saver.save(sess, os.path.join(dir,name), self.global_step)
        print("Model saved")

    def init_saver(self):
        self.saver = tf.train.Saver()
