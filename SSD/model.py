import collections
import os
from util import *


class SSD():
    def __init__(self, input, target=None):
        self.input = input
        self.target = target
        self.global_step = tf.train.create_global_step()
        self.build_model()
        self.init_saver()
        if target:
            self.cal_loss()
            self.optimizer()

    def build_model(self):
        """
        建立模型
        """
        end_points = collections.OrderedDict()

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME'):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME'):
                with tf.variable_scope('ssd_300_vgg', values=[self.input]):
                    net = slim.repeat(self.input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    end_points['block1'] = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    end_points['block2'] = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    end_points['block3'] = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    end_points['block4'] = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    end_points['block5'] = net
                    net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')  # max pool

                    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
                    end_points['block6'] = net
                    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
                    end_points['block7'] = net
                    net = slim.conv2d(net, 256, [1, 1], scope='block8/conv1x1')
                    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
                    net = slim.conv2d(net, 512, [3, 3], stride=2, scope='block8/conv3x3', padding='VALID')

                    end_points['block8'] = net
                    net = slim.conv2d(net, 128, [1, 1], scope='block9/conv1x1')
                    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
                    net = slim.conv2d(net, 256, [3, 3], stride=2, scope='block9/conv3x3', padding='VALID')
                    end_points['block9'] = net
                    net = slim.conv2d(net, 128, [1, 1], scope='block10/conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], scope='block10/conv3x3', padding='VALID')
                    end_points['block10'] = net
                    net = slim.conv2d(net, 128, [1, 1], scope='block11/conv1x1')
                    net = slim.conv2d(net, 256, [3, 3], scope='block11/conv3x3', padding='VALID')
                    end_points['block11'] = net

                    self.prediction_list, self.logits_list, self.locs_list = cal_all_pred_layer(end_points)

    def cal_loss(self, negative_ratio=3.0, alpha=1.0):
        """
        计算loss
        :param negative_ratio: 负样本与正样本的比值
        :param alpha: 位置损失的权重
        """
        glabels_list = self.target[:6]
        glocs_list = self.target[6:]

        with tf.variable_scope('loss'):
            def smooth_l1_loss(x):
                """
                计算smooth_l1_loss
                """
                absx = tf.abs(x)
                minx = tf.minimum(absx, 1)
                r = 0.5 * ((absx - 1) * minx + absx)
                return r

            flat_logits = []
            flat_glabels = []
            flat_locs = []
            flat_glocs = []
            # flatten 所有 tensors
            for i in range(len(self.logits_list)):
                flat_logits.append(tf.reshape(self.logits_list[i], [-1, NUM_CLASS]))
                flat_glabels.append(tf.reshape(glabels_list[i], [-1]))
                flat_locs.append(tf.reshape(self.locs_list[i], [-1, 4]))
                flat_glocs.append(tf.reshape(glocs_list[i], [-1, 4]))
            # concat之后, logits.shape为[N*38*38*B + N*19*19*B + ... + N*1*1*B,21]
            logits = tf.concat(flat_logits, 0)
            glabels = tf.concat(flat_glabels, 0)
            locs = tf.concat(flat_locs, 0)
            glocs = tf.concat(flat_glocs, 0)
            # glabels>0的为正样本
            positive_mask = tf.greater(glabels , 0)
            fpmask = tf.cast(positive_mask, tf.float32)
            num_positive = tf.reduce_sum(fpmask)
            predictions = tf.nn.softmax(logits)
            negative_mask = tf.logical_not(positive_mask)
            fnmask = tf.cast(negative_mask, tf.float32)
            # predictions[:,0] 为背景的预测概率 加入negative_ratio倍正样本的负样本
            negative_pred = tf.where(negative_mask, predictions[:, 0], 1 - fnmask)
            max_num_negative = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            num_negative = tf.minimum(tf.cast(negative_ratio * num_positive, tf.int32), max_num_negative)
            value, index = tf.nn.top_k(-negative_pred, k=num_negative)
            pred_threshold = -value[-1]
            negative_mask = tf.logical_and(negative_mask, negative_pred < pred_threshold)
            fnmask = tf.cast(negative_mask, tf.float32)
            # 计算分类loss
            with tf.variable_scope('cross_entropy_loss'):
                cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=glabels)
                p_cross_loss = tf.reduce_sum(cross_loss * fpmask)
                n_cross_loss = tf.reduce_sum(cross_loss * fnmask)
                cross_entropy_loss = (p_cross_loss + n_cross_loss) / num_positive
                tf.losses.add_loss(cross_entropy_loss)
            # 计算位置loss
            with tf.variable_scope('loc_loss'):
                weights = tf.expand_dims(alpha * fpmask, axis=-1)
                loc_loss = smooth_l1_loss(locs - glocs)
                loc_loss = tf.reduce_sum(loc_loss * weights) / num_positive
                tf.losses.add_loss(loc_loss)
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

    def save(self, sess,dir,name):
        print("Saving model...")
        self.saver.save(sess, os.path.join(dir,name), self.global_step)
        print("Model saved")

    def init_saver(self, var_list=None):
        self.saver = tf.train.Saver(var_list=var_list)


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 300, 300, 3])
    SSD(x)
    print(tf.trainable_variables())
