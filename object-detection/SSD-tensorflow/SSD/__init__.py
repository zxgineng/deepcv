from utils import Config
import collections
import tensorflow as tf
from tensorflow.contrib import slim

class Graph:

    def __init__(self, mode):
        self.mode = mode

    def build(self,inputs):

        end_points = collections.OrderedDict()

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME'):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME'):
                with tf.variable_scope('ssd_300_vgg', values=[inputs]):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
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

                    prediction_list, logits_list, locs_list = self.cal_all_pred_layer(end_points)
                    return logits_list,locs_list,prediction_list

    def cal_all_pred_layer(self,end_points):
        """
        cal logits of all feat layers, output as list
        :return logits_pred_list: list of tensor, class logits of all default boxes,shape[N,38,38,b,21]，[N,19,19,b,21],...,[N,1,1,b,21]
                 locs_pred_list: list of tensor, locs offset logits of all default boxes,shape:[N,38,38,b,4]，[N,19,19,b,4],...,[N,1,1,b,4]
        """
        logits_pred_list = []
        locs_pred_list = []
        softmax_logits = []

        def l2_normalization(inputs, scaling=True):
            """
            cal l2_norm on channel
            :param inputs: 4D tensor, shape-[N,H,W,C]
            :param scaling: bool
            :return outputs: 4D tensor, shape-[N,H,W,C]
            """
            with tf.variable_scope('L2Normalization'):
                inputs_shape = inputs.get_shape()
                channel_shape = inputs_shape[-1:]
                # cal l2_norm on channel
                outputs = tf.nn.l2_normalize(inputs, 3, epsilon=1e-12)
                # scalling
                if scaling:
                    # scale.shape == channel.shape
                    scale = slim.variable('gamma', channel_shape, tf.float32, tf.constant_initializer(1.0))
                    outputs = tf.multiply(outputs, scale)

                return outputs

        # cal the logits of every default box
        for i, layer in enumerate(Config.model.feat_layers):
            with tf.variable_scope(layer + '_box'):
                input = end_points[layer]
                # cal l2_norm of first feat layer
                if Config.model.normalizations[i] > 0:
                    input = l2_normalization(input)
                n, h, w, c = input.shape.as_list()
                num_box = len(Config.model.anchor_sizes[i]) + len(Config.model.anchor_ratios[i])
                loc_pred = slim.conv2d(input, num_box * 4, [3, 3], activation_fn=None, scope='conv_loc')
                # reshape:[N,H,W,B,4]
                loc_pred = tf.reshape(loc_pred, [-1, h, w, num_box, 4])
                cls_pred = slim.conv2d(input, num_box * Config.data.num_classes, [3, 3], activation_fn=None, scope='conv_cls')
                # reshape:[N,H,W,B,21]
                cls_pred = tf.reshape(cls_pred, [-1, h, w, num_box, Config.data.num_classes])

            logits_pred_list.append(cls_pred)
            locs_pred_list.append(loc_pred)
            softmax_logits.append(tf.nn.softmax(cls_pred))

        return softmax_logits, logits_pred_list, locs_pred_list

