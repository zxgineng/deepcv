from network import Graph
import tensorflow as tf
from tensorflow.contrib import slim
from utils import Config
from collections import OrderedDict


class Model:
    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.params = params
        self.inputs = features['inputs']
        self.targets = labels
        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            predictions=self.predictions)

    def build_graph(self):
        graph = Graph(self.mode)
        logits, locs, softmax_logits = graph.build(self.inputs)

        softmax_logits_dict = OrderedDict({f'softmax_feat{n+1}': softmax_logits[n] for n in range(len(softmax_logits))})
        locs_dict = OrderedDict({f'locs_feat{n+1}': locs[n] for n in range(len(locs))})
        softmax_logits_dict.update(locs_dict)
        self.predictions = softmax_logits_dict

        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(logits, locs)
            self._build_optimizer()

    def _build_loss(self, logits, locs):
        glabels_list = self.targets[:6]
        glocs_list = self.targets[6:]

        with tf.variable_scope('loss'):
            def smooth_l1_loss(x):
                """cal smooth_l1_loss"""
                absx = tf.abs(x)
                minx = tf.minimum(absx, 1)
                r = 0.5 * ((absx - 1) * minx + absx)
                return r

            flat_logits = []
            flat_glabels = []
            flat_locs = []
            flat_glocs = []
            # flatten all tensors
            for i in range(len(logits)):
                flat_logits.append(tf.reshape(logits[i], [-1, Config.data.num_classes]))
                flat_glabels.append(tf.reshape(glabels_list[i], [-1]))
                flat_locs.append(tf.reshape(locs[i], [-1, 4]))
                flat_glocs.append(tf.reshape(glocs_list[i], [-1, 4]))
            # after concatenation, logits.shape:[N*38*38*B + N*19*19*B + ... + N*1*1*B,21]
            logits = tf.concat(flat_logits, 0)
            glabels = tf.concat(flat_glabels, 0)
            locs = tf.concat(flat_locs, 0)
            glocs = tf.concat(flat_glocs, 0)
            # set glabels>0 as positive
            positive_mask = tf.greater(glabels, 0)
            fpmask = tf.cast(positive_mask, tf.float32)
            num_positive = tf.reduce_sum(fpmask)
            predictions = tf.nn.softmax(logits)
            negative_mask = tf.logical_not(positive_mask)
            fnmask = tf.cast(negative_mask, tf.float32)
            # predictions[:,0] is the prob of background
            # and negative equals negative_ratio times of positive
            negative_pred = tf.where(negative_mask, predictions[:, 0], 1 - fnmask)
            max_num_negative = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            num_negative = tf.minimum(tf.cast(Config.train.negative_ratio * num_positive, tf.int32), max_num_negative)
            value, index = tf.nn.top_k(-negative_pred, k=num_negative)
            pred_threshold = -value[-1]
            negative_mask = tf.logical_and(negative_mask, negative_pred < pred_threshold)
            fnmask = tf.cast(negative_mask, tf.float32)
            # cal class loss
            with tf.variable_scope('cross_entropy_loss'):
                cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=glabels)
                p_cross_loss = tf.reduce_sum(cross_loss * fpmask)
                n_cross_loss = tf.reduce_sum(cross_loss * fnmask)
                cross_entropy_loss = (p_cross_loss + n_cross_loss) / num_positive
                tf.losses.add_loss(cross_entropy_loss)
            # cal locs loss
            with tf.variable_scope('loc_loss'):
                weights = tf.expand_dims(Config.train.alpha * fpmask, axis=-1)
                loc_loss = smooth_l1_loss(locs - glocs)
                loc_loss = tf.reduce_sum(loc_loss * weights) / num_positive
                tf.losses.add_loss(loc_loss)
            self.loss = tf.losses.get_total_loss()

    def _build_optimizer(self):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(Config.train.learning_rate, global_step,
                                                   Config.train.learning_decay_steps, Config.train.learning_decay_rate)
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer=Config.train.get('optimizer', 'Adam'),
            learning_rate=learning_rate,
            summaries=['loss'],
            name="train_op")
