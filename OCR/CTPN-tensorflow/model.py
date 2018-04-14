import tensorflow as tf
from tensorflow.contrib import slim

from utils import Config
import CTPN


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
        # a= (tf.trainable_variables())
        # for v in a:
        #     print(v)
        # exit()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions=self.predictions)

    def build_graph(self):
        graph = CTPN.Graph(self.mode)
        vcoords_logits, scores_logits, side_logits = graph.build(self.inputs)
        scores_softmax = tf.nn.softmax(scores_logits)
        self.predictions = {'vcoords_logits': vcoords_logits, 'scores_softmax': scores_softmax, 'side_logits': side_logits}
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(vcoords_logits, scores_logits, side_logits)
            self._build_optimizer()
            # self._build_metric()

    def _build_loss(self, vcoords_logits, scores_logits, side_logits):
        coords, labels, side = self.targets
        coords = tf.squeeze(coords,0)
        labels = tf.squeeze(labels,0)
        side = tf.squeeze(side,0)

        with tf.variable_scope('loss'):
            def smooth_l1_loss(x):
                """cal smooth_l1_loss"""
                absx = tf.abs(x)
                minx = tf.minimum(absx, 1)
                r = 0.5 * ((absx - 1) * minx + absx)
                return r

            positive_mask = tf.greater(labels, 0)
            fpmask = tf.cast(positive_mask, tf.float32)
            num_positive = tf.reduce_sum(fpmask)
            num_positive = tf.reduce_min(tf.stack([num_positive, Config.train.num_positive]))

            negative_mask = tf.logical_not(positive_mask)
            fnmask = tf.cast(negative_mask, tf.float32)
            max_num_negative = tf.reduce_sum(fnmask)
            num_negative = tf.reduce_max(tf.stack([Config.train.num_negtive, (
                        Config.train.num_positive + Config.train.num_negtive - num_positive)]))
            num_negative = tf.reduce_min(tf.stack([num_negative,max_num_negative]))
            scores_softmax = tf.nn.softmax(scores_logits)
            # filter positive samples by most loss
            positive_pred = tf.where(positive_mask, scores_softmax[:,1], 1.0 - fpmask)
            value, pindex = tf.nn.top_k(-positive_pred, k=tf.cast(num_positive,tf.int32))
            # positive_threshold = -value[-1]
            # positive_mask = tf.logical_and(positive_mask,positive_pred<= positive_threshold)
            # fpmask = tf.cast(positive_mask, tf.float32)
            # num_positive = tf.reduce_sum(fpmask)

            # filter negative samples by most loss
            negative_pred = tf.where(negative_mask, scores_softmax[:,0], 1.0 - fnmask)
            value, nindex = tf.nn.top_k(-negative_pred, k=tf.cast(num_negative,tf.int32))
            # negative_threshold = -value[-1]
            # negative_mask = tf.logical_and(negative_mask, negative_pred<= negative_threshold)
            # fnmask = tf.cast(negative_mask, tf.float32)
            # num_negative = tf.reduce_sum(fnmask)

            # filter side loss
            side_mask = tf.greater(side,-1)
            fside_mask = tf.cast(side_mask,tf.float32)
            num_side = tf.reduce_sum(fside_mask)

            with tf.variable_scope('xentropy_loss'):
                xentropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_logits, labels=labels)
                # p_loss = tf.reduce_sum(xentropy_loss * fpmask)
                # n_loss = tf.reduce_sum(xentropy_loss * fnmask)
                p_loss = tf.reduce_sum(tf.gather(xentropy_loss,pindex))
                n_loss = tf.reduce_sum(tf.gather(xentropy_loss,nindex))
                xentropy_loss = (p_loss + n_loss) / (num_positive + num_negative)
                tf.losses.add_loss(xentropy_loss)

            with tf.variable_scope('reg_loss'):
                # weights = tf.expand_dims(Config.train.reg_weight * fpmask, axis=-1)
                reg_loss = smooth_l1_loss(vcoords_logits - coords)
                # reg_loss = tf.reduce_sum(reg_loss * weights) / num_positive
                reg_loss = tf.reduce_sum(tf.gather(reg_loss,pindex)) * Config.train.reg_weight /num_positive
                tf.losses.add_loss(reg_loss)

            with tf.variable_scope('side_loss'):
                side_loss = smooth_l1_loss(side_logits-side)
                side_loss = tf.reduce_sum(fside_mask * side_loss * Config.train.side_weight) / num_side
                tf.losses.add_loss(side_loss)

            self.loss = tf.losses.get_total_loss()
            pred = tf.argmax(scores_logits,-1)
            # accuracy = tf.metrics.accuracy(tf.cast(labels,tf.float32),tf.cast(pred,tf.float32))

    def _build_optimizer(self):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(Config.train.learning_rate, global_step,
                                                   Config.train.learning_decay_steps, Config.train.learning_decay_rate)
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer='Adam',
            learning_rate=learning_rate,
            clip_gradients= Config.train.max_gradient_norm,
            summaries=['loss'],
            name="train_op")
