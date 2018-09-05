from utils import Config
import tensorflow as tf
from tensorflow.contrib import slim


class GeneratorGraph():
    def __init__(self, mode, name):
        self.mode = mode
        self.name = name

    # follow the naming convention used in the original paper
    def _c7s1_k(self, inputs, num_outputs, activation_fn=tf.nn.relu, normalizer_fn=slim.instance_norm):
        """ a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1"""
        padded = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        conv = slim.conv2d(padded, num_outputs, 7, 1, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn,
                           biases_initializer=None)
        return conv

    def _dk(self, inputs, num_outputs, normalizer_fn=slim.instance_norm):
        """a 3 × 3Convolution-InstanceNorm-ReLU layer with k filters, andstride 2."""
        conv = slim.conv2d(inputs, num_outputs, 3, 2, 'SAME', normalizer_fn=normalizer_fn, biases_initializer=None)
        return conv

    def _Rk(self, inputs, num_outputs, normalizer_fn=slim.instance_norm):
        """ A residual block that contains two 3x3 convolutional layers with the same number of filters on both layer"""
        padded1 = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        conv1 = slim.conv2d(padded1, num_outputs, 3, 1, 'VALID', normalizer_fn=normalizer_fn, biases_initializer=None)

        padded2 = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        conv2 = slim.conv2d(padded2, num_outputs, 3, 1, 'VALID', normalizer_fn=normalizer_fn, activation_fn=None,
                            biases_initializer=None)
        outputs = inputs + conv2
        return outputs

    def _n_res_blocks(self, inputs, n=6):
        depth = inputs.shape[3]
        for i in range(1, n + 1):
            outputs = self._Rk(inputs, depth)
            inputs = outputs
        return outputs

    def _uk(self, inputs, num_outputs, normalizer_fn=slim.instance_norm, output_size=None, name=None):
        """ A 3x3 transposed-Convolution-BatchNorm-ReLU layer with k filters, stride 1/2"""
        with tf.variable_scope(name, 'transposed_convolution'):
            shape = inputs.shape
            weights = tf.get_variable('weights', [3, 3, num_outputs, shape[3]], tf.float32,
                                      tf.random_normal_initializer(stddev=0.02))
            if not output_size:
                output_size = shape[1] * 2
            output_shape = tf.stack([shape[0], output_size, output_size, num_outputs])
            fsconv = tf.nn.conv2d_transpose(inputs, weights, output_shape, [1, 2, 2, 1])
            normalized = normalizer_fn(fsconv)
            outputs = tf.nn.relu(normalized)
            return outputs

    def build(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            num = Config.model.base_generator_filter
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.random_normal_initializer(stddev=0.02)):
                # conv layers
                c7s1_32 = self._c7s1_k(inputs, num)

                d64 = self._dk(c7s1_32, num * 2)
                d128 = self._dk(d64, num * 4)

                # 9 blocks for 256x256 images
                res_outputs = self._n_res_blocks(d128, n=9)

                # transposed convolution
                u64 = self._uk(res_outputs, num * 2, name='u64')
                u32 = self._uk(u64, num, output_size=Config.model.image_size, name='u32')

                # conv layer
                # the paper said that ReLU and _norm were used but actually tanh was used and no _norm here
                outputs = self._c7s1_k(u32, 3, normalizer_fn=None, activation_fn=tf.nn.tanh)

                return outputs


class DiscriminatorGraph():
    def __init__(self, name):
        self.name = name

    def build(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.random_normal_initializer(stddev=0.02)):
                C64 = slim.conv2d(inputs, 64, 4, 2, activation_fn=tf.nn.leaky_relu, biases_initializer=None)
                C128 = slim.conv2d(C64, 128, 4, 2, normalizer_fn=slim.instance_norm, activation_fn=tf.nn.leaky_relu,
                                   biases_initializer=None)
                C256 = slim.conv2d(C128, 256, 4, 2, normalizer_fn=slim.instance_norm, activation_fn=tf.nn.leaky_relu,
                                   biases_initializer=None)
                C512 = slim.conv2d(C256, 512, 4, 2, normalizer_fn=slim.instance_norm, activation_fn=tf.nn.leaky_relu,
                                   biases_initializer=None)
                outputs = slim.conv2d(C512, 1, 4, 1, activation_fn=None)
                return outputs

