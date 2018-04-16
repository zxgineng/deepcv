import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import numpy as np

from utils import Config
import architecture


class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, params):
        self.mode = mode
        self.params = params
        self.inputs = features['inputs']
        self.loss, self.train_op, self.metrics, self.predictions = None, None, None, None
        self.build_graph()

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops=self.metrics,
            predictions={"prediction": self.predictions})

    def build_graph(self):
        graph = architecture.Graph(self.mode)
        outputs, end_points = graph.build(self.inputs)

        self.predictions = outputs
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self._build_loss(outputs, end_points)
            self._build_optimizer()

    def _build_loss(self, outputs, end_points):

        def content_loss(endpoints_dict, content_layers):
            content_loss = 0
            for layer in content_layers:
                generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
                size = tf.size(generated_images)
                content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)
            return content_loss

        def get_style_features():
            """
            For the "style_image", the preprocessing step is:
            1. Resize the shorter side to FLAGS.image_size
            2. Central crop
            """
            with tf.Graph().as_default():
                image = cv2.imread(Config.data.style_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                shape = image.shape
                scale = Config.model.image_size / min(shape[0:2])
                image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                shape = image.shape
                crop_xmin = (shape[1] - Config.model.image_size) // 2
                crop_xmax = (shape[1] - Config.model.image_size) // 2 + Config.model.image_size
                crop_ymin = (shape[0] - Config.model.image_size) // 2
                crop_ymax = (shape[0] - Config.model.image_size) // 2 + Config.model.image_size
                image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
                # Add the batch dimension
                image = np.expand_dims(image, 0)
                image = tf.cast(image, tf.float32) - Config.model.channels_mean
                _, end_points = architecture.vgg_16(image)

                features = []
                for layer in Config.model.style_layers:
                    feature = end_points[layer]
                    feature = tf.squeeze(gram(feature), [0])
                    features.append(feature)

                with tf.Session() as sess:
                    # Restore variables
                    init_func = slim.assign_from_checkpoint_fn('logs/pretrained/vgg_16.ckpt', tf.trainable_variables())
                    init_func(sess)
                    return sess.run(features)

        def gram(layer):
            shape = layer.shape
            num_images = shape[0]
            width = shape[1]
            height = shape[2]
            num_filters = shape[3]
            filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
            grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

            return grams

        def style_loss(endpoints_dict, style_features_t, style_layers):
            style_loss = 0
            for style_gram, layer in zip(style_features_t, style_layers):
                generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
                size = tf.size(generated_images)
                layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
                style_loss += layer_style_loss
            return style_loss

        def total_variation_loss(layer):
            shape = layer.shape
            height = shape[1]
            width = shape[2]
            y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0],
                                                                                             [-1, -1, -1, -1])
            x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0],
                                                                                            [-1, -1, -1, -1])
            loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
            return loss

        style_features_t = get_style_features()

        with tf.variable_scope('loss'):
            content_loss = Config.model.content_weight * content_loss(end_points, Config.model.content_layers)
            style_loss = Config.model.style_weight * style_loss(end_points, style_features_t, Config.model.style_layers)
            tv_loss = Config.model.tv_weight * total_variation_loss(outputs)

            self.loss = style_loss + content_loss + tv_loss

    def _build_optimizer(self):
        global_step = tf.train.get_global_step()
        self.train_op = slim.optimize_loss(
            self.loss, global_step,
            optimizer='Adam',
            learning_rate=Config.train.learning_rate,
            summaries=['loss'],
            variables=tf.trainable_variables('generation'),
            name="train_op")

