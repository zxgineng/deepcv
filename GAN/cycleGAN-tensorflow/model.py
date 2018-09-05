import tensorflow as tf
import numpy as np

from utils import Config
from network import GeneratorGraph
from network import DiscriminatorGraph


class Model:

    def __init__(self):
        pass

    def model_fn(self, mode, features, labels, params):
        self.mode = mode
        self.params = params
        self.imageX = features
        self.imageY = labels
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
            predictions=self.predictions,
            training_hooks=self.training_hooks)

    def build_graph(self):
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            if Config.predict.model not in ['G', 'F']:
                raise ValueError('Config.predict.model must be G or F')
            generator = GeneratorGraph(self.mode, Config.predict.model)
            gen_images = generator.build(self.imageX)
            images = tf.image.convert_image_dtype(gen_images / 2 + 0.5, tf.uint8, name='generated_images')
            self.predictions = images

        else:
            G = GeneratorGraph(self.mode, 'G')
            F = GeneratorGraph(self.mode, 'F')
            fake_Y = G.build(self.imageX)  # X->Y
            fake_X = F.build(self.imageY)  # Y->X
            cycle_X = F.build(fake_Y)  # fake_Y->X
            cycle_Y = G.build(fake_X)  # fake_X->Y
            fake_X_buffer = tf.placeholder(tf.float32, [None, Config.model.image_size, Config.model.image_size, 3],
                                           'fake_X_buffer')
            fake_Y_buffer = tf.placeholder(tf.float32, [None, Config.model.image_size, Config.model.image_size, 3],
                                           'fake_Y_buffer')

            D_X = DiscriminatorGraph('D_X')
            D_X_real = D_X.build(self.imageX)
            D_X_fake = D_X.build(fake_X)
            D_X_fake_buffer = D_X.build(fake_X_buffer)
            D_Y = DiscriminatorGraph('D_Y')
            D_Y_real = D_Y.build(self.imageY)
            D_Y_fake = D_Y.build(fake_Y)
            D_Y_fake_buffer = D_Y.build(fake_Y_buffer)

            tf.summary.image('X/cycle_X_image', cycle_X)
            tf.summary.image('X/fake_Y_image', fake_Y)
            tf.summary.image('X/imageX', self.imageX)
            tf.summary.image('Y/cycle_Y_image', cycle_Y)
            tf.summary.image('Y/fake_X_image', fake_X)
            tf.summary.image('Y/imageY', self.imageY)

            self._build_loss(cycle_X, cycle_Y, D_X_real, D_X_fake, D_Y_real, D_Y_fake, D_X_fake_buffer, D_Y_fake_buffer)
            self._build_train_op(fake_X, fake_Y, self.imageX, self.imageY)

    def _build_loss(self, cycle_X, cycle_Y, D_X_real, D_X_fake, D_Y_real, D_Y_fake, D_X_fake_buffer, D_Y_fake_buffer):
        with tf.variable_scope('loss'):
            with tf.variable_scope('cycle_consistency_loss'):
                # l1 norm
                cycle_X_loss = tf.reduce_mean(tf.abs(cycle_X - self.imageX))
                cycle_Y_loss = tf.reduce_mean(tf.abs(cycle_Y - self.imageY))
                cycle_loss = Config.model.cycle_loss_weight_X * cycle_X_loss + Config.model.cycle_loss_weight_Y * cycle_Y_loss

            with tf.variable_scope('D_X_loss'):
                self.D_X_loss = (tf.reduce_mean(
                    tf.squared_difference(D_X_real, Config.model.real_label)) + tf.reduce_mean(
                    tf.square(D_X_fake_buffer))) / 2
                tf.summary.scalar('D_X_loss', self.D_X_loss)

            with tf.variable_scope('D_Y_loss'):
                self.D_Y_loss = (tf.reduce_mean(
                    tf.squared_difference(D_Y_real, Config.model.real_label)) + tf.reduce_mean(
                    tf.square(D_Y_fake_buffer))) / 2
                tf.summary.scalar('D_Y_loss', self.D_Y_loss)

            with tf.variable_scope('G_loss'):
                self.G_loss = tf.reduce_mean(tf.squared_difference(D_Y_fake, Config.model.real_label)) + cycle_loss
                tf.summary.scalar('G_loss', self.G_loss)

            with tf.variable_scope('F_loss'):
                self.F_loss = tf.reduce_mean(tf.squared_difference(D_X_fake, Config.model.real_label)) + cycle_loss
                tf.summary.scalar('F_loss', self.F_loss)

            self.loss = (self.G_loss + self.F_loss) / 2

    def _build_train_op(self, fake_X, fake_Y, imageX, imageY):

        class GenTrainOpsHook(tf.train.SessionRunHook):

            def __init__(self, train_ops, train_steps):

                if not isinstance(train_ops, (list, tuple)):
                    train_ops = [train_ops]

                self._train_op = train_ops
                self._train_steps = train_steps

            def before_run(self, run_context):
                for _ in range(self._train_steps):
                    run_context.session.run(self._train_op)

        class GenImagesBufferHook(tf.train.SessionRunHook):

            def __init__(self, fake_X, fake_Y, imageX, imageY):
                self._fake_X = fake_X
                self._fake_Y = fake_Y
                self._imageX = imageX
                self._imageY = imageY
                self._no_op = tf.no_op()
                self._fake_X_buffer = None
                self._fake_Y_buffer = None

            def before_run(self, run_context):
                fake_X_numpy, fake_Y_numpy, imageX_numpy, imageY_numpy = run_context.session.run(
                    [self._fake_X, self._fake_Y, self._imageX, self._imageY])
                if self._fake_X_buffer is None:
                    self._fake_X_buffer = fake_X_numpy
                    self._fake_Y_buffer = fake_Y_numpy
                else:
                    self._fake_X_buffer = np.concatenate([self._fake_X_buffer, fake_X_numpy])[
                                          :Config.model.gen_images_buffer_size]
                    self._fake_Y_buffer = np.concatenate([self._fake_Y_buffer, fake_Y_numpy])[
                                          :Config.model.gen_images_buffer_size]

                return tf.train.SessionRunArgs(self._no_op, {'fake_Y_buffer:0': self._fake_Y_buffer,
                                                             'fake_X_buffer:0': self._fake_X_buffer,
                                                             'F/Conv_21/Tanh:0': fake_X_numpy,
                                                             'G/Conv_21/Tanh:0': fake_Y_numpy,
                                                             'train/IteratorGetNext:0': imageX_numpy,
                                                             'train_1/IteratorGetNext:0': imageY_numpy})

        global_step = tf.train.get_or_create_global_step()
        learning_rate = (tf.where(tf.greater_equal(global_step, Config.train.start_decay_step),
                                  tf.train.polynomial_decay(Config.train.learning_rate,
                                                            global_step - Config.train.start_decay_step,
                                                            Config.train.max_steps - Config.train.start_decay_step,
                                                            0.0, power=1.0), Config.train.learning_rate))
        tf.summary.scalar('learning_rate', learning_rate)

        G_train_op = tf.train.AdamOptimizer(learning_rate, Config.train.beta1).minimize(self.G_loss,
                                                                                        var_list=tf.trainable_variables(
                                                                                            'G'))
        F_train_op = tf.train.AdamOptimizer(learning_rate, Config.train.beta1).minimize(self.F_loss,
                                                                                        var_list=tf.trainable_variables(
                                                                                            'F'))
        D_X_train_op = tf.train.AdamOptimizer(learning_rate, Config.train.beta1).minimize(self.D_X_loss,
                                                                                          var_list=tf.trainable_variables(
                                                                                              'D_X'))
        D_Y_train_op = tf.train.AdamOptimizer(learning_rate, Config.train.beta1).minimize(self.D_Y_loss,
                                                                                          var_list=tf.trainable_variables(
                                                                                              'D_Y'))
        self.training_hooks = []
        self.training_hooks.append(GenImagesBufferHook(fake_X, fake_Y, imageX, imageY))

        if Config.train.generator_train_step != 1:
            self.training_hooks.append(GenTrainOpsHook((G_train_op, F_train_op), Config.train.generator_train_step - 1))

        self.train_op = tf.group([G_train_op, F_train_op, D_X_train_op, D_Y_train_op, global_step.assign_add(1)])
