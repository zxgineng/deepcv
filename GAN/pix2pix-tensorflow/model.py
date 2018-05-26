import tensorflow as tf

from utils import Config
import architecture


class Model:

    def __init__(self):
        self._build_loss_fn()
        self._build_optimizer()

    def generator_fn(self, contours, mode):
        graph = architecture.Generator_Graph(mode)
        outputs = graph.build(contours)
        images = tf.image.convert_image_dtype(outputs/2 + 0.5,tf.uint8,name='generated_images')
        tf.summary.image('generated_images',images,max_outputs=4)
        if mode != tf.estimator.ModeKeys.PREDICT:
            return outputs
        else:
            contours = tf.image.convert_image_dtype(contours/2 + 0.5,tf.uint8,name='generated_images')
            return tf.concat([contours,images],2)


    def discriminator_fn(self, images, contours):
        graph = architecture.Discriminator_Graph()
        logits = graph.build(images, contours)
        return logits

    def _build_loss_fn(self):
        def generator_loss_fn(gan_model, add_summaries=True):
            gen_loss_gan = tf.reduce_mean(-tf.log(tf.nn.sigmoid(gan_model.discriminator_gen_outputs) + 1e-12))
            gen_loss_l1 = tf.reduce_mean(tf.abs(gan_model.generated_data - gan_model.real_data))
            G_loss = tf.add(gen_loss_gan, Config.model.l1_weight * gen_loss_l1, 'G_loss')
            tf.summary.scalar('generator_loss',G_loss)
            return G_loss

        def discriminator_loss_fn(gan_model, add_summaries=True):
            dis_loss_real = tf.reduce_mean(-tf.log(tf.nn.sigmoid(gan_model.discriminator_real_outputs) + 1e-12),
                                           name='D_real_loss')
            dis_loss_gen = tf.reduce_mean(-tf.log(1 - tf.nn.sigmoid(gan_model.discriminator_gen_outputs) + 1e-12),
                                          name='D_gen_loss')
            D_loss = tf.add(dis_loss_real, dis_loss_gen, 'D_loss')
            tf.summary.scalar('discriminator_loss',D_loss)
            return D_loss

        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn

    def _build_optimizer(self):
        self.generator_optimizer = tf.train.AdamOptimizer(Config.train.learning_rate, Config.train.beta1)
        self.discriminator_optimizer = tf.train.AdamOptimizer(Config.train.learning_rate, Config.train.beta1)

