import model
import reader
import trainer
import predictor
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', 'test', 'train or test')
tf.flags.DEFINE_string('dataset_file', '', 'dataset file')
tf.flags.DEFINE_integer('batch_size',32,'batch size')
tf.flags.DEFINE_integer('epoch',20,'epoch')
tf.flags.DEFINE_string('test_file','demo/dog.jpg','test file')
tf.flags.DEFINE_string('ckpt_file','checkpoints/ssd_300_vgg.ckpt','ckpt file')

def main(_):
    if FLAGS.mode == 'train':
        generator = reader.Generator([FLAGS.dataset_file])
        g = generator.batch_generator(FLAGS.batch_size,FLAGS.epoch)
        model_c = model.SSD
        with tf.Session() as sess:
            trainer.Trainer(sess,model_c,g[0],g[1:]).train()

    elif FLAGS.mode == 'test':
        model_c = model.SSD
        with tf.Session() as sess:
            p = predictor.Predictor(sess, model_c)
            p.load(FLAGS.ckpt_file)
            p.predict(FLAGS.test_file)
    else:
        raise ValueError("choose 'train' or 'test'")


if __name__ == '__main__':
    tf.app.run()