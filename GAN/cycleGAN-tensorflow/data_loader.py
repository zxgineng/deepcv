import tensorflow as tf
import argparse
import os
import sys

from utils import Config


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def _bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(value)))


def convert_to_example(bytes):
    """convert one sample to example"""
    data = {
        'image': _bytes_feature(bytes),
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        """parse tfrecord"""
        features = {
            'image': tf.FixedLenFeature([], tf.string),
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        image = tf.image.decode_jpeg(parsed_example['image'], 3)
        return image

    image = parse_tfrecord(serialized)
    image = (tf.image.convert_image_dtype(image, tf.float32) - 0.5) * 2
    image = tf.image.resize_images(image, [Config.model.image_size, Config.model.image_size])

    return image


def get_dataset_batch(data, buffer_size=1000, batch_size=64, scope="train"):
    """create input func"""

    class IteratorInitializerHook(tf.train.SessionRunHook):
        """Hook to initialise data iterator after Session is created."""

        def __init__(self):
            super(IteratorInitializerHook, self).__init__()
            self.iterator_initializer_func = None

        def after_create_session(self, session, coord):
            """Initialise the iterator after the session has been created."""
            self.iterator_initializer_func(session)

    iterator_initializer_hook = IteratorInitializerHook()

    def inputs():
        with tf.name_scope(scope):
            # Define placeholders
            input_placeholder = tf.placeholder(tf.string)
            # Build dataset iterator
            dataset = tf.data.TFRecordDataset(input_placeholder)
            dataset = dataset.map(preprocess)

            if scope == "train":
                dataset = dataset.repeat()
            else:
                dataset = dataset.repeat(1)
            dataset = dataset.shuffle(buffer_size)
            # dataset = dataset.batch(batch_size)
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            images = next_batch

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features, labels)
            return images

    # Return function and hook
    return inputs, iterator_initializer_hook


def get_both_batch(dataA, dataB, buffer_size=1000, batch_size=64, scope="train"):
    inputsA, iterator_initializer_hookA = get_dataset_batch(dataA, buffer_size, batch_size, scope)
    inputsB, iterator_initializer_hookB = get_dataset_batch(dataB, buffer_size, batch_size, scope)

    def inputs():
        imagesA = inputsA()
        imagesB = inputsB()

        return imagesA, imagesB

    return inputs, [iterator_initializer_hookA, iterator_initializer_hookB]


def get_tfrecord(mode):
    tfrecords = []
    path = os.path.join(Config.data.base_path, Config.data.data_path, Config.data.processed_path)
    files = os.listdir(path)
    for file in files:
        if mode in file and file.endswith('.tfrecord'):
            tfrecords.append(os.path.join(path, file))
    if len(tfrecords) == 0:
        raise RuntimeError('TFrecord not found.')

    return tfrecords


def read_image(path, file):
    file = os.path.join(path, file)
    with tf.gfile.FastGFile(file, 'rb') as f:
        bytes = f.read()

    return bytes


def create_tfrecord():
    for file in [Config.data.imageX_path, Config.data.imageY_path]:
        dataset_dir = os.path.join(Config.data.base_path, Config.data.data_path, file)
        output_dir = os.path.join(Config.data.base_path, Config.data.data_path, Config.data.processed_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filenames = sorted(os.listdir(dataset_dir))

        i = 0
        tf_filename = '%s/%s.tfrecord' % (output_dir, file)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            while i < len(filenames):
                sys.stdout.write('\r>> converting images %d/%d' % (i + 1, len(filenames)))
                sys.stdout.flush()
                filename = filenames[i]

                bytes = read_image(dataset_dir, filename)
                example = convert_to_example(bytes)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)

                i += 1

        print('\n%s image converted' % (file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/cycleGAN.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    create_tfrecord()
