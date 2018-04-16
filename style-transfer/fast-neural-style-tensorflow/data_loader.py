import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf

from utils import Config


def _int64_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def _float32_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(value)))


def convert_to_example(bytes, shape):
    """convert one sample to example"""
    data = {
        'image': _bytes_feature(bytes),
        'height': _int64_feature(shape[0]),
        'width': _int64_feature(shape[1])
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def create_tfrecord():
    """create tfrecord"""
    dataset_dir = os.path.join(Config.data.base_path, Config.data.raw_data_path)
    output_dir = os.path.join(Config.data.base_path, Config.data.processed_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filenames = sorted(os.listdir(dataset_dir))
    i = 0
    fidx = 0
    error = 0

    while i < len(filenames):
        tf_filename = '%s/%02d.tfrecord' % (output_dir, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < Config.data.samples_per_tfrecord:
                sys.stdout.write('\r>> converting images %d/%d' % (i + 1, len(filenames)))
                sys.stdout.flush()
                filename = filenames[i]
                try:
                    bytes, shape = read_image(dataset_dir, filename)
                except TypeError:
                    print('  skip ', filename)
                    error += 1
                    i += 1
                    continue
                example = convert_to_example(bytes, shape)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)
                j += 1
                i += 1
            fidx += 1
    print('\nskip %d invalid image' % (error))


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
                dataset = dataset.repeat(None)  # Infinite iterations
            else:
                dataset = dataset.repeat(1)  # 1 Epoch
            dataset = dataset.shuffle(buffer_size=buffer_size)
            # dataset = dataset.batch(batch_size)
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            inputs = next_batch

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features)
            return {'inputs': inputs}

    # Return function and hook
    return inputs, iterator_initializer_hook


def get_tfrecord(mode, shuffle=True):
    if mode == 'train':
        path = os.path.join(Config.data.base_path, Config.data.processed_path)
        tfr_data = np.array(os.listdir(path))
        if shuffle:
            print("shuffle dataset ...")
            index = np.random.permutation(len(tfr_data))
            tfr_data = tfr_data[index]
        tfr_data = [os.path.join(path, data) for data in tfr_data]
    else:
        raise ValueError('Invalid mode!')
    return tfr_data


def preprocess(serialized):
    def parse_tfrecord(serialized):
        """parse tfrecord"""
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        image = tf.decode_raw(parsed_example['image'], tf.uint8)
        width = parsed_example['width']
        height = parsed_example['height']
        image = tf.reshape(image, tf.stack([height, width, 3]))
        image = tf.image.resize_images(image, [Config.model.image_size, Config.model.image_size])
        return image

    image = parse_tfrecord(serialized)
    image = tf.to_float(image)
    image = image - Config.model.channels_mean
    return image


def read_image(dir, name):
    path = os.path.join(dir, name)
    image = cv2.imread(path)
    if image is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = image.shape
    return image.tostring(), shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/COCO.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    create_tfrecord()
