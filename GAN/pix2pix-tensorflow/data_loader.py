import tensorflow as tf
import argparse
import os
import sys
import cv2

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


def convert_to_example(input, target,input_shape,target_shape):
    """convert one sample to example"""
    data = {
        'input': _bytes_feature(input),
        'target': _bytes_feature(target),
        'input_h': _int64_feature(input_shape[0]),
        'input_w': _int64_feature(input_shape[1]),
        'target_h': _int64_feature(target_shape[0]),
        'target_w': _int64_feature(target_shape[1])
    }
    features = tf.train.Features(feature=data)
    example = tf.train.Example(features=features)
    return example


def preprocess(serialized):
    def parse_tfrecord(serialized):
        """parse tfrecord"""
        features = {
            'input': tf.FixedLenFeature([], tf.string),
            'target': tf.FixedLenFeature([], tf.string),
            'input_h': tf.FixedLenFeature([], tf.int64),
            'input_w': tf.FixedLenFeature([], tf.int64),
            'target_h': tf.FixedLenFeature([], tf.int64),
            'target_w': tf.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        input = tf.decode_raw(parsed_example['input'],tf.uint8)
        target = tf.decode_raw(parsed_example['target'],tf.uint8)
        input = tf.reshape(input,tf.stack([parsed_example['input_h'],parsed_example['input_w'],3]))
        target= tf.reshape(target,tf.stack([parsed_example['target_h'],parsed_example['target_w'],3]))
        return input,target

    input, target = parse_tfrecord(serialized)
    input = (tf.image.convert_image_dtype(input,tf.float32) - 0.5) *2
    target = (tf.image.convert_image_dtype(target,tf.float32) - 0.5) * 2
    input = tf.image.resize_images(input,[Config.model.image_size]*2)
    target = tf.image.resize_images(target,[Config.model.image_size]*2)

    return input, target

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
                dataset = dataset.repeat(Config.train.epoch)
            else:
                dataset = dataset.repeat(1)
            dataset = dataset.shuffle(buffer_size)
            # dataset = dataset.batch(batch_size)
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            contour = next_batch[0]
            real_image = next_batch[1]

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features, labels)
            return contour,real_image

    # Return function and hook
    return inputs, iterator_initializer_hook


def get_tfrecord(mode):
    return os.path.join(Config.data.base_path,Config.data.data_path,'processed_'+mode,mode+'.tfrecord')

def read_image(path, file):
    file = os.path.join(path, file)
    image = cv2.imread(file)
    width = image.shape[1]
    if Config.data.image_direction == 'AtoB':
        input = image[:, :width // 2, :]
        target = image[:, width // 2:, :]
    elif Config.data.image_direction == 'BtoA':
        input = image[:, width // 2:, :]
        target = image[:, :width // 2, :]
    else:
        raise Exception('image_direction must be in AtoB or BtoA')
    return input.tostring(), target.tostring(),input.shape,target.shape


def create_tfrecord():
    for mode in ['train', 'val', 'test']:
        dataset_dir = os.path.join(Config.data.base_path, Config.data.data_path, mode)
        output_dir = os.path.join(Config.data.base_path, Config.data.data_path, 'processed_' + mode)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filenames = sorted(os.listdir(dataset_dir))

        i = 0
        tf_filename = '%s/%s.tfrecord' % (output_dir, mode)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            while i < len(filenames):

                sys.stdout.write('\r>> converting images %d/%d' % (i + 1, len(filenames)))
                sys.stdout.flush()
                filename = filenames[i]

                input_byte, target_byte,input_shape,target_shape = read_image(dataset_dir, filename)
                example = convert_to_example(input_byte, target_byte,input_shape,target_shape)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)

                i += 1

        print('\n%s image converted' % (mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/pix2pix.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    create_tfrecord()



