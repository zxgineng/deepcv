import tensorflow as tf
import argparse
import os
import sys
import cv2
import numpy as np
import math

from utils import Config


def _int64_feature(value):
    if not isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float32_feature(value):
    if not isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def parse_txt(dir, fname, im_scale):
    """parse txt"""
    filename = os.path.join(dir, Config.data.raw_txt_path, fname)
    coords = []
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            split_line = line.strip().split(',')
            if split_line[-1] == '###':
                continue
            split_line[:8] = [float(n) for n in split_line[:8]]
            # resize coords
            coord = np.array(split_line[:8]).astype(float) * im_scale
            coords.append(coord)
    return coords


def process_anchors_coords(shape, coords):
    num_anchors = len(Config.model.anchor_height)
    num_height = math.floor(shape[0] / Config.model.anchor_width)
    num_width = math.floor(shape[1] / Config.model.anchor_width)
    gridy, gridx = np.mgrid[0:num_height, 0:num_width]
    cy = (gridy + 0.5) * Config.model.anchor_width
    cx = (gridx + 0.5) * Config.model.anchor_width
    cy = np.expand_dims(cy, -1)
    cx = np.expand_dims(cx, -1)
    w = np.ones(num_anchors, dtype='float32') * Config.model.anchor_width
    h = np.array(Config.model.anchor_height, dtype='float32')

    def divide_coords():
        width_slice = gridx[0] * Config.model.anchor_width
        divided_coords = []

        for coord in coords:
            # cv2.polylines(image_data,np.array([[[coord[0],coord[1]],[coord[2],coord[3]],[coord[4],coord[5]],[coord[6],coord[7]]]]).astype(int),1, 255)
            up_slope = (coord[1] - coord[3]) / (coord[2] - coord[0])
            bottom_slope = (coord[7] - coord[5]) / (coord[4] - coord[6])
            if coord[0] > coord[6]:
                coord[1] = coord[1] + (coord[0] - coord[6]) * up_slope
            else:
                coord[7] = coord[7] + (coord[6] - coord[0]) * bottom_slope
            left_side = min(coord[0], coord[6])
            right_side = max(coord[2], coord[4])
            ymin = coord[1]
            xmin = left_side
            ymax = coord[7]
            for x_slice in width_slice:

                if x_slice > left_side:
                    if x_slice < right_side:
                        ymin_temp = min(ymin, ymin - up_slope * (x_slice - xmin))
                        ymax_temp = max(ymax, ymax - bottom_slope * (x_slice - xmin))
                        xmax = x_slice
                        divided_coords.append([ymin_temp, xmin, ymax_temp, xmax])
                        # cv2.rectangle(Config.image, (int(xmin), int(ymin_temp)), (int(xmax), int(ymax_temp)), (255,0, 0), 1)
                        ymin = ymin - up_slope * (x_slice - xmin)
                        ymax = ymax - bottom_slope * (x_slice - xmin)
                        xmin = x_slice
                    else:
                        divided_coords.append([min(ymin, coord[3]), xmin, max(ymax, coord[5]), right_side])
                        # cv2.rectangle(Config.image, (int(xmin), int(min(ymin,coord[3]))), (int(right_side), int(max(ymax,coord[5]))), (255, 0, 0), 1)
                        break
        return divided_coords

    coords = divide_coords()
    return [cy, cx, h, w], coords


def transfer_targets(anchors, coords):
    cy, cx, h, w = anchors
    # shape:[h,w,10]
    ymin = cy - h / 2.0
    xmin = cx - w / 2.0
    ymax = cy + h / 2.0
    xmax = cx + w / 2.0
    shape = (cy.shape[0], cy.shape[1], h.shape[0])
    default_area = (xmax - xmin) * (ymax - ymin)
    # feat_labels = np.zeros(shape)
    feat_iou = np.zeros(shape)
    feat_ymin = np.zeros(shape)
    feat_xmin = np.zeros(shape)
    feat_xmax = np.zeros(shape)
    # in case of log0
    feat_ymax = np.ones(shape)
    x_side = np.ones(shape) * (-Config.model.anchor_width)

    def cal_iou(bbox):
        """
        cal iou
        :param box: single bbox (ymin,xmin,ymax,xmax)
        :return: iou, --shape[H,W,B]
        """
        # cal inter
        int_ymin = np.maximum(ymin, bbox[0])
        int_xmin = np.maximum(xmin, bbox[1])
        int_ymax = np.minimum(ymax, bbox[2])
        int_xmax = np.minimum(xmax, bbox[3])
        h = np.maximum(int_ymax - int_ymin, 0.)
        w = np.maximum(int_xmax - int_xmin, 0.)
        inter_area = h * w
        # cal union
        union_area = default_area - inter_area + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        iou = inter_area / union_area
        return iou

    for coord in coords:
        iou = cal_iou(coord)
        max_mask = np.logical_and(iou == np.max(iou), iou > feat_iou)
        mask = np.logical_and(iou > Config.model.threshold, iou > feat_iou)
        mask = np.logical_or(max_mask, mask)
        feat_iou = np.where(mask, iou, feat_iou)
        feat_ymin = np.where(mask, coord[0], feat_ymin)
        feat_xmin = np.where(mask, coord[1], feat_xmin)
        feat_ymax = np.where(mask, coord[2], feat_ymax)
        feat_xmax = np.where(mask, coord[3], feat_xmax)
    left_side_mask = xmin < feat_xmin
    x_side = np.where(left_side_mask, feat_xmin - cx, x_side)
    right_side_mask = np.logical_and((xmax > feat_xmax), (feat_xmax != 0))
    x_side = np.where(right_side_mask, feat_xmax - cx, x_side)

    # print(np.sum(x_side > -Config.model.anchor_width))
    cy_offset = ((feat_ymax + feat_ymin) / 2 - cy) / h
    h_offset = np.log((feat_ymax - feat_ymin) / h)
    side_offset = x_side / Config.model.anchor_width
    coords = np.stack([cy_offset, h_offset], axis=-1)
    coords = coords.astype(np.float32)
    labels = (feat_iou > 0).astype(int)

    # a = np.stack([ymin,xmin,ymax,xmax],-1)
    # # a = np.stack([feat_ymin, feat_xmin, feat_ymax, feat_xmax], -1)
    # # rec = np.maximum(a[labels==1],0.0)
    # rec = a[x_side > -Config.model.anchor_width]
    #
    # for r in rec:
    #     cv2.rectangle(Config.image,(int(r[1]),int(r[0])),(int(r[3]),int(r[2])),(0, 255, 0), 1)
    #
    # cv2.imshow('image', Config.image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return labels, coords, side_offset


def parse_data(dir, fname):
    """parse image and txt"""
    img_name = fname[3:-3] + 'jpg'
    filename = os.path.join(dir, Config.data.raw_image_path, img_name)
    if os.path.exists(filename):
        image_data = cv2.imread(filename)
    else:
        filename = filename[:-3] + 'png'
        image_data = cv2.imread(filename)
    if image_data is None:
        return
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    origin_shape = list(image_data.shape)
    size_min = min(origin_shape[0:2])
    im_scale = 600 / size_min
    # resize image by setting its short side to 600 for training,
    # while keeping its original aspect ratio
    image_data = cv2.resize(image_data, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    # Config.image = image_data
    resize_shape = list(image_data.shape)
    image_byte = image_data.tostring()
    coords = parse_txt(dir, fname, im_scale)
    if len(coords) == 0 :
        return
    anchors, coords = process_anchors_coords(resize_shape, coords)
    labels, coords, side_offset = transfer_targets(anchors, coords)
    return image_byte, resize_shape, coords, labels, side_offset


def convert_to_example(image_byte, image_shape, coords, labels, side_offset):
    """convert one sample to example"""
    coords = coords.astype(np.float32)
    side_offset = side_offset.astype(np.float32)
    labels = labels.astype(np.int64)
    data = {
        'image': _bytes_feature(image_byte),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'tar_height': _int64_feature(coords.shape[0]),
        'tar_width': _int64_feature(coords.shape[1]),
        'coords': _bytes_feature(coords.tostring()),
        'labels': _bytes_feature(labels.tostring()),
        'side': _bytes_feature(side_offset.tostring())
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
    path = os.path.join(dataset_dir, Config.data.raw_txt_path)
    filenames = sorted(os.listdir(path))
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
                    image_byte, resize_shape, coords, labels, side_offset = parse_data(dataset_dir, filename)
                except TypeError:
                    print('  skip ', filename)
                    error += 1
                    i += 1
                    continue
                example = convert_to_example(image_byte, resize_shape, coords, labels, side_offset)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)
                j += 1
                i += 1
            fidx += 1
    print('\nskip %d invalid image' % (error))


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

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            inputs = next_batch[0]
            targets = next_batch[1:]

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features, labels)
            return {'inputs': inputs}, targets

    # Return function and hook
    return inputs, iterator_initializer_hook


def preprocess(serialized):
    def parse_tfrecord(serialized):
        """parse tfrecord"""
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'tar_height': tf.FixedLenFeature([], tf.int64),
            'tar_width': tf.FixedLenFeature([], tf.int64),
            'coords': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
            'side': tf.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        image = tf.decode_raw(parsed_example['image'], tf.uint8)
        width = parsed_example['width']
        height = parsed_example['height']
        image = tf.reshape(image, tf.stack([height, width, 3]))
        coords = tf.decode_raw(parsed_example['coords'], tf.float32)
        coords = tf.reshape(coords, [-1,2])
        labels = tf.decode_raw(parsed_example['labels'], tf.int64)
        labels = tf.reshape(labels, [-1])
        side = tf.decode_raw(parsed_example['side'], tf.float32)
        side = tf.reshape(side, [-1])
        return image, coords, labels, side

    image, coords, labels, side = parse_tfrecord(serialized)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, [0.5, 0.5, 0.5])
    return image,coords,labels,side


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/MLT.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    create_tfrecord()



