import tensorflow as tf
import xml.etree.ElementTree as ET
import argparse
import os
import sys
import numpy as np

from utils import Config


def create_default_boxes():
    """
    cal coords of all default boxes and output as a list
    :return default_boxes_loc: list of list of numpy,
    """
    default_boxes_loc = []
    # cal coords of all default boxes of every feat layer
    for i, feat_shape in enumerate(Config.model.feat_shape):
        num_box = len(Config.model.anchor_sizes[i]) + len(Config.model.anchor_ratios[i])

        cy, cx = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        # set center in each pix as centers, and relative position of image,  range(0,1)
        cy = (cy + 0.5) * Config.model.anchor_steps[i] / Config.model.image_shape[0]
        cx = (cx + 0.5) * Config.model.anchor_steps[i] / Config.model.image_shape[1]
        # cy,cx --shape[H,W,1]
        cy = np.expand_dims(cy, -1).astype('float32')
        cx = np.expand_dims(cx, -1).astype('float32')
        w = np.zeros(num_box, dtype='float32')
        h = np.zeros(num_box, dtype='float32')
        # use anchor_sizes, anchor_ratios and original image size to get relative H,W ,  shape:[B,]
        h[0] = Config.model.anchor_sizes[i][0] / Config.model.image_shape[0]
        w[0] = Config.model.anchor_sizes[i][0] / Config.model.image_shape[1]
        h[1] = np.sqrt(Config.model.anchor_sizes[i][0] * Config.model.anchor_sizes[i][1]) / Config.model.image_shape[0]
        w[1] = np.sqrt(Config.model.anchor_sizes[i][0] * Config.model.anchor_sizes[i][1]) / Config.model.image_shape[1]
        for j, ratio in enumerate(Config.model.anchor_ratios[i]):
            h[j + 2] = h[0] / np.sqrt(ratio)
            w[j + 2] = w[0] * np.sqrt(ratio)
        default_boxes_loc.append([cy, cx, h, w])
    return default_boxes_loc


def convert_image_to_target(labels, bboxes, default_boxes, threshold=0.5, scaling=(0.1, 0.1, 0.2, 0.2)):
    """
    encode one image into target
    :param labels: 1D Tensor(int64) labels in the image
    :param bboxes: 2D Tensor(float32), --shape[num_label,4], the relative coords of bboxes of each label(ymin,xmin,ymax,xmax)
    :param default_boxes: list of list of numpy
    :param threshold: threshold of positive iou
    :param scaling: scaling of encoding
    :return target_labels: list of tensor, class target of all default boxes,shape:[38,38,b]，[19,19,b],...,[1,1,b]
             target_locs: list of tensor, locs offset target of all default boxes,shape:[38,38,b,4]，[19,19,b,4],...,[1,1,b,4]
    """
    target_labels_list = []
    target_locs_list = []
    # cal default box respectively
    for default_box in default_boxes:
        # cal four corners and crop bbox area outside of the image
        cy, cx, h, w = default_box
        ymin = tf.maximum(cy - h / 2.0, 0.0)
        xmin = tf.maximum(cx - w / 2.0, 0.0)
        ymax = tf.minimum(cy + h / 2.0, 1.0)
        xmax = tf.minimum(cx + w / 2.0, 1.0)
        shape = (cy.shape[0], cy.shape[1], h.shape[0])
        default_area = (xmax - xmin) * (ymax - ymin)
        # save last labels,iou,ymin,etc.initialize 0
        feat_labels = tf.zeros(shape, tf.int64)
        feat_iou = tf.zeros(shape)
        feat_ymin = tf.zeros(shape)
        feat_xmin = tf.zeros(shape)
        # initialize 1,prevent log0
        feat_ymax = tf.ones(shape)
        feat_xmax = tf.ones(shape)

        def iou_with_bbox(bbox):
            """
            cal iou
            :param box: single bbox (ymin,xmin,ymax,xmax)
            :return: iou tensor, --shape[H,W,B]
            """
            # cal inter
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            inter_area = h * w
            # cal union
            union_area = default_area - inter_area + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            iou = tf.div(inter_area, union_area)
            return iou

        def condition(i, feat_labels, feat_iou, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """loop condition: all targets in labels"""
            r = tf.less(i, tf.shape(labels))
            return r[0]

        def body(i, feat_labels, feat_iou, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """loop body: update value"""
            label = labels[i]
            box = bboxes[i]
            iou = iou_with_bbox(box)
            # iou>0.5 & > original iou
            mask = tf.logical_and(tf.greater(iou, threshold), tf.greater(iou, feat_iou))
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, tf.float32)
            feat_labels = tf.where(mask, imask * label, feat_labels)
            feat_iou = tf.where(mask, iou, feat_iou)
            feat_ymin = tf.where(mask, fmask * box[0], feat_ymin)
            feat_xmin = tf.where(mask, fmask * box[1], feat_xmin)
            feat_ymax = tf.where(mask, fmask * box[2], feat_ymax)
            feat_xmax = tf.where(mask, fmask * box[3], feat_xmax)

            return [i + 1, feat_labels, feat_iou,
                    feat_ymin, feat_xmin, feat_ymax, feat_xmax]

        # loop
        i = 0
        [i, feat_labels, feat_iou, feat_ymin, feat_xmin, feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                                                                               [i, feat_labels,
                                                                                                feat_iou, feat_ymin,
                                                                                                feat_xmin, feat_ymax,
                                                                                                feat_xmax])

        # encode locs and calculate offset
        cy_offset = ((feat_ymax + feat_ymin) / 2 - cy) / h / scaling[0]
        cx_offset = ((feat_xmax + feat_xmin) / 2 - cx) / w / scaling[1]
        h_offset = tf.log((feat_ymax - feat_ymin) / h) / scaling[2]
        w_offset = tf.log((feat_xmax - feat_xmin) / w) / scaling[3]
        encode_locs = tf.stack([cx_offset, cy_offset, w_offset, h_offset], axis=-1)

        target_labels_list.append(feat_labels)
        target_locs_list.append(encode_locs)
    return target_labels_list, target_locs_list


def parse(serialized):
    """parse tfrecord"""
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'labels': tf.VarLenFeature(tf.int64),
        'ymin': tf.VarLenFeature(tf.float32),
        'xmin': tf.VarLenFeature(tf.float32),
        'ymax': tf.VarLenFeature(tf.float32),
        'xmax': tf.VarLenFeature(tf.float32),
    }
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    ymin = tf.sparse_tensor_to_dense(parsed_example['ymin'])
    xmin = tf.sparse_tensor_to_dense(parsed_example['xmin'])
    ymax = tf.sparse_tensor_to_dense(parsed_example['ymax'])
    xmax = tf.sparse_tensor_to_dense(parsed_example['xmax'])
    image = tf.image.decode_jpeg(parsed_example['image'], 3)
    labels = tf.sparse_tensor_to_dense(parsed_example['labels'])
    bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
    return image, bboxes, labels


def preprocess(serialized):
    """preprocess tfrecord"""
    image, bboxes, labels = parse(serialized)
    image = tf.to_float(image)
    image = tf.subtract(image, tf.constant([123.0, 117.0, 104.0], tf.float32))
    image = tf.image.resize_images(image, [300, 300], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    default_boxes_loc = create_default_boxes()
    target_labels, target_locs = convert_image_to_target(labels, bboxes, default_boxes_loc)
    flat_all = []
    for n in [image, target_labels, target_locs]:
        if isinstance(n, list):
            flat_all = flat_all + n
        else:
            flat_all.append(n)
    return flat_all


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
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_batch = iterator.get_next()
            next_X = next_batch[0]
            next_y = next_batch[1:]

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={input_placeholder: data})

            # Return batched (features, labels)
            return {'inputs': next_X}, next_y

    # Return function and hook
    return inputs, iterator_initializer_hook


def get_tfrecord(shuffle=True):
    path = os.path.join(Config.data.base_path, Config.data.processed_path)
    tfr_data = np.array(os.listdir(path))
    if shuffle:
        print("shuffle dataset ...")
        index = np.random.permutation(len(tfr_data))
        tfr_data = tfr_data[index]
    tfr_data = [os.path.join(path,data) for data in tfr_data]
    return tfr_data


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


def parse_xml(directory, name):
    """
    parse xml
    :return image_data: image bytes
             bboxes: list of list
             labels: list
    """
    # load file
    filename = os.path.join(directory, Config.data.directory_images, name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    # load annotations
    filename = os.path.join(directory, Config.data.directory_annotations, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # search labels
    bboxes = []
    labels = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(Config.data.voc_labels[label]))
        bbox = obj.find('bndbox')
        bboxes.append([float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ])
    return image_data, bboxes, labels


def convert_to_tf_example(image_data, labels, bboxes):
    """convert one sample into example"""
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
    data = {
        'image': _bytes_feature(image_data),
        'ymin': _float32_feature(ymin),
        'xmin': _float32_feature(xmin),
        'ymax': _float32_feature(ymax),
        'xmax': _float32_feature(xmax),
        'labels': _int64_feature(labels),
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
    path = os.path.join(dataset_dir, Config.data.directory_annotations)
    filenames = sorted(os.listdir(path))
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = '%s/voc2007_%03d.tfrecord' % (output_dir, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < Config.data.samples_per_tfrecord:
                sys.stdout.write('\r>> converting images %d/%d' % (i + 1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                image_data, bboxes, labels = parse_xml(dataset_dir, img_name)
                example = convert_to_tf_example(image_data, labels, bboxes)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)
                i += 1
                j += 1
            fidx += 1

    print('\nconvertion completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/voc2007.yml',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    create_tfrecord()
