import tensorflow as tf
import xml.etree.ElementTree as ET
import os
import sys
from config import VOC_LABELS
# 将voc2007转换为tfrecord

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
SAMPLES_PER_FILES = 6000


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
    解析xml
    :param directory: 文档目录
    :param name: 文件名
    :return image_data: image bytes
             bboxes: list of list
             labels: list

    """
    #读取文件
    filename = os.path.join(directory,DIRECTORY_IMAGES ,name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    # 读取标注
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # 搜索标签
    bboxes = []
    labels = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label]))
        bbox = obj.find('bndbox')
        bboxes.append([float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ])
    return image_data, bboxes, labels


def convert_to_tf_example(image_data,labels,bboxes):
    """
    将单个sample转换为example
    """
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

def create_tfrecord(dataset_dir, output_dir):
    """
    创建tfrecord
    :param dataset_dir: 读取目录
    :param output_dir: 输出目录
    :return:
    """
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = '%s/voc2007_%03d.tfrecord' % (output_dir,fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> 转换图片中 %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                image_data, bboxes, labels = parse_xml(dataset_dir,img_name)
                example = convert_to_tf_example(image_data, labels, bboxes)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)
                i += 1
                j += 1
            fidx += 1

    print('\n转换完成!')


if __name__ == '__main__':
    # parse_xml('D:/darknet/scripts/VOCdevkit/VOC2007','000005')
    create_tfrecord('D:/darknet/scripts/VOCdevkit/VOC2007','D:/dataset/test')
