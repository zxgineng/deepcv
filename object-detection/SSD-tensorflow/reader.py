from util import *


def parse(serialized):
    features = {
        'image': tf.FixedLenFeature([],tf.string),
        'labels': tf.VarLenFeature(tf.int64),
        'ymin': tf.VarLenFeature(tf.float32),
        'xmin': tf.VarLenFeature(tf.float32),
        'ymax': tf.VarLenFeature(tf.float32),
        'xmax': tf.VarLenFeature(tf.float32),
    }
    parsed_example = tf.parse_single_example(serialized=serialized,features=features)
    ymin = tf.sparse_tensor_to_dense(parsed_example['ymin'])
    xmin = tf.sparse_tensor_to_dense(parsed_example['xmin'])
    ymax = tf.sparse_tensor_to_dense(parsed_example['ymax'])
    xmax = tf.sparse_tensor_to_dense(parsed_example['xmax'])
    image = tf.image.decode_jpeg(parsed_example['image'],3)
    labels = tf.sparse_tensor_to_dense(parsed_example['labels'])
    bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
    return image,bboxes,labels

def preprocess(serialized):
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

class Generator():
    def __init__(self,train_file,test_file=None):
        self.train_file = train_file
        self.test_file = test_file

    def batch_generator(self, batch_size, epoch):
        dataset = tf.data.TFRecordDataset(self.train_file)
        dataset = dataset.map(preprocess)
        dataset = dataset.shuffle(1000).repeat(epoch).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        generator = iterator.get_next()
        return generator














