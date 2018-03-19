import tensorflow as tf

class Reader():
    def __init__(self,args=None):
        self.image_paths = tf.placeholder(tf.string)
        self.args = args

    def next_batch(self):

        def parse(image_path):
            file_contents = tf.read_file(image_path)
            image = tf.image.decode_image(file_contents, channels=3)
            if self.args.random_crop:
                image = tf.random_crop(image, [self.args.image_size, self.args.image_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, self.args.image_size, self.args.image_size)
            if self.args.random_flip:
                image = tf.image.random_flip_left_right(image)
            image.set_shape((self.args.image_size, self.args.image_size, 3))
            image = tf.image.per_image_standardization(image)
            return image

        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        dataset = dataset.map(parse)
        dataset = dataset.repeat().batch(self.args.batch_size)
        self.iterator = dataset.make_initializable_iterator()
        generator = self.iterator.get_next()
        return generator


