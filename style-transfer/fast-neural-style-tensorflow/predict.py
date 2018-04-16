import cv2
import numpy as np
import tensorflow as tf
import argparse
import os

from utils import Config
from model import Model


def predict(image,estimator):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"inputs": image},
        num_epochs=1,
        shuffle=False)

    result = next(estimator.predict(input_fn=predict_input_fn))
    return result

def show_image(result):
    image = result['prediction']
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image = image.astype(np.uint8)
    cv2.imwrite('images/result2.jpg',image)
    cv2.imshow('image',image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def _make_estimator():
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    run_config = tf.contrib.learn.RunConfig(model_dir=Config.train.model_dir)

    model = Model()
    return tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)

def preprocess(fname):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    image = image - np.array(Config.model.channels_mean)
    image = np.expand_dims(image,0).astype(np.float32)
    return image


def main(fnames):
    estimator = _make_estimator()
    for fname in fnames:
        image = preprocess(fname)
        result = predict(image, estimator)
        show_image(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/COCO.yml',
                        help='config file name')
    parser.add_argument('--file', type=str, nargs='+', help='file names,use space as separater')
    args = parser.parse_args()

    Config(args.config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    main(args.file)