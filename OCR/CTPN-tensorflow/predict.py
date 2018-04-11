import cv2
import numpy as np
import tensorflow as tf
import argparse
import os
import math

from utils import Config
from model import Model

def create_anchors(shape):
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
    return cy, cx, h, w



def predict(image,estimator):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"inputs": image},
        num_epochs=1,
        shuffle=False)

    result = next(estimator.predict(input_fn=predict_input_fn))
    return result

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
    size_min = min(image.shape[0:2])
    im_scale = 600 / size_min
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    image = image/255 - np.array([0.5,0.5,0.5])
    return image,im_scale

def anchors_select(scores_softmax,coords,anchors):
    """
    use threshold to filter prediction
    :param scores_softmax: 4D numpy, shape:[H,W,B,2]
    :param coords: 4D numpy, shape:[H,W,B,2]
    :param anchors: list of numpy, [cy, cx, h, w ]
    :return prob: 1D numpy
             coords: 2D numpy
    """
    # decode
    def decode_coords(coords,anchors):
        """decode"""
        cy, cx, h, w = anchors
        new_coords = np.zeros([coords.shape[0],coords.shape[1],coords.shape[2],4])
        cy = coords[:,:,:,0] * h + cy
        h = np.exp(coords[:,:,:,1]) * h
        # [ymin,xmin,ymax,xmax]
        new_coords[:,:,:,0] = cy - h/2
        new_coords[:,:,:,1] = cx - w/2
        new_coords[:,:,:,2] = cy + h/2
        new_coords[:,:,:,3] = cx + w/2
        return new_coords

    positive_prob = scores_softmax[:,:,:,1]
    coords = decode_coords(coords,anchors)
    idxes = np.where(positive_prob > Config.predict.prob_threshold)
    prob = positive_prob[idxes]
    coords = coords[idxes]
    return prob,coords

def bboxes_clip(bboxes,shape):
    """crop the boxes outside of the image"""
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    # set limitation of ymin,xmin,ymax,xmax
    bboxes[0] = np.maximum(bboxes[0], 0.0)
    bboxes[1] = np.maximum(bboxes[1], 0.0)
    bboxes[2] = np.minimum(bboxes[2], shape[0])
    bboxes[3] = np.minimum(bboxes[3], shape[1])
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_nms(prob,bboxes,nms_threshold=0.45):
    """nms"""
    keep_bboxes = np.ones(prob.shape, dtype=np.bool)
    for i in range(prob.size - 1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i + 1):])
            # filter iou<threshold
            keep_overlap = overlap < nms_threshold
            keep_bboxes[(i + 1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return prob[idxes], bboxes[idxes]


def bboxes_jaccard(bboxes1, bboxes2):
    """cal iou"""
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard

def show_image(image,scale,result):
    scores_softmax = result['scores_softmax']
    coords = result['vcoords_logits']
    side = result['side_logits']
    anchors = create_anchors(image.shape)
    cy, cx, h, w = anchors
    coords = coords.reshape([cy.shape[0], cy.shape[1], len(h), 2])
    scores_softmax = scores_softmax.reshape([cy.shape[0], cy.shape[1], len(h), 2])
    side = side.reshape([cy.shape[0], cy.shape[1], len(h)])
    prob,coords = anchors_select(scores_softmax,coords,anchors)
    bboxes = bboxes_clip(coords,image.shape)
    prob,bboxes = bboxes_nms(prob,bboxes,Config.predict.nms_threshold)


def draw(image,prob,bboxes,scale):
    pass



def main(fnames):
    estimator = _make_estimator()
    for fname in fnames:
        image, scale = preprocess(fname)
        result = predict(image, estimator)
        show_image(image,scale,result)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/MLT.yml',
                        help='config file name')
    parser.add_argument('--file_names', type=str, nargs='+', help='file names,use space as separater')
    args = parser.parse_args()

    Config(args.config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    main(args.file_names)
