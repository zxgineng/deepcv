import cv2
import numpy as np
import tensorflow as tf
import argparse
import os

import visualization
from data_loader import create_default_boxes
from utils import Config
from model import Model


def ssd_bboxes_select(predictions_net,
                      localizations_net,
                      anchors_net,
                      select_threshold=0.5,
                      decode=True):
    """
    filter all layer
    :param predictions_net: list of numpy, individual shape:[1,H,W,B,21]
    :param localizations_net: list of numpy, individual shape:[1,H,W,B,4]
    :param anchors_net: list of list of numpy,
    :param decode: decode or not
    :return classes: 1D numpy, class of all layer
             scores: 1D numpy, prediction of all layer
             bboxes: 2D numpy, locs of all layer
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = ssd_bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            select_threshold, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)

    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes


def locs_decode(pred_locs_offset, anchors_layer, scaling=(0.1, 0.1, 0.2, 0.2)):
    """
    decode locs offset of one layer into [ymin,xmin,ymax,xmax]
    :param pred_locs_offset: numpy, --shape[1,H,W,B,4], locs offset of one layer
    :param anchors_layer: list of numpy, [y,x,h,w], anchors_layer of one layer
    :return bboxes: numpy, --shape[1,H,W,B,4], relative locs of one layer
    """
    # reshape:[H*W,B,4]
    shape = pred_locs_offset.shape
    pred_locs_offset = np.reshape(pred_locs_offset, (-1, shape[-2], shape[-1]))
    yref, xref, href, wref = anchors_layer
    # reshape:[H*W,1]
    xref = np.reshape(xref, [-1, 1])
    yref = np.reshape(yref, [-1, 1])
    # decode
    cx = pred_locs_offset[:, :, 0] * wref * scaling[0] + xref
    cy = pred_locs_offset[:, :, 1] * href * scaling[1] + yref
    w = wref * np.exp(pred_locs_offset[:, :, 2] * scaling[2])
    h = href * np.exp(pred_locs_offset[:, :, 3] * scaling[3])
    # bboxes: [ymin, xmin, xmax, ymax]
    locs = np.zeros_like(pred_locs_offset)
    locs[:, :, 0] = cy - h / 2.
    locs[:, :, 1] = cx - w / 2.
    locs[:, :, 2] = cy + h / 2.
    locs[:, :, 3] = cx + w / 2.
    locs = np.reshape(locs, shape)
    return locs


def ssd_bboxes_select_layer(predictions_layer,
                            localizations_layer,
                            anchors_layer,
                            select_threshold=0.5,
                            decode=True):
    """
    use threshold to filter prediction in single layer
    :param predictions_layer: numpy, --shape[1,H,W,B,21]
    :param localizations_layer: numpy, --shape[1,H,W,B,4]
    :param anchors_layer: list of numpy, [y,x,h,w]
    :param decode: decode or not
    :return classes: 1D numpy, clase
             scores: 1D numpy, prob
             bboxes: 2D numpy, locs
    """
    # decode
    if decode:
        localizations_layer = locs_decode(localizations_layer, anchors_layer)

    p_shape = predictions_layer.shape
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    # reshape:[1,H*W*B,21]
    predictions_layer = np.reshape(predictions_layer,
                                   (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    # reshape:[1,H*W*B,21]
    localizations_layer = np.reshape(localizations_layer,
                                     (batch_size, -1, l_shape[-1]))

    # filter predictions> select_threshold
    sub_predictions = predictions_layer[:, :, 1:]
    idxes = np.where(sub_predictions > select_threshold)
    classes = idxes[-1] + 1
    scores = sub_predictions[idxes]
    bboxes = localizations_layer[idxes[:-1]]

    return classes, scores, bboxes


def bboxes_sort(classes, scores, bboxes, top_k=400):
    """filter top k bboxes, sorted by prediction descending"""
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """nms"""
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size - 1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i + 1):])
            # filter iou>0.45 with bboxes of same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i + 1):] != classes[i])
            keep_bboxes[(i + 1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


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


def bboxes_clip(bboxes):
    """
    crop the boxes outside of the image
    :param bboxes: 2D numpy, shape--[N,4]
    :return bboxes: 2D numpy, shape--[N,4]
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    # set limitation of ymin,xmin,ymax,xmax
    bboxes[0] = np.maximum(bboxes[0], 0.0)
    bboxes[1] = np.maximum(bboxes[1], 0.0)
    bboxes[2] = np.minimum(bboxes[2], 1.0)
    bboxes[3] = np.minimum(bboxes[3], 1.0)
    bboxes = np.transpose(bboxes)
    return bboxes


def predict(images):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"inputs": images},
        num_epochs=1,
        shuffle=False)

    estimator = _make_estimator()
    result = estimator.predict(input_fn=predict_input_fn)
    return result


def show_image(fname, pred):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    predictions = [np.expand_dims(n, 0) for n in pred.values()]
    rpredictions = predictions[:6]
    rlocalisations = predictions[6:]
    ssd_anchors = create_default_boxes()
    rclasses, rscores, rbboxes = ssd_bboxes_select(rpredictions, rlocalisations, ssd_anchors, select_threshold=Config.predict.select_threshold,
                                                   decode=True)
    rbboxes = bboxes_clip(rbboxes)
    rclasses, rscores, rbboxes = bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=Config.predict.nms_threshold)

    visualization.plt_bboxes(image, rclasses, rscores, rbboxes)


def _make_estimator():
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    run_config = tf.contrib.learn.RunConfig(model_dir=Config.train.model_dir)

    model = Model()
    return tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)


def preprocess(fnames):
    vgg_mean = np.array([123, 117, 104])
    images = [cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB) - vgg_mean for fname in fnames]
    images = np.array([cv2.resize(image.astype(np.uint8), tuple(Config.model.image_shape)) for image in images],
                      np.float32)
    return images


def main(fnames):
    images = preprocess(fnames)
    pred_gen = predict(images)

    for fname in fnames:
        show_image(fname, next(pred_gen))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/voc2007.yml',
                        help='config file name')
    parser.add_argument('--file_names', type=str, nargs='+', help='file names,use space as separater')
    args = parser.parse_args()

    Config(args.config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    main(args.file_names)
