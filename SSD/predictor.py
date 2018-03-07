import tensorflow as tf
from PIL import Image
from util import create_default_boxes
from pred_util import *
import visualization


class Predictor():
    def __init__(self,sess,model):
        self.x = tf.placeholder(tf.float32,[None,None,3])
        self.input = self.preprocess()
        self.model = model(self.input)
        self.sess = sess

    def preprocess(self):
        vgg_mean = tf.constant([123, 117, 104],tf.float32)
        image = tf.subtract(self.x,vgg_mean)
        image = tf.image.resize_images(image,(300,300))
        image = tf.expand_dims(image,0)
        return image

    def load(self,check_point_dir):
        self.model.load(self.sess,check_point_dir)

    def predict(self,file_name):
        image = np.array(Image.open(file_name))
        rimg, rpredictions, rlocalisations = self.sess.run([self.input, self.model.prediction_list, self.model.locs_list],{self.x: image})
        ssd_anchors = create_default_boxes()
        rclasses, rscores, rbboxes = ssd_bboxes_select(rpredictions, rlocalisations, ssd_anchors,select_threshold=0.5, decode=True)
        rbboxes = bboxes_clip(rbboxes)
        rclasses, rscores, rbboxes = bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)

        visualization.plt_bboxes(image,rclasses, rscores, rbboxes)




