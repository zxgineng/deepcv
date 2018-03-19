"""
训练pkl，或使用pkl进行人脸识别
"""
import tensorflow as tf
import numpy as np
import argparse
from predictor import predictor_utils
import os
import sys
import pickle
from scipy import misc
from data_reader import detect_face
from six.moves import xrange
import cv2
from predictor import classifier

def main(args):

    if args.mode == 'Train':
        classifier.classify(args)

    else:
        images, cout_per_image, nrof_samples,total_box = load_and_align_data(args.image_files,args.image_size, args.margin, args.gpu_memory_fraction)
        with tf.Graph().as_default():

            with tf.Session() as sess:

            # Load the model
                predictor_utils.load_model(args.model)
            # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images , phase_train_placeholder:False}
                emb = sess.run(embeddings, feed_dict=feed_dict)
                classifier_filename_exp = os.path.expanduser(args.classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                predictions = model.predict_proba(emb)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                k=0
                #print predictions
                for i in range(nrof_samples):
                    img = cv2.imread(args.image_files[i])
                    print("\npeople in image %s :" %(args.image_files[i]))
                    for j in range(cout_per_image[i]):
                        bb = total_box[i][j]
                        cv2.rectangle(img, (bb[0],bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        cv2.putText(img, '{}'.format(class_names[best_class_indices[k]]), (bb[0], bb[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255, 0, 0),
                                    thickness=2, lineType=2)
                        print('%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                        k+=1
                    cv2.imshow('test', img)
                    cv2.imwrite('exmaple%d.jpg'%(i),img)
                    cv2.waitKey()
                    cv2.destroyAllWindows()

                    
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    img_list = [] 
    count_per_image = []
    total_bbox = []
    for i in xrange(nrof_samples):
        bbox=[]
        img = cv2.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        count_per_image.append(len(bounding_boxes))
        for j in range(len(bounding_boxes)):	
            det = np.squeeze(bounding_boxes[j,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = predictor_utils.prewhiten(aligned)
            img_list.append(prewhitened)
            bbox.append([bb[0],bb[1],bb[2],bb[3]])
        total_bbox.append(bbox)
    images = np.stack(img_list)
    return images, count_per_image, nrof_samples,total_bbox

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' +
        'model should be used for classification', default='TRAIN')
    parser.add_argument('--image_files', type=str, nargs='+', help='Path(s) of the image(s)',default=['D:/dataset/test/1.jpg','D:/dataset/test/2.jpg'])
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='checkpoints/facenet.pb')
    parser.add_argument('--classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.',default='D:/dataset/my_classifier.pkl')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',default='D:/dataset/aligned')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

