import tensorflow as tf
import cv2
from model.MTCNN_model import run_mtcnn

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('file_name', 'example/test.jpg', 'file_name')

def main(_):
    img = cv2.imread(FLAGS.file_name)
    bounding_boxes, landmarks = run_mtcnn(img, 20, [0.6, 0.6, 0.7], 0.709)
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)

        cv2.rectangle(img, (face_position[0],
                            face_position[1]),
                      (face_position[2], face_position[3]),
                      (0, 255, 0), 2)

    for landmark in landmarks:
        for i in range(5):
            cv2.circle(img, (int(landmark[i]), int(int(landmark[5 + i]))), 1, (0, 255, 255))

    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    tf.app.run()