import tensorflow as tf
import time
from train import trainer_utils
from data_reader import reader_utils
import numpy as np
import itertools
import argparse
import sys

class Trainer():
    def __init__(self, sess, model_c,reader,args):
        self.reader = reader
        generator = reader.next_batch()
        self.model = model_c(generator,args=args)
        self.sess = sess
        self.args = args
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        self.model.load_latest(self.sess, 'checkpoints/')

    def train(self):
        summary_writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)
        train_set = reader_utils.get_dataset(self.args.data_dir)
        epoch = 0
        # 循环args.max_nrof_epochs次
        while epoch < self.args.max_nrof_epochs:
            step = self.sess.run(self.model.global_step, feed_dict=None)
            epoch = step // self.args.epoch_size

            image_paths, num_per_class = trainer_utils.sample_people(train_set, self.args.sample_size)
            # path对应的index
            labels_array = np.arange(self.args.sample_size)
            # 把2000个sample放入dataset
            self.sess.run(self.reader.iterator.initializer,{self.reader.image_paths:image_paths})
            batch_number = 0
            if self.args.learning_rate > 0.0:
                lr = self.args.learning_rate
            else:
                lr = trainer_utils.get_learning_rate_from_file('train/learning_rate_schedule.txt', epoch)
            while batch_number < self.args.epoch_size:
                print('Running forward pass on sampled images: ', end='')
                start_time = time.time()
                emb_array = np.zeros((self.args.sample_size, self.args.embedding_size))
                nrof_batches = int(np.ceil(self.args.sample_size / self.args.batch_size))
                # 每次运行batch_size个，将2000个sample全部运行后，填入emb_array
                for i in range(nrof_batches):
                    if i < nrof_batches-1:
                    # lab:序号
                        lab = labels_array[i * self.args.batch_size:(i+1) * self.args.batch_size]
                    else:
                        lab = labels_array[i * self.args.batch_size:]
                    emb = self.sess.run(self.model.embeddings,{self.model.keep_probability:self.args.keep_probability})
                    emb_array[lab, :] = emb[:lab.shape[0]]
                print('%.3f' % (time.time() - start_time))
                # Select triplets based on the embeddings
                print('Selecting suitable triplets for training')
                triplets, nrof_random_negs, nrof_triplets = trainer_utils.select_triplets(emb_array, num_per_class,
                                                                            image_paths,self.args.alpha)
                selection_time = time.time() - start_time
                print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
                      (nrof_random_negs, nrof_triplets, selection_time))

                # Perform training on the selected triplets
                nrof_batches = int(np.ceil(nrof_triplets * 3 / self.args.batch_size))
                triplet_paths = list(itertools.chain(*triplets))
                labels_array = np.arange(len(triplet_paths))
                # 将triplet_paths传入dataset
                self.sess.run(self.reader.iterator.initializer, {self.reader.image_paths: triplet_paths})
                nrof_examples = len(triplet_paths)
                train_time = 0
                i = 0
                emb_array = np.zeros((nrof_examples, self.args.embedding_size))
                loss_array = np.zeros((nrof_triplets,))
                summary = tf.Summary()
                # 将所有triplet送入网络得到loss
                while i < nrof_batches:
                    start_time = time.time()
                    # batch_size = min(nrof_examples - i * self.args.batch_size, self.args.batch_size)
                    if i < nrof_batches - 1:
                        # lab:序号
                        lab = labels_array[i * self.args.batch_size:(i + 1) * self.args.batch_size]
                    else:
                        lab = labels_array[i * self.args.batch_size:]
                    err, _, step, emb = self.sess.run([self.model.loss, self.model.train_op, self.model.global_step, self.model.embeddings],
                                                      feed_dict={self.model.initial_learning_rate: lr,self.model.keep_probability:self.args.keep_probability})
                    emb_array[lab, :] = emb[:lab.shape[0]]
                    loss_array[i] = err
                    duration = time.time() - start_time
                    print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                          (epoch, batch_number + 1, self.args.epoch_size, duration, err))
                    batch_number += 1
                    i += 1
                    train_time += duration
                    summary.value.add(tag='loss', simple_value=err)

                summary.value.add(tag='time/selection', simple_value=selection_time)
                summary_writer.add_summary(summary, step)

            if epoch%10 == 0 :
                self.model.save(self.sess,self.args.model_base_dir,'facenet.ckpt')

