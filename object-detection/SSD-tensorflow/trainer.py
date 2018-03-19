import tensorflow as tf
import time

class Trainer():
    def __init__(self,sess,model_c,input,target):
        self.model = model_c(input,target)
        self.sess = sess
        self.init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        self.model.load_latest(self.sess,'checkpoints/')
        loss_summary = tf.summary.scalar('loss',self.model.loss)
        writer = tf.summary.FileWriter('../output/logs')
        writer.add_graph(self.sess.graph)
        start_time = time.time()
        try:
            while True:
                _,loss,global_step = self.sess.run([self.model.train_op,self.model.loss,self.model.global_step])
                if global_step % 50 == 0:
                    loss_sum = self.sess.run(loss_summary)
                    writer.add_summary(loss_sum)
                    end_time = time.time()
                    cost_time = round((end_time - start_time)/60,1)
                    print('%f min/50iteration'%(cost_time))
                    start_time = time.time()
                if global_step%1000 == 0:
                    self.model.save(self.sess,'../output/checkpoint','ssd.ckpt')
                    print('step: %d, train_loss: %f' % (global_step,loss))
        except tf.errors.OutOfRangeError:
            print('end!')
        finally:
            writer.flush()
