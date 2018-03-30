import argparse
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import data_loader
from model import Model
from utils import Config


def run(mode, run_config, params):
    model = Model()
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        config=run_config)

    if Config.train.debug:
        debug_hooks = tf_debug.LocalCLIDebugHook()
        hooks = [debug_hooks]
    else:
        hooks = []

    loss_hooks = tf.train.LoggingTensorHook({'loss': 'loss/total_loss:0', 'step': 'global_step:0'},
                                            every_n_iter=Config.train.check_hook_n_iter)

    train_data = data_loader.get_tfrecord(shuffle=True)

    train_input_fn, train_input_hook = data_loader.get_dataset_batch(train_data, batch_size=Config.model.batch_size,
                                                                     scope="train")

    hooks.extend([train_input_hook, loss_hooks])
    estimator.train(input_fn=train_input_fn, hooks=hooks, max_steps=Config.train.max_steps)

def main(mode):
    params = tf.contrib.training.HParams(**Config.train.to_dict())

    run_config = tf.estimator.RunConfig(
        model_dir=Config.train.model_dir,
        save_checkpoints_steps=Config.train.save_checkpoints_steps)

    run(mode, run_config, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/voc2007.yml', help='config file name')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    Config(args.config)
    print(Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    main('train')