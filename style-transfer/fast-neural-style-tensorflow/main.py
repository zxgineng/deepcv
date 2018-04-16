import argparse
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import data_loader
from model import Model
from utils import Config


def run(mode, run_config, params):
    model = Model()
    # ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from='logs/pretrained/vgg_16.ckpt',vars_to_warm_start='vgg_16.*')
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=Config.train.model_dir,
        params=params,
        # warm_start_from=ws,
        config=run_config)

    if Config.train.debug:
        debug_hooks = tf_debug.LocalCLIDebugHook()
        hooks = [debug_hooks]
    else:
        hooks = []

    loss_hooks = tf.train.LoggingTensorHook({'total_loss': 'loss/add_7',
                                             'content_loss': 'loss/mul_1',
                                             'style_loss': 'loss/mul_6:0',
                                             'step': 'global_step:0'}, every_n_iter=Config.train.check_hook_n_iter)

    if mode == 'train':
        train_data = data_loader.get_tfrecord(mode, shuffle=True)
        train_input_fn, train_input_hook = data_loader.get_dataset_batch(train_data, buffer_size=1000,
                                                                         batch_size=Config.model.batch_size,
                                                                         scope="train")
        hooks.extend([train_input_hook, loss_hooks])
        estimator.train(input_fn=train_input_fn, hooks=hooks, max_steps=Config.train.max_steps)

    else:
        raise ValueError('no %s mode' % (mode))


def main(mode):
    params = tf.contrib.training.HParams(**Config.train.to_dict())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        model_dir=Config.train.model_dir,
        save_checkpoints_steps=Config.train.save_checkpoints_steps,
        session_config=config,
        log_step_count_steps=None)

    run(mode, run_config, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='train', choices=['train'],
                        help='Mode (train)')
    parser.add_argument('--config', type=str, default='config/COCO.yml', help='config file name')

    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    Config(args.config)
    print(Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    main(args.mode)
