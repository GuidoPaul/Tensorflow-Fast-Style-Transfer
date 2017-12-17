#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import os

import tensorflow as tf

import style_transfer_trainer
import utils
import vgg19


def parse_args():
    """
    parsing and configuration
    """
    desc = "Tensorflow implementation of 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--vgg_model',
        type=str,
        default='pre_trained_model',
        help='The directory where the pre-trained model was saved',
        required=True)
    parser.add_argument(
        '--trainDB_path',
        type=str,
        default='train2014',
        help='The directory where MSCOCO DB was saved',
        required=True)
    parser.add_argument(
        '--style',
        type=str,
        default='style/wave.jpg',
        help='File path of style image (notation in the paper : a)',
        required=True)
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='File path for trained-model. Train-log is also saved here.',
        required=True)

    parser.add_argument(
        '--content_layers',
        nargs='+',
        type=str,
        default=['relu4_2'],
        help='VGG19 layers used for content loss')
    parser.add_argument(
        '--style_layers',
        nargs='+',
        type=str,
        default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
        help='VGG19 layers used for style loss')
    parser.add_argument(
        '--content_weight',
        type=float,
        default=7.5e0,
        help='Weight of content-loss')
    parser.add_argument(
        '--style_weight', type=float, default=5e2, help='Weight of style-loss')
    parser.add_argument(
        '--tv_weight',
        type=float,
        default=2e2,
        help='Weight of total-variance-loss')
    parser.add_argument(
        '--content_layer_weights',
        nargs='+',
        type=float,
        default=[1.0],
        help='Content loss for each content layer')
    parser.add_argument(
        '--style_layer_weights',
        nargs='+',
        type=float,
        default=[.2, .2, .2, .2, .2],
        help='Style loss for each style laryer')

    parser.add_argument(
        '--learn_rate',
        type=float,
        default=1e-3,
        help='Learning rate for Adam optimizer')
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=1000,
        help='save a trained model every after this number of iterations')

    parser.add_argument(
        '--test_image',
        type=str,
        default=None,
        help='File path of content image (notation in the paper : x)')
    parser.add_argument(
        '--max_size',
        type=int,
        default=None,
        help='The maximum width or height of input images')

    return check_args(parser.parse_args())


def check_args(args):
    """
    checking arguments
    """
    # --vgg_model
    model_file_path = args.vgg_model + '/' + vgg19.MODEL_FILE_NAME
    try:
        assert os.path.exists(model_file_path)
    except Exception:
        print('There is no %s' % model_file_path)
        return None
    try:
        size_in_KB = os.path.getsize(model_file_path)
        assert abs(size_in_KB - 534904783) < 10
    except Exception:
        print('check file size of \'imagenet-vgg-verydeep-19.mat\'')
        print('there are some files with the same name')
        print('pre_trained_model used here can be downloaded from bellow')
        print(
            'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
        )
        return None

    # --trainDB_path
    try:
        assert os.path.exists(args.trainDB_path)
    except Exception:
        print('There is no %s' % args.trainDB_path)
        return None

    # --style
    try:
        assert os.path.exists(args.style)
    except Exception:
        print('There is no %s' % args.style)
        return None

    # --output
    dirname = os.path.dirname(args.output)
    try:
        if len(dirname) > 0:
            os.stat(dirname)
    except Exception:
        os.mkdir(dirname)

    # --content_weight
    try:
        assert args.content_weight > 0
    except Exception:
        print('content weight must be positive')

    # --style_weight
    try:
        assert args.style_weight > 0
    except Exception:
        print('style weight must be positive')

    # --tv_weight
    try:
        assert args.tv_weight > 0
    except Exception:
        print('total variance weight must be positive')

    # --content_layer_weights
    try:
        assert len(args.content_layers) == len(args.content_layer_weights)
    except Exception:
        print('content layer info and weight info must be matched')
        return None

    # --style_layer_weights
    try:
        assert len(args.style_layers) == len(args.style_layer_weights)
    except Exception:
        print('style layer info and weight info must be matched')
        return None

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except Exception:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except Exception:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except Exception:
        print('batch size must be larger than or equal to one')

    # --checkpoint_every
    try:
        assert args.checkpoint_every >= 1
    except Exception:
        print('checkpoint period must be larger than or equal to one')

    # --test_image
    try:
        if args.test_image is not None:
            assert os.path.exists(args.test_image)
    except Exception:
        print('There is no %s' % args.test_image)
        return None

    # --max_size
    try:
        if args.max_size is not None:
            assert args.max_size > 0
    except Exception:
        print('The maximum width or height of input image must be positive')
        return None

    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # initiate VGG19 model
    model_file_path = args.vgg_model + '/' + vgg19.MODEL_FILE_NAME
    vgg_net = vgg19.VGG19(model_file_path)

    # get file list for training
    content_images = utils.get_files(args.trainDB_path)

    # load style image
    style_image = utils.load_image(args.style)

    # create a map for content layers info
    CONTENT_LAYERS = {}
    for layer, weight in zip(args.content_layers, args.content_layer_weights):
        CONTENT_LAYERS[layer] = weight

    # create a map for style layers info
    STYLE_LAYERS = {}
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
        STYLE_LAYERS[layer] = weight

    # open session
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=soft_config)

    # build the graph for train
    trainer = style_transfer_trainer.StyleTransferTrainer(
        content_layer_ids=CONTENT_LAYERS,
        style_layer_ids=STYLE_LAYERS,
        session=sess,
        net=vgg_net,
        content_images=content_images,
        style_image=style_image,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        learn_rate=args.learn_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        check_period=args.checkpoint_every,
        save_path=args.output,
        test_image=args.test_image,
        max_size=args.max_size)

    # launch the graph in a session
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    # close session
    sess.close()

    # report execution time
    print('Execution time: {}'.format((end_time - start_time)))


if __name__ == "__main__":
    main()
