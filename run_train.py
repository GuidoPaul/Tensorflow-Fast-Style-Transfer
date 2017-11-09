#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
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

    return check_args(parser.parse_args())


def check_args(args):
    """
    checking arguments
    """
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
        save_path=args.output)

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
