#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import os

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

import utils
import style_transfer_tester


def parse_args():
    """
    parsing and configuration
    """
    desc = "Tensorflow implementation of 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--style_model',
        type=str,
        default='models/wave.ckpt',
        help='location for model file (*.ckpt)',
        required=True)

    parser.add_argument(
        '--content',
        type=str,
        default='content/female_knight.jpg',
        help='File path of content image (notation in the paper : x)',
        required=True)

    parser.add_argument(
        '--output',
        type=str,
        default='result',
        help='File path of output image (notation in the paper : y_c)',
        required=True)

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
    # --style_model
    try:
        assert os.path.exists(args.style_model + '.index') and os.path.exists(
            args.style_model + '.meta') and os.path.exists(
                args.style_model + '.data-00000-of-00001')
    except Exception:
        print('There is no %s' % args.style_model)
        print('Tensorflow r0.12 requires 3 files related to *.ckpt')
        print(
            'If you want to restore any models generated from old tensorflow versions, this assert might be ignored'
        )
        return None

    # --content
    try:
        assert os.path.exists(args.content)
    except Exception:
        print('There is no %s' % args.content)
        return None

    # --output
    dirname = os.path.dirname(args.output)
    try:
        if len(dirname) > 0:
            os.stat(dirname)
    except Exception:
        os.makedirs(dirname)

    # --max_size
    try:
        if args.max_size is not None:
            assert args.max_size > 0
        assert os.path.exists(args.content)
    except Exception:
        print('The maximum width or height of input image must be positive')
        return None

    return args


def freeze(sess, style_model, output, style_name):
    tf.train.write_graph(
        sess.graph.as_graph_def(), output, style_name + '.pb', as_text=False)

    freeze_graph(
        input_graph=os.path.join(output, style_name + '.pb'),
        input_saver='',
        input_binary=True,
        input_checkpoint=style_model,
        output_node_names='output',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=os.path.join(output, 'frozen_%s.pb' % style_name),
        clear_devices=False,
        initializer_nodes='')


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # load content image
    content_image = utils.load_image(
        args.content, shape=(480, 640), max_size=args.max_size)

    # style and content name
    style_name = os.path.basename(args.style_model).split('.')[0]
    content_name = os.path.basename(args.content).split('.')[0]

    # open session
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True  # to deal with large image
    sess = tf.Session(config=soft_config)

    # build the graph for test
    transformer = style_transfer_tester.StyleTransferTester(
        session=sess, content_image=content_image, model_path=args.style_model)

    # launch the graph in a session
    start_time = datetime.now()
    output_image = transformer.test()
    end_time = datetime.now()

    # freeze graph
    freeze(sess, args.style_model, args.output, style_name)

    # close session
    sess.close()

    # save result
    utils.save_image(output_image,
                     os.path.join(args.output, '%s_%s.jpg' % (content_name,
                                                              style_name)))

    # report execution time
    shape = content_image.shape  # (batch, width, height, channel)
    print('Execution time for a {} x {} image: {}'.format(
        shape[0], shape[1], (end_time - start_time)))


if __name__ == "__main__":
    main()
