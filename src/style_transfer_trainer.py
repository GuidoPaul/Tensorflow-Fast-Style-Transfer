#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy as np
import os
import time

import tensorflow as tf

import utils
import transform


class StyleTransferTrainer(object):
    def __init__(self, content_layer_ids, style_layer_ids, session, net,
                 content_images, style_image, content_weight, style_weight,
                 tv_weight, learn_rate, num_epochs, batch_size, check_period,
                 save_dir, style_name, test_image, max_size):
        self.net = net
        self.sess = session

        # sort layers info
        self.CONTENT_LAYERS = collections.OrderedDict(
            sorted(content_layer_ids.items()))
        self.STYLE_LAYERS = collections.OrderedDict(
            sorted(style_layer_ids.items()))

        # input images
        mod = len(content_images) % batch_size
        self.x_list = content_images[:-mod]
        self.x_s0 = style_image

        # parameters for optimization
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.check_period = check_period

        # path for model to be saved
        self.save_dir = save_dir
        self.style_name = style_name
        self.save_path = os.path.join(self.save_dir, self.style_name + '.ckpt')

        # image transform network
        self.transform = transform.Transform()
        self.tester = transform.Transform('test')

        # build graph for style transfer
        self._build_graph()

        if test_image is not None:
            self.TEST = True

            # load content image
            self.test_image = utils.load_image(test_image, max_size=max_size)

            # build graph
            self.x_test = tf.placeholder(
                tf.float32, shape=self.test_image.shape, name='test_input')
            self.xi_test = tf.expand_dims(self.x_test,
                                          0)  # add one dim for batch

            # result image from transform-net
            self.y_hat_test = self.tester.net(
                self.xi_test / 255.0
            )  # please build graph for train first. tester.net reuses variables.
        else:
            self.TEST = False

    def _gram_matrix(self, tensor, shape=None):
        if shape is not None:
            B = shape[0]  # batch size
            HW = shape[1]  # height x width
            C = shape[2]  # channels
            CHW = C * HW
        else:
            B, H, W, C = map(lambda i: i.value, tensor.get_shape())
            HW = H * W
            CHW = W * H * C

        # reshape the tensor so it is a (B, 2-dim) matrix
        # so that 'B'th gram matrix can be computed
        feats = tf.reshape(tensor, (B, HW, C))

        # leave dimension of batch as it is
        feats_T = tf.transpose(feats, perm=[0, 2, 1])

        # paper suggests to normalize gram matrix by its number of elements
        gram = tf.matmul(feats_T, feats) / CHW

        return gram

    def _get_total_variation_loss(self, img):
        b, h, w, d = img.get_shape()
        b = b.value
        h = h.value
        w = w.value
        d = d.value
        tv_y_size = (h - 1) * w * d
        tv_x_size = h * (w - 1) * d
        y_tv = tf.nn.l2_loss(img[:, 1:, :, :] -
                             img[:, :self.batch_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(img[:, :, 1:, :] -
                             img[:, :, :self.batch_shape[2] - 1, :])
        loss = 2. * (x_tv / tv_x_size + y_tv / tv_y_size) / b

        loss = tf.cast(loss, tf.float32)
        return loss

    def _build_graph(self):
        # ----- Prepare data -----
        self.batch_shape = (self.batch_size, 256, 256, 3)

        # graph input
        self.x_c = tf.placeholder(
            tf.float32, shape=self.batch_shape, name='content')
        self.x_s = tf.placeholder(
            tf.float32, shape=self.x_s0.shape, name='style')
        self.x_si = tf.expand_dims(self.x_s, 0)  # add one dim for batch

        # preprocess for VGG
        self.x_c_pre = self.net.preprocess(self.x_c)
        self.x_s_pre = self.net.preprocess(self.x_si)

        # get content-layer-feature for content loss
        content_layers = self.net.feed_forward(self.x_c_pre, scope='content')
        self.Ps = {}
        for id in self.CONTENT_LAYERS:
            self.Ps[id] = content_layers[id]

        # get style-layer-feature for style loss
        style_layers = self.net.feed_forward(self.x_s_pre, scope='style')
        self.As = {}
        for id in self.STYLE_LAYERS:
            self.As[id] = self._gram_matrix(style_layers[id])

        # result of image transform net
        self.y_hat = self.transform.net(self.x_c / 255.0)

        # get layer-values for x
        self.y_hat_pre = self.net.preprocess(self.y_hat)
        self.Fs = self.net.feed_forward(self.y_hat_pre, scope='mixed')

        # ----- Compute loss -----
        # style & content losses
        L_content = 0
        L_style = 0

        for id in self.Fs:
            if id in self.CONTENT_LAYERS:
                # content loss
                F = self.Fs[id]  # content feature of x
                P = self.Ps[id]  # content feature of p

                b, h, w, d = F.get_shape(
                )  # first return value is batch size (must be one)
                b = b.value  # batch size
                N = h.value * w.value  # product of width and height
                M = d.value  # number of filters

                w = self.CONTENT_LAYERS[id]  # weight for this layer
                L_content += w * 2 * tf.nn.l2_loss(F - P) / (b * N * M)
            elif id in self.STYLE_LAYERS:
                # style loss
                F = self.Fs[id]

                b, h, w, d = F.get_shape(
                )  # first return value is batch size (must be one)
                b = b.value  # batch size
                N = h.value * w.value  # product of width and height
                M = d.value  # number of filters

                w = self.STYLE_LAYERS[id]  # weight for this layer

                G = self._gram_matrix(F, (b, N, M))  # style feature of x
                A = self.As[id]  # style feature of a

                L_style += w * 2 * tf.nn.l2_loss(G - A) / (b * (M**2))

        # total variation loss
        L_tv = self._get_total_variation_loss(self.y_hat)

        # ----- Compute total loss -----
        # loss of total variation regularization
        alpha = self.content_weight
        beta = self.style_weight
        gamma = self.tv_weight

        self.L_content = alpha * L_content
        self.L_style = beta * L_style
        self.L_tv = gamma * L_tv
        self.L_total = self.L_content + self.L_style + self.L_tv

        # add summary for each loss
        tf.summary.scalar('L_content', self.L_content)
        tf.summary.scalar('L_style', self.L_style)
        tf.summary.scalar('L_tv', self.L_tv)
        tf.summary.scalar('L_total', self.L_total)

    def train(self):
        # ----- Define optimizer Adam -----
        global_step = tf.train.get_or_create_global_step()
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.L_total, trainable_variables)

        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=global_step,
            name='train_step')

        # ----- Tensorboard -----
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(
            self.save_dir, graph=tf.get_default_graph())

        # ----- Session run -----
        self.sess.run(tf.global_variables_initializer())

        # saver to save model
        saver = tf.train.Saver(max_to_keep=1)

        # restore check-point if it exits
        checkpoint_exists = True
        try:
            ckpt_state = tf.train.get_checkpoint_state(self.save_dir)
        except tf.errors.OutOfRangeError as e:
            print('Cannot restore checkpoint: %s' % e)
            checkpoint_exists = False
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            print('No model to restore at %s' % self.save_dir)
            checkpoint_exists = False

        if checkpoint_exists:
            tf.logging.info('Loading checkpoint %s',
                            ckpt_state.model_checkpoint_path)
            saver.restore(self.sess, ckpt_state.model_checkpoint_path)

        # ----- Loop for train -----
        num_examples = len(self.x_list)

        # get iteration info
        if checkpoint_exists:
            iterations = self.sess.run(global_step)
            epoch = (iterations * self.batch_size) // num_examples
            iterations = iterations - epoch * (num_examples // self.batch_size)
        else:
            epoch = 0
            iterations = 0

        while epoch < self.num_epochs:
            while iterations * self.batch_size < num_examples:
                curr = iterations * self.batch_size
                step = curr + self.batch_size
                x_batch = np.zeros(self.batch_shape, dtype=np.float32)
                for j, img_p in enumerate(self.x_list[curr:step]):
                    x_batch[j] = utils.get_img(img_p, (256, 256,
                                                       3)).astype(np.float32)

                iterations += 1

                assert x_batch.shape[0] == self.batch_size

                _, summary, L_total, L_content, L_style, L_tv, gstep = self.sess.run(
                    [
                        train_op, merged_summary_op, self.L_total,
                        self.L_content, self.L_style, self.L_tv, global_step
                    ],
                    feed_dict={self.x_c: x_batch,
                               self.x_s: self.x_s0})

                ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(
                    '%s, epoch: %d, iter: %4d,' % (ftime, epoch, gstep),
                    'L_total: %.4e, L_content: %.4e, L_style: %.4e, L_tv: %.4e'
                    % (L_total, L_content, L_style, L_tv))

                # write logs at every iteration
                summary_writer.add_summary(summary, iterations)

                if gstep % self.check_period == 0:
                    saver.save(self.sess, self.save_path, gstep)

                    if self.TEST:
                        output_image = self.sess.run(
                            [self.y_hat_test],
                            feed_dict={self.x_test: self.test_image})
                        output_image = np.squeeze(
                            output_image[0])  # remove one dim for batch
                        output_image = np.clip(output_image, 0., 255.)

                        test_save_path = os.path.join(self.save_dir,
                                                      '%05d.jpg' % gstep)
                        utils.save_image(output_image, test_save_path)
            epoch += 1
            iterations = 0
        saver.save(self.sess, self.save_path)
