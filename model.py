from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from warp_utils import dense_image_warp

FLAGS = tf.app.flags.FLAGS
epsilon = 0.001

from functools import partial


def create_outgoing_mask(flow):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))

    grid_x = tf.reshape(tf.range(width), [1, 1, width])
    grid_x = tf.tile(grid_x, [num_batch, height, 1])
    grid_y = tf.reshape(tf.range(height), [1, height, 1])
    grid_y = tf.tile(grid_y, [num_batch, 1, width])

    flow_u, flow_v = tf.unstack(flow, 2, 3)
    pos_x = tf.cast(grid_x, dtype=tf.float32) + flow_u
    pos_y = tf.cast(grid_y, dtype=tf.float32) + flow_v
    inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                              pos_x >= 0.0)
    inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                              pos_y >= 0.0)
    inside = tf.logical_and(inside_x, inside_y)
    return tf.expand_dims(tf.cast(inside, tf.float32), 3)


class ImageReconstruction_reflection(object):
    def __init__(self, batch_size, CROP_PATCH_H, CROP_PATCH_W, level=4):
        self.batch_size = batch_size
        self.CROP_PATCH_H = CROP_PATCH_H
        self.CROP_PATCH_W = CROP_PATCH_W
        self.level = level

    def down(self, x, outChannels, filterSize):
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        return x

    def up(self, x, outChannels, skpCn):
        x = tf.image.resize_bilinear(x, [x.get_shape().as_list()[1] * 2, x.get_shape().as_list()[2] * 2])
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, 3, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(tf.concat([x, skpCn], -1), outChannels, 3, 1, 'same'), 0.1)
        return x

    def FusionLayer_F(self, image_2_F, image_2_B, image_2, image_0, image_1, image_3, image_4,
                      flow20F, flow21F, flow23F, flow24F, lvl):
        with tf.variable_scope("FusionLayer_F_" + str(lvl), reuse=tf.AUTO_REUSE):
            b, h, w, _ = tf.unstack(tf.shape(image_2))

            registrated_foreground_20 = self.warp(image_0, flow20F, b, h, w, 3)
            registrated_foreground_21 = self.warp(image_1, flow21F, b, h, w, 3)
            registrated_foreground_23 = self.warp(image_3, flow23F, b, h, w, 3)
            registrated_foreground_24 = self.warp(image_4, flow24F, b, h, w, 3)

            tf.summary.image('registrated_foreground_20_'+str(lvl), registrated_foreground_20, 3)
            tf.summary.image('registrated_foreground_21_'+str(lvl), registrated_foreground_21, 3)
            tf.summary.image('registrated_foreground_23_'+str(lvl), registrated_foreground_23, 3)
            tf.summary.image('registrated_foreground_24_'+str(lvl), registrated_foreground_24, 3)

            outgoing_mask_20 = create_outgoing_mask(flow20F)
            outgoing_mask_21 = create_outgoing_mask(flow21F)
            outgoing_mask_23 = create_outgoing_mask(flow23F)
            outgoing_mask_24 = create_outgoing_mask(flow24F)

            diff_20 = tf.abs(image_2_F - registrated_foreground_20)
            diff_21 = tf.abs(image_2_F - registrated_foreground_21)
            diff_23 = tf.abs(image_2_F - registrated_foreground_23)
            diff_24 = tf.abs(image_2_F - registrated_foreground_24)

            F_registrated = tf.concat([image_2_F, image_2_B, image_2,
                                       registrated_foreground_20, outgoing_mask_20, diff_20,
                                       registrated_foreground_21, outgoing_mask_21, diff_21,
                                       registrated_foreground_23, outgoing_mask_23, diff_23,
                                       registrated_foreground_24, outgoing_mask_24, diff_24], -1)

            x = tf.concat(F_registrated, axis=3)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(96, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(3, (3, 3), (1, 1), 'same')(x)
            return image_2_F + x[..., 0:3]

    def FusionLayer_B(self, image_2_F, image_2_B, image_2, image_0, image_1, image_3, image_4,
                      flow20B, flow21B, flow23B, flow24B, lvl):
        with tf.variable_scope("FusionLayer_B_" + str(lvl), reuse=tf.AUTO_REUSE):
            b, h, w, _ = tf.unstack(tf.shape(image_2))

            registrated_background_20 = self.warp(image_0, flow20B, b, h, w, 3)
            registrated_background_21 = self.warp(image_1, flow21B, b, h, w, 3)
            registrated_background_23 = self.warp(image_3, flow23B, b, h, w, 3)
            registrated_background_24 = self.warp(image_4, flow24B, b, h, w, 3)

            tf.summary.image('registrated_background_20_'+str(lvl), registrated_background_20, 3)
            tf.summary.image('registrated_background_21_'+str(lvl), registrated_background_21, 3)
            tf.summary.image('registrated_background_23_'+str(lvl), registrated_background_23, 3)
            tf.summary.image('registrated_background_24_'+str(lvl), registrated_background_24, 3)

            outgoing_mask_20 = create_outgoing_mask(flow20B)
            outgoing_mask_21 = create_outgoing_mask(flow21B)
            outgoing_mask_23 = create_outgoing_mask(flow23B)
            outgoing_mask_24 = create_outgoing_mask(flow24B)

            diff_20 = tf.abs(image_2_B - registrated_background_20)
            diff_21 = tf.abs(image_2_B - registrated_background_21)
            diff_23 = tf.abs(image_2_B - registrated_background_23)
            diff_24 = tf.abs(image_2_B - registrated_background_24)

            B_registrated = tf.concat([image_2_F, image_2_B, image_2,
                                       registrated_background_20, outgoing_mask_20, diff_20,
                                       registrated_background_21, outgoing_mask_21, diff_21,
                                       registrated_background_23, outgoing_mask_23, diff_23,
                                       registrated_background_24, outgoing_mask_24, diff_24], -1)

            x = tf.concat(B_registrated, axis=3)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(96, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(3, (3, 3), (1, 1), 'same')(x)
            return image_2_B + x

    def warp(self, I, F, b, h, w, c):
        return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [b, h, w, c])

    def _build_model(self, input_images,
                     F0_last, F1_last, F2_last, F3_last, F4_last,
                     B0_last, B1_last, B2_last, B3_last, B4_last,
                     FF01, FF02, FF03, FF04,
                     FF10, FF12, FF13, FF14,
                     FF20, FF21, FF23, FF24,
                     FF30, FF31, FF32, FF34,
                     FF40, FF41, FF42, FF43,
                     FB01, FB02, FB03, FB04,
                     FB10, FB12, FB13, FB14,
                     FB20, FB21, FB23, FB24,
                     FB30, FB31, FB32, FB34,
                     FB40, FB41, FB42, FB43):

        b = self.batch_size
        h = self.CROP_PATCH_H // (2 ** self.level)
        w = self.CROP_PATCH_W // (2 ** self.level)

        I0 = tf.image.resize_bilinear(input_images[..., 0:3], (h, w))
        I1 = tf.image.resize_bilinear(input_images[..., 3:6], (h, w))
        I2 = tf.image.resize_bilinear(input_images[..., 6:9], (h, w))
        I3 = tf.image.resize_bilinear(input_images[..., 9:12], (h, w))
        I4 = tf.image.resize_bilinear(input_images[..., 12:15], (h, w))


        if self.level == 4:
            F0_last = (I0 + self.warp(I1, FF01, b, h, w, 3)
                         + self.warp(I2, FF02, b, h, w, 3) + self.warp(I3, FF03, b, h, w, 3)
                         + self.warp(I4, FF04, b, h, w, 3))/5.0
            F1_last = (I1 + self.warp(I0, FF10, b, h, w, 3)
                         + self.warp(I2, FF12, b, h, w, 3) + self.warp(I3, FF13, b, h, w, 3)
                         + self.warp(I4, FF14, b, h, w, 3))/5.0
            F2_last = (I2 + self.warp(I0, FF20, b, h, w, 3)
                         + self.warp(I1, FF21, b, h, w, 3) + self.warp(I3, FF23, b, h, w, 3)
                         + self.warp(I4, FF24, b, h, w, 3))/5.0
            F3_last = (I3 + self.warp(I0, FF30, b, h, w, 3)
                         + self.warp(I1, FF31, b, h, w, 3) + self.warp(I2, FF32, b, h, w, 3)
                         + self.warp(I4, FF34, b, h, w, 3))/5.0
            F4_last = (I4 + self.warp(I0, FF40, b, h, w, 3)
                         + self.warp(I1, FF41, b, h, w, 3) + self.warp(I2, FF42, b, h, w, 3)
                         + self.warp(I3, FF43, b, h, w, 3))/5.0
            B0_last = (I0 + self.warp(I1, FB01, b, h, w, 3)
                         + self.warp(I2, FB02, b, h, w, 3) + self.warp(I3, FB03, b, h, w, 3)
                         + self.warp(I4, FB04, b, h, w, 3))/5.0
            B1_last = (I1 + self.warp(I0, FB10, b, h, w, 3)
                         + self.warp(I2, FB12, b, h, w, 3) + self.warp(I3, FB13, b, h, w, 3)
                         + self.warp(I4, FB14, b, h, w, 3))/5.0
            B2_last = (I2 + self.warp(I0, FB20, b, h, w, 3)
                         + self.warp(I1, FB21, b, h, w, 3) + self.warp(I3, FB23, b, h, w, 3)
                         + self.warp(I4, FB24, b, h, w, 3))/5.0
            B3_last = (I3 + self.warp(I0, FB30, b, h, w, 3)
                         + self.warp(I1, FB31, b, h, w, 3) + self.warp(I2, FB32, b, h, w, 3)
                         + self.warp(I4, FB34, b, h, w, 3))/5.0
            B4_last = (I4 + self.warp(I0, FB40, b, h, w, 3)
                         + self.warp(I1, FB41, b, h, w, 3) + self.warp(I2, FB42, b, h, w, 3)
                         + self.warp(I3, FB43, b, h, w, 3))/5.0

        F0_pred = self.FusionLayer_F(F0_last, B0_last, I0, I1, I2, I3, I4, FF01, FF02, FF03, FF04, self.level)
        F1_pred = self.FusionLayer_F(F1_last, B1_last, I1, I0, I2, I3, I4, FF10, FF12, FF13, FF14, self.level)
        F2_pred = self.FusionLayer_F(F2_last, B2_last, I2, I0, I1, I3, I4, FF20, FF21, FF23, FF24, self.level)
        F3_pred = self.FusionLayer_F(F3_last, B3_last, I3, I0, I1, I2, I4, FF30, FF31, FF32, FF34, self.level)
        F4_pred = self.FusionLayer_F(F4_last, B4_last, I4, I0, I1, I2, I3, FF40, FF41, FF42, FF43, self.level)
        B0_pred = self.FusionLayer_B(F0_last, B0_last, I0, I1, I2, I3, I4, FB01, FB02, FB03, FB04, self.level)
        B1_pred = self.FusionLayer_B(F1_last, B1_last, I1, I0, I2, I3, I4, FB10, FB12, FB13, FB14, self.level)
        B2_pred = self.FusionLayer_B(F2_last, B2_last, I2, I0, I1, I3, I4, FB20, FB21, FB23, FB24, self.level)
        B3_pred = self.FusionLayer_B(F3_last, B3_last, I3, I0, I1, I2, I4, FB30, FB31, FB32, FB34, self.level)
        B4_pred = self.FusionLayer_B(F4_last, B4_last, I4, I0, I1, I2, I3, FB40, FB41, FB42, FB43, self.level)

        return F0_pred, F1_pred, F2_pred, F3_pred, F4_pred, B0_pred, B1_pred, B2_pred, B3_pred, B4_pred

class ImageReconstruction_chain_obstruction_1029(object):
    def __init__(self, batch_size, CROP_PATCH_H, CROP_PATCH_W, level=4, weighted_fusion=True):
        self.batch_size = batch_size
        self.CROP_PATCH_H = CROP_PATCH_H
        self.CROP_PATCH_W = CROP_PATCH_W
        self.level = level
        self.weighted_fusion = weighted_fusion

    def down(self, x, outChannels, filterSize):
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        return x

    def up(self, x, outChannels, skpCn):
        x = tf.image.resize_bilinear(x, [x.get_shape().as_list()[1] * 2, x.get_shape().as_list()[2] * 2])
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, 3, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(tf.concat([x, skpCn], -1), outChannels, 3, 1, 'same'), 0.1)
        return x

    def FusionLayer_B(self, image_2_B, alpha, image_2, image_0, image_1, image_3, image_4,
                      flow20B, flow21B, flow23B, flow24B, lvl):
        with tf.variable_scope("FusionLayer_B_" + str(lvl), reuse=tf.AUTO_REUSE):
        # with tf.variable_scope("FusionLayer_B", reuse=tf.AUTO_REUSE):
            b, h, w, _ = tf.unstack(tf.shape(image_2))

            # image_2_B = tf.image.resize_bilinear(image_2_B, (h, w))

            registrated_background_20 = self.warp(image_0, flow20B, b, h, w, 3)
            registrated_background_21 = self.warp(image_1, flow21B, b, h, w, 3)
            registrated_background_23 = self.warp(image_3, flow23B, b, h, w, 3)
            registrated_background_24 = self.warp(image_4, flow24B, b, h, w, 3)

            tf.summary.image('registrated_background_20_'+str(lvl), registrated_background_20, 3)
            tf.summary.image('registrated_background_21_'+str(lvl), registrated_background_21, 3)
            tf.summary.image('registrated_background_23_'+str(lvl), registrated_background_23, 3)
            tf.summary.image('registrated_background_24_'+str(lvl), registrated_background_24, 3)

            outgoing_mask_20 = create_outgoing_mask(flow20B)
            outgoing_mask_21 = create_outgoing_mask(flow21B)
            outgoing_mask_23 = create_outgoing_mask(flow23B)
            outgoing_mask_24 = create_outgoing_mask(flow24B)

            diff_20 = tf.abs(image_2_B - registrated_background_20)
            diff_21 = tf.abs(image_2_B - registrated_background_21)
            diff_23 = tf.abs(image_2_B - registrated_background_23)
            diff_24 = tf.abs(image_2_B - registrated_background_24)

            B_registrated = tf.concat([image_2_B, alpha, image_2,
                                       registrated_background_20, outgoing_mask_20, diff_20,
                                       registrated_background_21, outgoing_mask_21, diff_21,
                                       registrated_background_23, outgoing_mask_23, diff_23,
                                       registrated_background_24, outgoing_mask_24, diff_24], -1)

            x = tf.concat(B_registrated, axis=3)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(96, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            x = tf.nn.leaky_relu(x)
            if self.weighted_fusion:
                x = tf.layers.Conv2D(5, (3, 3), (1, 1), 'same')(x)  # 0 1 3 4, alpha
                weights = tf.nn.softmax(x[..., 0:4], axis=-1)
                img_diff_0 = registrated_background_20 - image_2_B
                img_diff_1 = registrated_background_21 - image_2_B
                img_diff_3 = registrated_background_23 - image_2_B
                img_diff_4 = registrated_background_24 - image_2_B
                output_B = image_2_B + (weights[..., 0:1] * img_diff_0 + weights[..., 1:2] * img_diff_1 + weights[..., 2:3] * img_diff_3 + weights[..., 3:4] * img_diff_4)
                return output_B, alpha + x[..., 4:5]
            else:
                x = tf.layers.Conv2D(4, (3, 3), (1, 1), 'same')(x)
                return image_2_B + x[..., 0:3], alpha + x[..., 3:4]

    def warp(self, I, F, b, h, w, c):
        return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [b, h, w, c])

    def _build_model(self, input_images,
                     B0_last, B1_last, B2_last, B3_last, B4_last,
                     A0_last, A1_last, A2_last, A3_last, A4_last,
                     FB01, FB02, FB03, FB04,
                     FB10, FB12, FB13, FB14,
                     FB20, FB21, FB23, FB24,
                     FB30, FB31, FB32, FB34,
                     FB40, FB41, FB42, FB43):

        b = self.batch_size
        h = self.CROP_PATCH_H // (2 ** self.level)
        w = self.CROP_PATCH_W // (2 ** self.level)

        I0 = tf.image.resize_bilinear(input_images[..., 0:3], (h, w))
        I1 = tf.image.resize_bilinear(input_images[..., 3:6], (h, w))
        I2 = tf.image.resize_bilinear(input_images[..., 6:9], (h, w))
        I3 = tf.image.resize_bilinear(input_images[..., 9:12], (h, w))
        I4 = tf.image.resize_bilinear(input_images[..., 12:15], (h, w))


        if self.level == 4:
            B0_last = (I0 + self.warp(I1, FB01, b, h, w, 3)
                         + self.warp(I2, FB02, b, h, w, 3) + self.warp(I3, FB03, b, h, w, 3)
                         + self.warp(I4, FB04, b, h, w, 3))/5.0
            B1_last = (I1 + self.warp(I0, FB10, b, h, w, 3)
                         + self.warp(I2, FB12, b, h, w, 3) + self.warp(I3, FB13, b, h, w, 3)
                         + self.warp(I4, FB14, b, h, w, 3))/5.0
            B2_last = (I2 + self.warp(I0, FB20, b, h, w, 3)
                         + self.warp(I1, FB21, b, h, w, 3) + self.warp(I3, FB23, b, h, w, 3)
                         + self.warp(I4, FB24, b, h, w, 3))/5.0
            B3_last = (I3 + self.warp(I0, FB30, b, h, w, 3)
                         + self.warp(I1, FB31, b, h, w, 3) + self.warp(I2, FB32, b, h, w, 3)
                         + self.warp(I4, FB34, b, h, w, 3))/5.0
            B4_last = (I4 + self.warp(I0, FB40, b, h, w, 3)
                         + self.warp(I1, FB41, b, h, w, 3) + self.warp(I2, FB42, b, h, w, 3)
                         + self.warp(I3, FB43, b, h, w, 3))/5.0
            # A0_last = 1 - tf.abs(tf.image.rgb_to_grayscale(F0_last) - tf.image.rgb_to_grayscale(I0))
            # A1_last = 1 - tf.abs(tf.image.rgb_to_grayscale(F1_last) - tf.image.rgb_to_grayscale(I1))
            # A2_last = 1 - tf.abs(tf.image.rgb_to_grayscale(F2_last) - tf.image.rgb_to_grayscale(I2))
            # A3_last = 1 - tf.abs(tf.image.rgb_to_grayscale(F3_last) - tf.image.rgb_to_grayscale(I3))
            # A4_last = 1 - tf.abs(tf.image.rgb_to_grayscale(F4_last) - tf.image.rgb_to_grayscale(I4))
            A0_last = tf.zeros_like(B0_last[..., 0:1])
            A1_last = tf.zeros_like(B1_last[..., 0:1])
            A2_last = tf.zeros_like(B2_last[..., 0:1])
            A3_last = tf.zeros_like(B3_last[..., 0:1])
            A4_last = tf.zeros_like(B4_last[..., 0:1])

        B0_pred, A0_pred = self.FusionLayer_B(B0_last, A0_last, I0, I1, I2, I3, I4, FB01, FB02, FB03, FB04, self.level)
        B1_pred, A1_pred = self.FusionLayer_B(B1_last, A1_last, I1, I0, I2, I3, I4, FB10, FB12, FB13, FB14, self.level)
        B2_pred, A2_pred = self.FusionLayer_B(B2_last, A2_last, I2, I0, I1, I3, I4, FB20, FB21, FB23, FB24, self.level)
        B3_pred, A3_pred = self.FusionLayer_B(B3_last, A3_last, I3, I0, I1, I2, I4, FB30, FB31, FB32, FB34, self.level)
        B4_pred, A4_pred = self.FusionLayer_B(B4_last, A4_last, I4, I0, I1, I2, I3, FB40, FB41, FB42, FB43, self.level)

        return B0_pred, B1_pred, B2_pred, B3_pred, B4_pred, A0_pred, A1_pred, A2_pred, A3_pred, A4_pred


class Decomposition_Net_Translation(object):
    def __init__(self, H, W, use_Homography, is_training, use_BN=False):
        self.lvl = 4
        self.filters = [16, 32, 64, 96]
        self.s_range = 4
        self.H = H
        self.W = W
        self.use_Homography = use_Homography
        self.is_training = is_training
        self.use_BN = use_BN

    def inference(self, I0, I1, I2, I3, I4):
        """Inference on a set of input_images.
        Args:
        """
        return self._build_model(I0, I1, I2, I3, I4)

    def down(self, x, outChannels, filterSize):
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, filterSize, 1, 'same'), 0.1)
        return x

    def up(self, x, outChannels, skpCn):
        x = tf.image.resize_bilinear(x, [x.get_shape().as_list()[1] * 2, x.get_shape().as_list()[2] * 2])
        x = tf.nn.leaky_relu(tf.layers.conv2d(x, outChannels, 3, 1, 'same'), 0.1)
        x = tf.nn.leaky_relu(tf.layers.conv2d(tf.concat([x, skpCn], -1), outChannels, 3, 1, 'same'), 0.1)
        return x

    def FeaturePyramidExtractor(self, x):
        with tf.variable_scope("FeaturePyramidExtractor", reuse=tf.AUTO_REUSE):
            for l in range(self.lvl):
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (2, 2), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
            return x

    def CostVolumeLayer(self, features_0, features_0from1):
        cost_length = (2 * self.s_range + 1) ** 2

        def get_cost(features_0, features_0from1, shift):
            def pad2d(x, vpad, hpad):
                return tf.pad(x, [[0, 0], vpad, hpad, [0, 0]])

            def crop2d(x, vcrop, hcrop):
                return tf.keras.layers.Cropping2D([vcrop, hcrop])(x)

            """
            Calculate cost volume for specific shift
            - inputs
            features_0 (batch, h, w, nch): feature maps at time slice 0
            features_0from1 (batch, h, w, nch): feature maps at time slice 0 warped from 1
            shift (2): spatial (vertical and horizontal) shift to be considered
            - output
            cost (batch, h, w): cost volume map for the given shift
            """
            v, h = shift  # vertical/horizontal element
            vt, vb, hl, hr = max(v, 0), abs(min(v, 0)), max(h, 0), abs(min(h, 0))  # top/bottom left/right
            f_0_pad = pad2d(features_0, [vt, vb], [hl, hr])
            f_0from1_pad = pad2d(features_0from1, [vb, vt], [hr, hl])
            cost_pad = f_0_pad * f_0from1_pad
            return tf.reduce_mean(crop2d(cost_pad, [vt, vb], [hl, hr]), axis=3)

        get_c = partial(get_cost, features_0, features_0from1)
        cv = [0] * cost_length
        depth = 0
        for v in range(-self.s_range, self.s_range + 1):
            for h in range(-self.s_range, self.s_range + 1):
                cv[depth] = get_c(shift=[v, h])
                depth += 1

        cv = tf.stack(cv, axis=3)
        cv = tf.nn.leaky_relu(cv, 0.1)
        return cv

    def TranslationEstimator(self, feature_2, feature_0):
        def _conv_block(filters, kernel_size=(3, 3), strides=(1, 1)):
            def f(x):
                x = tf.layers.Conv2D(filters, kernel_size, strides, 'valid')(x)
                x = tf.nn.leaky_relu(x, 0.2)
                return x

            return f

        with tf.variable_scope("TranslationEstimator", reuse=tf.AUTO_REUSE):
            cost = self.CostVolumeLayer(feature_2, feature_0)
            x = tf.concat([feature_2, cost], axis=3)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(96, (3, 3), (1, 1))(x)
            x = _conv_block(64, (3, 3), (1, 1))(x)
            feature = _conv_block(32, (3, 3), (1, 1))(x)
            x = tf.reduce_mean(feature, axis=[1, 2])
            flow1 = tf.layers.dense(x, 2)
            flow2 = tf.layers.dense(x, 2)
            flow1 = tf.expand_dims(tf.expand_dims(flow1, 1), 1)
            flow2 = tf.expand_dims(tf.expand_dims(flow2, 1), 1)
            flow1 = tf.tile(flow1, [1, self.H, self.W, 1])
            flow2 = tf.tile(flow2, [1, self.H, self.W, 1])
            return flow1, flow2

    def HomographyEstimator(self, feature_2, feature_0):
        def _conv_block(filters, kernel_size=(3, 3), strides=(1, 1)):
            def f(x):
                x = tf.layers.Conv2D(filters, kernel_size, strides, 'same')(x)
                if self.use_BN:
                    x = tf.layers.batch_normalization(x, training=self.is_training, trainable=self.is_training)
                x = tf.nn.leaky_relu(x, 0.2)
                return x
            return f

        def homography_matrix_to_flow(tf_homography_matrix, im_shape_w, im_shape_h):
            # tf_homography_matrix [B, 3, 3]
            import numpy as np
            grid_x, grid_y = tf.meshgrid(tf.range(im_shape_w), tf.range(im_shape_h))
            if not self.is_training:
                grid_x = tf.cast(grid_x, tf.float32) / tf.convert_to_tensor(float(self.W)) * tf.convert_to_tensor(20.0)
                grid_y = tf.cast(grid_y, tf.float32) / tf.convert_to_tensor(float(self.H)) * tf.convert_to_tensor(12.0)

            grid_z = tf.ones_like(grid_x)
            tf_XYZ = tf.cast(tf.stack([grid_y, grid_x, grid_z], axis=-1), tf.float32)

            tf_XYZ = tf_XYZ[tf.newaxis, :, :, :, tf.newaxis]  # [1, H, W, 3, 1]
            tf_XYZ = tf.tile(tf_XYZ, [tf_homography_matrix.get_shape().as_list()[0], 1, 1, 1, 1])  # [B, H, W, 3, 1]
            tf_homography_matrix = tf.tile(tf_homography_matrix[:, tf.newaxis, tf.newaxis], (1, im_shape_h, im_shape_w, 1, 1))  # [B, H, W, 3, 3]
            tf_unnormalized_transformed_XYZ = tf.matmul(tf_homography_matrix, tf_XYZ, transpose_b=False)  # [B, H, W, 3, 1]
            tf_transformed_XYZ = tf_unnormalized_transformed_XYZ / tf_unnormalized_transformed_XYZ[:, :, :, -1][:, :, :, tf.newaxis, :]
            flow = -(tf_transformed_XYZ - tf_XYZ)[..., :2, 0]

            if not self.is_training:
                ratio_h = float(self.H) / 12.0
                ratio_w = float(self.W) / 20.0
                ratio_tensor = tf.expand_dims(tf.expand_dims(
                    tf.expand_dims(tf.convert_to_tensor(np.asarray([ratio_w, ratio_h]), dtype=tf.float32), 0), 0), 0)
                flow = flow * ratio_tensor

            return flow

        with tf.variable_scope("HomographyEstimator", reuse=tf.AUTO_REUSE):
            cost = self.CostVolumeLayer(feature_2, feature_0)
            grid_x, grid_y = tf.meshgrid(tf.range(self.W), tf.range(self.H))
            grid_x = tf.cast(grid_x, tf.float32) / (tf.ones([1, 1])*self.W)
            grid_y = tf.cast(grid_y, tf.float32) / (tf.ones([1, 1])*self.H)
            grid_x = tf.tile(tf.expand_dims(tf.expand_dims(grid_x, 0), -1), [feature_2.get_shape().as_list()[0], 1, 1, 1])
            grid_y = tf.tile(tf.expand_dims(tf.expand_dims(grid_y, 0), -1), [feature_2.get_shape().as_list()[0], 1, 1, 1])
            x = tf.concat([feature_2, cost, grid_x, grid_y], axis=3)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(128, (3, 3), (1, 1))(x)
            x = _conv_block(96, (3, 3), (1, 1))(x)
            x = _conv_block(64, (3, 3), (1, 1))(x)
            feature = _conv_block(32, (3, 3), (1, 1))(x)
            x = tf.reduce_mean(feature, axis=[1, 2])
            flow1 = tf.layers.dense(x, 8)
            flow1 = tf.concat([flow1, tf.zeros([flow1.get_shape().as_list()[0], 1], tf.float32)], -1)
            flow1 = tf.reshape(flow1, [flow1.get_shape().as_list()[0], 3, 3])
            flow1 = tf.eye(3, 3, [flow1.get_shape().as_list()[0]]) + flow1
            flow1 = homography_matrix_to_flow(flow1, self.W, self.H)

            flow2 = tf.layers.dense(x, 8)
            flow2 = tf.concat([flow2, tf.zeros([flow2.get_shape().as_list()[0], 1], tf.float32)], -1)
            flow2 = tf.reshape(flow2, [flow2.get_shape().as_list()[0], 3, 3])
            flow2 = tf.eye(3, 3, [flow2.get_shape().as_list()[0]]) + flow2
            flow2 = homography_matrix_to_flow(flow2, self.W, self.H)
            return flow1, flow2

    def warp(self, I, F, b, h, w, c):
        return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [b, h, w, c])

    def _build_model(self, image_0, image_1, image_2, image_3, image_4):

        """P"""
        feature_0 = self.FeaturePyramidExtractor(image_0)
        feature_1 = self.FeaturePyramidExtractor(image_1)
        feature_2 = self.FeaturePyramidExtractor(image_2)
        feature_3 = self.FeaturePyramidExtractor(image_3)
        feature_4 = self.FeaturePyramidExtractor(image_4)

        if self.use_Homography:
            Estimator = self.HomographyEstimator
        else:
            Estimator = self.TranslationEstimator

        FF01, FB01 = Estimator(feature_0, feature_1)
        FF02, FB02 = Estimator(feature_0, feature_2)
        FF03, FB03 = Estimator(feature_0, feature_3)
        FF04, FB04 = Estimator(feature_0, feature_4)

        FF10, FB10 = Estimator(feature_1, feature_0)
        FF12, FB12 = Estimator(feature_1, feature_2)
        FF13, FB13 = Estimator(feature_1, feature_3)
        FF14, FB14 = Estimator(feature_1, feature_4)

        FF20, FB20 = Estimator(feature_2, feature_0)
        FF21, FB21 = Estimator(feature_2, feature_1)
        FF23, FB23 = Estimator(feature_2, feature_3)
        FF24, FB24 = Estimator(feature_2, feature_4)

        FF30, FB30 = Estimator(feature_3, feature_0)
        FF31, FB31 = Estimator(feature_3, feature_1)
        FF32, FB32 = Estimator(feature_3, feature_2)
        FF34, FB34 = Estimator(feature_3, feature_4)

        FF40, FB40 = Estimator(feature_4, feature_0)
        FF41, FB41 = Estimator(feature_4, feature_1)
        FF42, FB42 = Estimator(feature_4, feature_2)
        FF43, FB43 = Estimator(feature_4, feature_3)

        return FF01, FF02, FF03, FF04, \
               FF10, FF12, FF13, FF14, \
               FF20, FF21, FF23, FF24, \
               FF30, FF31, FF32, FF34, \
               FF40, FF41, FF42, FF43, \
               FB01, FB02, FB03, FB04, \
               FB10, FB12, FB13, FB14, \
               FB20, FB21, FB23, FB24, \
               FB30, FB31, FB32, FB34, \
               FB40, FB41, FB42, FB43
