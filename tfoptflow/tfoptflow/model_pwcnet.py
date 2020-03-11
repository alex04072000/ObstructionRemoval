"""
model_pwcnet.py
PWC-Net model class.
Written by Phil Ferriere
Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import trange
from tensorflow.contrib.mixed_precision import LossScaleOptimizer, FixedLossScaleManager

from model_base import ModelBase
from optflow import flow_write, flow_write_as_png, flow_mag_stats
from losses import pwcnet_loss
from logger import OptFlowTBLogger
from multi_gpus import assign_to_device, average_gradients
from core_warp import dense_image_warp
from core_costvol import cost_volume
from utils import tf_where

_DEBUG_USE_REF_IMPL = False

# Default options
_DEFAULT_PWCNET_TRAIN_OPTIONS = {
    'verbose': False,
    'ckpt_dir': './ckpts_trained/',  # where training checkpoints are stored
    'max_to_keep': 10,
    'x_dtype': tf.float32,  # image pairs input type
    'x_shape': [2, 384, 448, 3],  # image pairs input shape [2, H, W, 3]
    'y_dtype': tf.float32,  # u,v flows output type
    'y_shape': [384, 448, 2],  # u,v flows output shape [H, W, 2]
    'train_mode': 'train',  # in ['train', 'fine-tune']
    'adapt_info': None,  # if predicted flows are padded by the model, crop them back by to this size
    'sparse_gt_flow': False,  # if gt flows are sparse (KITTI), only compute average EPE where gt flows aren't (0., 0.)
    # Logging/Snapshot params
    'display_step': 100,  # show progress every 100 training batches
    'snapshot_step': 1000,  # save trained model every 1000 training batches
    'val_step': 1000,  # Test trained model on validation split every 1000 training batches
    'val_batch_size': -1,  # Use -1 to use entire validation split, or set number of val samples (0 disables it)
    # None or in ['top_flow', 'pyramid'|; runs trained model on batch_size random val images, log results
    'tb_val_imgs': 'pyramid',
    # None or in ['top_flow', 'pyramid'|; runs trained model on batch_size random test images, log results
    'tb_test_imgs': None,
    # Multi-GPU config
    # list devices on which to run the model's train ops (can be more than one GPU)
    'gpu_devices': ['/device:GPU:0', '/device:GPU:1'],
    # controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
    'controller': '/device:CPU:0',
    # Training config and hyper-params
    'use_tf_data': True,  # Set to True to get data from tf.data.Dataset; otherwise, use feed_dict with numpy
    'use_mixed_precision': False,  # Set to True to use mixed precision training (fp16 inputs)
    'loss_scaler': 128.,  # Loss scaler (only used in mixed precision training)
    'batch_size': 8,
    'lr_policy': 'multisteps',  # choose between None, 'multisteps', and 'cyclic'; adjust the max_steps below too
    # Multistep lr schedule
    'init_lr': 1e-04,  # initial learning rate
    'max_steps': 1200000,  # max number of training iterations (i.e., batches to run)
    'lr_boundaries': [400000, 600000, 800000, 1000000, 1200000],  # step schedule boundaries
    'lr_values': [0.0001, 5e-05, 2.5e-05, 1.25e-05, 6.25e-06, 3.125e-06],  # step schedule values
    # Cyclic lr schedule
    'cyclic_lr_max': 5e-04,  # max bound, anything higher will generate NaNs on `FlyingChairs+FlyingThings3DHalfRes` mix
    'cyclic_lr_base': 1e-05,  # min bound
    'cyclic_lr_stepsize': 20000,  # step schedule values
    # 'max_steps': 200000, # max number of training iterations
    # Loss functions hyper-params
    'loss_fn': 'loss_multiscale',  # See 'Implementation details" on page 5 of ref PDF
    'alphas': [0.32, 0.08, 0.02, 0.01, 0.005, 0.0025],  # See 'Implementation details" on page 5 of ref PDF
    'gamma': 0.0004,  # See 'Implementation details" on page 5 of ref PDF
    'q': 1.,  # See 'Implementation details" on page 5 of ref PDF
    'epsilon': 0.,  # See 'Implementation details" on page 5 of ref PDF
    # Model hyper-params
    'pyr_lvls': 6,  # number of feature levels in the flow pyramid
    'flow_pred_lvl': 2,  # which level to upsample to generate the final optical flow prediction
    'search_range': 4,  # cost volume search range
    # if True, use model with dense connections (4705064 params w/o, 9374274 params with (no residual conn.))
    'use_dense_cx': False,
    # if True, use model with residual connections (4705064 params w/o, 6774064 params with (+2069000) (no dense conn.))
    'use_res_cx': False,
}

_DEFAULT_PWCNET_FINETUNE_OPTIONS = {
    'verbose': False,
    'ckpt_path': './ckpts_trained/pwcnet.ckpt',  # original checkpoint to finetune
    'ckpt_dir': './ckpts_finetuned/',  # where finetuning checkpoints are stored
    'max_to_keep': 10,
    'x_dtype': tf.float32,  # image pairs input type
    'x_shape': [2, 384, 768, 3],  # image pairs input shape [2, H, W, 3]
    'y_dtype': tf.float32,  # u,v flows output type
    'y_shape': [384, 768, 2],  # u,v flows output shape [H, W, 2]
    'train_mode': 'fine-tune',  # in ['train', 'fine-tune']
    'adapt_info': None,  # if predicted flows are padded by the model, crop them back by to this size
    'sparse_gt_flow': False,  # if gt flows are sparse (KITTI), only compute average EPE where gt flows aren't (0., 0.)
    # Logging/Snapshot params
    'display_step': 100,  # show progress every 100 training batches
    'snapshot_step': 1000,  # save trained model every 1000 training batches
    'val_step': 1000,  # Test trained model on validation split every 1000 training batches
    'val_batch_size': -1,  # Use -1 to use entire validation split, or set number of val samples (0 disables it)
    'tb_val_imgs': 'top_flow',  # None, 'top_flow', or 'pyramid'; runs model on batch_size val images, log results
    'tb_test_imgs': None,  # None, 'top_flow', or 'pyramid'; runs trained model on batch_size test images, log results
    # Multi-GPU config
    # list devices on which to run the model's train ops (can be more than one GPU)
    'gpu_devices': ['/device:GPU:0', '/device:GPU:1'],
    # controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
    'controller': '/device:CPU:0',
    # Training config and hyper-params
    'use_tf_data': True,  # Set to True to get data from tf.data.Dataset; otherwise, use feed_dict with numpy
    'use_mixed_precision': False,  # Set to True to use mixed precision training (fp16 inputs)
    'loss_scaler': 128.,  # Loss scaler (only used in mixed precision training)
    'batch_size': 4,
    'lr_policy': 'multisteps',  # choose between None, 'multisteps', and 'cyclic'; adjust the max_steps below too
    # Multistep lr schedule
    'init_lr': 1e-05,  # initial learning rate
    'max_steps': 500000,  # max number of training iterations (i.e., batches to run)
    'lr_boundaries': [200000, 300000, 400000, 500000],  # step schedule boundaries
    'lr_values': [1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07],  # step schedule values
    # Cyclic lr schedule
    'cyclic_lr_max': 2e-05,  # maximum bound
    'cyclic_lr_base': 1e-06,  # min bound
    'cyclic_lr_stepsize': 20000,  # step schedule values
    # 'max_steps': 200000, # max number of training iterations
    # Loss functions hyper-params
    'loss_fn': 'loss_robust',  # 'loss_robust' doesn't really work; the loss goes down but the EPE doesn't
    'alphas': [0.32, 0.08, 0.02, 0.01, 0.005],  # See 'Implementation details" on page 5 of ref PDF
    'gamma': 0.0004,  # See 'Implementation details" on page 5 of ref PDF
    'q': 0.4,  # See 'Implementation details" on page 5 of ref PDF
    'epsilon': 0.01,  # See 'Implementation details" on page 5 of ref PDF
    # Model hyper-params
    'pyr_lvls': 6,  # number of feature levels in the flow pyramid
    'flow_pred_lvl': 2,  # which level to upsample to generate the final optical flow prediction
    'search_range': 4,  # cost volume search range
    # if True, use model with dense connections (4705064 params w/o, 9374274 params with (no residual conn.))
    'use_dense_cx': False,
    # if True, use model with residual connections (4705064 params w/o, 6774064 params with (+2069000) (no dense conn.))
    'use_res_cx': False,
}

_DEFAULT_PWCNET_VAL_OPTIONS = {
    'verbose': False,
    'ckpt_path': './ckpts_trained/pwcnet.ckpt',
    'x_dtype': tf.float32,  # image pairs input type
    'x_shape': [2, None, None, 3],  # image pairs input shape [2, H, W, 3]
    'y_dtype': tf.float32,  # u,v flows output type
    'y_shape': [None, None, 2],  # u,v flows output shape [H, W, 2]
    'adapt_info': None,  # if predicted flows are padded by the model, crop them back by to this size
    'sparse_gt_flow': False,  # if gt flows are sparse (KITTI), only compute average EPE where gt flows aren't (0., 0.)
    # Multi-GPU config
    # list devices on which to run the model's train ops (can be more than one GPU)
    'gpu_devices': ['/device:GPU:0', '/device:GPU:1'],
    # controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
    'controller': '/device:CPU:0',
    # Eval config and hyper-params
    'batch_size': 1,
    'use_tf_data': True,  # Set to True to get data from tf.data.Dataset; otherwise, use feed_dict with numpy
    'use_mixed_precision': False,  # Set to True to use fp16 inputs
    # Model hyper-params
    'pyr_lvls': 6,  # number of feature levels in the flow pyramid
    'flow_pred_lvl': 2,  # which level to upsample to generate the final optical flow prediction
    'search_range': 4,  # cost volume search range
    # if True, use model with dense connections (4705064 params w/o, 9374274 params with (no residual conn.))
    'use_dense_cx': False,
    # if True, use model with residual connections (4705064 params w/o, 6774064 params with (+2069000) (no dense conn.))
    'use_res_cx': False,
}

_DEFAULT_PWCNET_TEST_OPTIONS = {
    'verbose': False,
    'ckpt_path': './ckpts_trained/pwcnet.ckpt',
    'x_dtype': tf.float32,  # image pairs input type
    'x_shape': [2, None, None, 3],  # image pairs input shape
    'y_dtype': tf.float32,  # u,v flows output type
    'y_shape': [None, None, 2],  # u,v flows output shape
    # Multi-GPU config
    # list devices on which to run the model's train ops (can be more than one GPU)
    'gpu_devices': ['/device:GPU:0', '/device:GPU:1'],
    # controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
    'controller': '/device:CPU:0',
    # Eval config and hyper-params
    'batch_size': 1,
    'use_tf_data': True,  # Set to True to get data from tf.data.Dataset; otherwise, use feed_dict with numpy
    'use_mixed_precision': False,  # Set to True to use fp16 inputs
    # Model hyper-params
    'pyr_lvls': 6,  # number of feature levels in the flow pyramid
    'flow_pred_lvl': 2,  # which level to upsample to generate the final optical flow prediction
    'search_range': 4,  # cost volume search range
    # if True, use model with dense connections (4705064 params w/o, 9374274 params with (no residual conn.))
    'use_dense_cx': False,
    # if True, use model with residual connections (4705064 params w/o, 6774064 params with (+2069000) (no dense conn.))
    'use_res_cx': False,
}

# from ref_model import PWCNet


class ModelPWCNet(ModelBase):
    def __init__(self, name='pwcnet', mode='train', session=None, options=_DEFAULT_PWCNET_TEST_OPTIONS, dataset=None):
        """Initialize the ModelPWCNet object
        Args:
            name: Model name
            mode: Possible values: 'train', 'val', 'test'
            session: optional TF session
            options: see _DEFAULT_PWCNET_TRAIN_OPTIONS comments
            dataset: Dataset loader
        Training Ref:
            Per page 4 of paper, section "Training loss," the loss function used in regular training mode is the same as
            the one used in Dosovitskiy et al's "FlowNet: Learning optical flow with convolutional networks" paper
            (multiscale training loss). For fine-tuning, the loss function used is described at the top of page 5
            (robust training loss).
            Per page 5 of paper, section "Implementation details," the trade-off weight gamma in the regularization term
            is usually set to 0.0004.
            Per page 5 of paper, section "Implementation details," we first train the models using the FlyingChairs
            dataset using the S<sub>long</sub> learning rate schedule introduced in E. Ilg et al.'s "FlowNet 2.0:
            Evolution of optical flow estimation with deep networks", starting from 0.0001 and reducing the learning
            rate by half at 0.4M, 0.6M, 0.8M, and 1M iterations. The data augmentation scheme is the same as in that
            paper. We crop 448 ? 384 patches during data augmentation and use a batch size of 8. We then fine-tune the
            models on the FlyingThings3D dataset using the S<sub>fine</sub> schedule while excluding image pairs with
            extreme motion (magnitude larger than 1000 pixels). The cropped image size is 768 ? 384 and the batch size
            is 4. Finally, we finetune the models using the Sintel and KITTI training set as detailed in section "4.1.
            Main results".
        """
        super().__init__(name, mode, session, options)
        self.ds = dataset
        # self.adapt_infos = []
        # self.unique_y_shapes = []

    ###
    # Model mgmt
    ###
    def build_model(self):
        print('yulunliu build_model')
        """Build model
        Called by the base class when building the TF graph to setup the list of output tensors
        """
        if self.opts['verbose']:
            print("Building model...")
        assert(self.num_gpus <= 1)

        # Build the backbone neural nets and collect the output tensors
        with tf.device(self.opts['controller']):
            self.flow_pred_tnsr, self.flow_pyr_tnsr = self.nn(self.x_tnsr)

        if self.opts['verbose']:
            print("... model built.")

    def build_model_towers(self):
        """Build model towers. A tower is the name used to describe a copy of the model on a device.
        Called by the base class when building the TF graph to setup the list of output tensors
        """
        if self.opts['verbose']:
            print("Building model towers...")

        # Setup a learning rate training schedule
        self.setup_lr_sched()

        # Instantiate an optimizer
        # see https://stackoverflow.com/questions/42064941/tensorflow-float16-support-is-broken
        # for float32 epsilon=1e-08, for float16 use epsilon=1e-4
        epsilon = 1e-08 if self.opts['use_mixed_precision'] is False else 1e-4
        assert (self.opts['train_mode'] in ['train', 'fine-tune'])
        if self.opts['loss_fn'] == 'loss_multiscale':
            self.optim = tf.train.AdamOptimizer(self.lr, epsilon=epsilon)
        else:
            self.optim = tf.train.ProximalGradientDescentOptimizer(self.lr)

        # Keep track of the gradients and losses per tower
        tower_grads, losses, metrics = [], [], []

        # Get the current variable scope so we can reuse all variables we need once we get
        # to the next iteration of the for loop below
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            for n, ops_device in enumerate(self.opts['gpu_devices']):
                print("  Building tower_"+str(n)+"...")
                # Use the assign_to_device function to ensure that variables are created on the controller.
                with tf.device(assign_to_device(ops_device, self.opts['controller'])), tf.name_scope('tower_'+str(n)):
                    # Get a slice of the input batch and groundtruth label
                    x_tnsr = self.x_tnsr[n * self.opts['batch_size']:(n + 1) * self.opts['batch_size'], :]
                    y_tnsr = self.y_tnsr[n * self.opts['batch_size']:(n + 1) * self.opts['batch_size'], :]

                    # Build the model for that slice
                    flow_pred_tnsr, flow_pyr_tnsr = self.nn(x_tnsr)

                    # The first tower is also the model we will use to perform online evaluation
                    if n == 0:
                        self.flow_pred_tnsr, self.flow_pyr_tnsr = flow_pred_tnsr, flow_pyr_tnsr

                    # Compute the loss for this tower, with regularization term if requested
                    loss_unreg = pwcnet_loss(y_tnsr, flow_pyr_tnsr, self.opts)
                    if self.opts['gamma'] == 0.:
                        loss = loss_unreg
                    else:
                        loss_reg = self.opts['gamma'] * \
                            tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                        loss = loss_unreg + loss_reg

                    # Evaluate model performance on this tower
                    metrics.append(tf.reduce_mean(tf.norm(y_tnsr - flow_pred_tnsr, ord=2, axis=3)))

                    # Compute the gradients for this tower, but don't apply them yet
                    with tf.name_scope("compute_gradients"):
                        # The function compute_gradients() returns a list of (gradient, variable) pairs
                        if self.opts['use_mixed_precision'] is True:
                            grads, vars = zip(*self.optim.compute_gradients(loss * self.opts['loss_scaler']))
                            # Return the gradients (now float32) to the correct exponent and keep them in check
                            grads = [grad / self.opts['loss_scaler'] for grad in grads]
                            grads, _ = tf.clip_by_global_norm(grads, 5.0)
                            tower_grads.append(zip(grads, vars))
                        else:
                            grad_and_vars = self.optim.compute_gradients(loss)
                            tower_grads.append(grad_and_vars)

                    losses.append(loss)

                # After the first iteration, we want to reuse the variables.
                outer_scope.reuse_variables()
                print("  ...tower_"+str(n)+" built.")

        # Apply the gradients on the controlling device
        with tf.name_scope("apply_gradients"), tf.device(self.opts['controller']):
            # Note that what we are doing here mathematically is equivalent to returning the average loss over the
            # towers and compute the gradients relative to that. Unfortunately, this would place all gradient
            # computations on one device, which is why we had to compute the gradients above per tower and need to
            # average them here. The function average_gradients() takes the list of (gradient, variable) lists
            # and turns it into a single (gradient, variables) list.
            avg_grads_op = average_gradients(tower_grads)
            self.optim_op = self.optim.apply_gradients(avg_grads_op, self.g_step_op)
            self.loss_op = tf.reduce_mean(losses)
            self.metric_op = tf.reduce_mean(metrics)

        if self.opts['verbose']:
            print("... model towers built.")

    def set_output_tnsrs(self):
        """Initialize output tensors
        """
        if self.mode in ['train_noval', 'train_with_val']:
            # self.y_hat_train_tnsr = [self.loss_op, self.metric_op, self.optim_op, self.g_step_inc_op]
            self.y_hat_train_tnsr = [self.loss_op, self.metric_op, self.optim_op]

        if self.mode == 'train_with_val':
            # In online evaluation mode, we only care about the average loss and metric for the batch:
            self.y_hat_val_tnsr = [self.loss_op, self.metric_op]

        if self.mode in ['val', 'val_notrain']:
            # In offline evaluation mode, we only care about the individual predictions and metrics:
            self.y_hat_val_tnsr = [self.flow_pred_tnsr, self.metric_op]

            # if self.opts['sparse_gt_flow'] is True:
            #     # Find the location of the zerod-out flows in the gt
            #     zeros_loc = tf.logical_and(tf.equal(self.y_tnsr[:, :, :, 0], 0.0), tf.equal(self.y_tnsr[:, :, :, 1], 0.0))
            #     zeros_loc = tf.expand_dims(zeros_loc, -1)
            #
            #     # Zero out flow predictions at the same location so we only compute the EPE at the sparse flow points
            #     sparse_flow_pred_tnsr = tf_where(zeros_loc, tf.zeros_like(self.flow_pred_tnsr), self.flow_pred_tnsr)
            #
            #     self.y_hat_val_tnsr = [sparse_flow_pred_tnsr, self.metric_op]

        self.y_hat_test_tnsr = [self.flow_pred_tnsr, self.flow_pyr_tnsr]

    ###
    # Sample mgmt
    ###
    def adapt_x(self, x):
        """Preprocess the input samples to adapt them to the network's requirements
        Here, x, is the actual data, not the x TF tensor.
        Args:
            x: input samples in list[(2,H,W,3)] or (N,2,H,W,3) np array form
        Returns:
            Samples ready to be given to the network (w. same shape as x)
            Also, return adaptation info in (N,2,H,W,3) format
        """
        # Ensure we're dealing with RGB image pairs
        assert (isinstance(x, np.ndarray) or isinstance(x, list))
        if isinstance(x, np.ndarray):
            assert (len(x.shape) == 5)
            assert (x.shape[1] == 2 and x.shape[4] == 3)
        else:
            assert (len(x[0].shape) == 4)
            assert (x[0].shape[0] == 2 or x[0].shape[3] == 3)

        # Bring image range from 0..255 to 0..1 and use floats (also, list[(2,H,W,3)] -> (batch_size,2,H,W,3))
        if self.opts['use_mixed_precision'] is True:
            x_adapt = np.array(x, dtype=np.float16) if isinstance(x, list) else x.astype(np.float16)
        else:
            x_adapt = np.array(x, dtype=np.float32) if isinstance(x, list) else x.astype(np.float32)
        x_adapt /= 255.

        # Make sure the image dimensions are multiples of 2**pyramid_levels, pad them if they're not
        _, pad_h = divmod(x_adapt.shape[2], 2**self.opts['pyr_lvls'])
        if pad_h != 0:
            pad_h = 2 ** self.opts['pyr_lvls'] - pad_h
        _, pad_w = divmod(x_adapt.shape[3], 2**self.opts['pyr_lvls'])
        if pad_w != 0:
            pad_w = 2 ** self.opts['pyr_lvls'] - pad_w
        x_adapt_info = None
        if pad_h != 0 or pad_w != 0:
            padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)]
            x_adapt_info = x_adapt.shape  # Save original shape
            x_adapt = np.pad(x_adapt, padding, mode='constant', constant_values=0.)

        return x_adapt, x_adapt_info

    def adapt_y(self, y):
        """Preprocess the labels to adapt them to the loss computation requirements of the network
        Here, y, is the actual data, not the y TF tensor.
        Args:
            y: labels in list[(H,W,2)] or (N,H,W,2) np array form
        Returns:
            Labels ready to be used by the network's loss function (w. same shape as y)
            Also, return adaptation info in (N,H,W,2) format
        """
        # Ensure we're dealing with u,v flows
        assert (isinstance(y, np.ndarray) or isinstance(y, list))
        if isinstance(y, np.ndarray):
            assert (len(y.shape) == 4)
            assert (y.shape[3] == 2)
        else:
            assert (len(y[0].shape) == 3)
            assert (y[0].shape[2] == 2)

        y_adapt = np.array(y, dtype=np.float32) if isinstance(y, list) else y  # list[(H,W,2)] -> (batch_size,H,W,2)

        # Make sure the flow dimensions are multiples of 2**pyramid_levels, pad them if they're not
        _, pad_h = divmod(y.shape[1], 2**self.opts['pyr_lvls'])
        if pad_h != 0:
            pad_h = 2 ** self.opts['pyr_lvls'] - pad_h
        _, pad_w = divmod(y.shape[2], 2**self.opts['pyr_lvls'])
        if pad_w != 0:
            pad_w = 2 ** self.opts['pyr_lvls'] - pad_w
        y_adapt_info = None
        if pad_h != 0 or pad_w != 0:
            padding = [(0, 0), (0, pad_h), (0, pad_w), (0, 0)]
            y_adapt_info = y_adapt.shape  # Save original shape
            y_adapt = np.pad(y_adapt, padding, mode='constant', constant_values=0.)

        # if y_adapt_info is not None and not y_adapt_info in self.adapt_infos: self.adapt_infos.append(y_adapt_info)
        # if not y.shape in self.unique_y_shapes: self.unique_y_shapes.append(y.shape)

        return y_adapt, y_adapt_info

    def postproc_y_hat_test(self, y_hat, adapt_info=None):
        """Postprocess the results coming from the network during the test mode.
        Here, y_hat, is the actual data, not the y_hat TF tensor. Override as necessary.
        Args:
            y_hat: predictions, see set_output_tnsrs() for details
            adapt_info: adaptation information in (N,H,W,2) format
        Returns:
            Postprocessed labels
        """
        assert (isinstance(y_hat, list) and len(y_hat) == 2)

        # Have the samples been padded to fit the network's requirements? If so, crop flows back to original size.
        pred_flows = y_hat[0]
        if adapt_info is not None:
            pred_flows = pred_flows[:, 0:adapt_info[1], 0:adapt_info[2], :]

        # Individuate flows of the flow pyramid (at this point, they are still batched)
        pyramids = y_hat[1]
        pred_flows_pyramid = []
        for idx in range(len(pred_flows)):
            pyramid = []
            for lvl in range(self.opts['pyr_lvls'] - self.opts['flow_pred_lvl'] + 1):
                pyramid.append(pyramids[lvl][idx])
            pred_flows_pyramid.append(pyramid)

        return pred_flows, pred_flows_pyramid

    def postproc_y_hat_train(self, y_hat, adapt_info=None):
        """Postprocess the results coming from the network during training.
        Here, y_hat, is the actual data, not the y_hat TF tensor. Override as necessary.
        Args:
            y_hat: losses and metrics, see set_output_tnsrs() for details
            adapt_info: adaptation information in (N,H,W,2) format
        Returns:
            Batch loss and metric
        """
        assert (isinstance(y_hat, list) and len(y_hat) == 3)

        return y_hat[0], y_hat[1]

    def postproc_y_hat_val(self, y_hat, adapt_info=None):
        """Postprocess the results coming from the network during validation.
        Here, y_hat, is the actual data, not the y_hat TF tensor. Override as necessary.
        Args:
            y_hat: batch loss and metric, or predicted flows and metrics, see set_output_tnsrs() for details
            adapt_info: adaptation information in (N,H,W,2) format
        Returns:
            Either, batch loss and metric
            Or, predicted flows and metrics
        """
        if self.mode in ['train_noval', 'train_with_val']:
            # In online evaluation mode, we only care about the average loss and metric for the batch:
            assert (isinstance(y_hat, list) and len(y_hat) == 2)
            return y_hat[0], y_hat[1]

        if self.mode in ['val', 'val_notrain']:
            # Have the samples been padded to fit the network's requirements? If so, crop flows back to original size.
            pred_flows = y_hat[0]
            if adapt_info is not None:
                pred_flows = pred_flows[:, 0:adapt_info[1], 0:adapt_info[2], :]
            return pred_flows, y_hat[1]

    ###
    # Training  helpers
    ###
    def setup_loss_ops(self):
        """Setup loss computations. See pwcnet_loss() function for unregularized loss implementation details.
        """
        # Setup unregularized loss
        loss_unreg = pwcnet_loss(self.y_tnsr, self.flow_pyr_tnsr, self.opts)

        # Add regularization term
        if self.opts['gamma'] == 0.:
            self.loss_op = loss_unreg
        else:
            loss_reg = self.opts['gamma'] * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            self.loss_op = loss_unreg + loss_reg

    def setup_optim_op(self):
        """Select the Adam optimizer, define the optimization process.
        """
        # Instantiate optimizer
        # see https://stackoverflow.com/questions/42064941/tensorflow-float16-support-is-broken
        # for float32 epsilon=1e-08, for float16 use epsilon=1e-4
        epsilon = 1e-08 if self.opts['use_mixed_precision'] is False else 1e-4
        if self.opts['loss_fn'] == 'loss_multiscale':
            self.optim = tf.train.AdamOptimizer(self.lr, epsilon=epsilon)
        else:
            self.optim = tf.train.ProximalGradientDescentOptimizer(self.lr)

        if self.opts['use_mixed_precision'] is True:
            # Choose a loss scale manager which decides how to pick the right loss scale throughout the training process.
            loss_scale_mgr = FixedLossScaleManager(self.opts['loss_scaler'])

            # Wrap the original optimizer in a LossScaleOptimizer
            self.optim = LossScaleOptimizer(self.optim, loss_scale_mgr)

            # Let minimize() take care of both computing the gradients and applying them to the model variables
            self.optim_op = self.optim.minimize(self.loss_op, self.g_step_op, tf.trainable_variables())
        else:
            # Let minimize() take care of both computing the gradients and applying them to the model variables
            self.optim_op = self.optim.minimize(self.loss_op, self.g_step_op, tf.trainable_variables())

    def config_train_ops(self):
        """Configure training ops.
        Called by the base class when building the TF graph to setup all the training ops, including:
            - setting up loss computations,
            - setting up metrics computations,
            - creating a learning rate training schedule,
            - selecting an optimizer,
            - creating lists of output tensors.
        """
        assert (self.opts['train_mode'] in ['train', 'fine-tune'])
        if self.opts['verbose']:
            print("Configuring training ops...")

        # Setup loss computations
        self.setup_loss_ops()

        # Setup metrics computations
        self.setup_metrics_ops()

        # Setup a learning rate training schedule
        self.setup_lr_sched()

        # Setup optimizer computations
        self.setup_optim_op()

        if self.opts['verbose']:
            print("... training ops configured.")

    def config_loggers(self):
        """Configure train logger and, optionally, val logger. Here add a logger for test images, if requested.
        """
        super().config_loggers()
        if self.opts['tb_test_imgs'] is True:
            self.tb_test = OptFlowTBLogger(self.opts['ckpt_dir'], 'test')

    def train(self):
        """Training loop
        """
        with self.graph.as_default():
            # Reset step counter
            if self.opts['train_mode'] == 'fine-tune':
                step = 1
                self.sess.run(self.g_step_op.assign(0))
                if self.opts['verbose']:
                    print("Start finetuning...")
            else:
                if self.last_ckpt is not None:
                    step = self.g_step_op.eval(session=self.sess) + 1
                    if self.opts['verbose']:
                        print("Resume training from step "+str(step)+"...")
                else:
                    step = 1
                    if self.opts['verbose']:
                        print("Start training from scratch...")

            # Get batch sizes
            batch_size = self.opts['batch_size']
            val_batch_size = self.opts['val_batch_size']
            if self.mode == 'train_noval':
                warnings.warn("Setting val_batch_size=0 because dataset is in 'train_noval' mode")
                val_batch_size = 0
            if val_batch_size == -1:
                val_batch_size = self.ds.val_size

            # Init batch progress trackers
            train_loss, train_epe, duration = [], [], []
            ranking_value = 0

            # Only load Tensorboard validation/test images once
            if self.opts['tb_val_imgs'] is not None:
                tb_val_loaded = False
            if self.opts['tb_test_imgs'] is not None:
                tb_test_loaded = False

            # Use feed_dict from np or with tf.data.Dataset?
            if self.opts['use_tf_data'] is True:
                # Create tf.data.Dataset managers
                train_tf_ds = self.ds.get_tf_ds(batch_size, self.num_gpus, split='train', sess=self.sess)
                val_tf_ds = self.ds.get_tf_ds(batch_size, self.num_gpus, split='val', sess=self.sess)

                # Ops for initializing the two different iterators
                train_next_batch = train_tf_ds.make_one_shot_iterator().get_next()
                val_next_batch = val_tf_ds.make_one_shot_iterator().get_next()

            while step < self.opts['max_steps'] + 1:

                # Get a batch of samples and make them conform to the network's requirements
                # x: [batch_size*num_gpus,2,H,W,3] uint8 y: [batch_size*num_gpus,H,W,2] float32
                # x_adapt: [batch_size,2,H,W,3] float32 y_adapt: [batch_size,H,W,2] float32
                if self.opts['use_tf_data'] is True:
                    x, y, _ = self.sess.run(train_next_batch)
                else:
                    x, y, _ = self.ds.next_batch(batch_size * self.num_gpus, split='train')
                x_adapt, _ = self.adapt_x(x)
                y_adapt, _ = self.adapt_y(y)

                # Run the samples through the network (loss, error rate, and optim ops (backprop))
                feed_dict = {self.x_tnsr: x_adapt, self.y_tnsr: y_adapt}
                start_time = time.time()
                y_hat = self.sess.run(self.y_hat_train_tnsr, feed_dict=feed_dict)
                duration.append(time.time() - start_time)
                loss, epe = self.postproc_y_hat_train(y_hat)  # y_hat: [107.0802, 5.8556495, None]
                # if self.num_gpus == 1: # Single-GPU case
                # else: # Multi-CPU case

                train_loss.append(loss), train_epe.append(epe)

                # Show training progress
                if step % self.opts['display_step'] == 0:
                    # Send results to tensorboard
                    loss, epe = np.mean(train_loss), np.mean(train_epe)
                    ranking_value = epe
                    self.tb_train.log_scalar("losses/loss", loss, step)
                    self.tb_train.log_scalar("metrics/epe", epe, step)
                    lr = self.lr.eval(session=self.sess)
                    self.tb_train.log_scalar("optim/lr", lr, step)

                    # Print results, if requested
                    if self.opts['verbose']:
                        sec_per_step = np.mean(duration)
                        samples_per_step = batch_size * self.num_gpus
                        samples_per_sec = samples_per_step / sec_per_step
                        eta = round((self.opts['max_steps'] - step) * sec_per_step)
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        # status = f"{ts} Iter {self.g_step_op.eval(session=self.sess)}" \
                        #          f" [Train]: loss={loss:.2f}, epe={epe:.2f}, lr={lr:.6f}," \
                        #          f" samples/sec={samples_per_sec:.1f}, sec/step={sec_per_step:.3f}," \
                        #          f" eta={datetime.timedelta(seconds=eta)}"
                        # print(status)

                    # Reset batch progress trackers
                    train_loss, train_epe, duration = [], [], []

                # Show progress on validation ds, if requested
                if val_batch_size > 0 and step % self.opts['val_step'] == 0:

                    val_loss, val_epe = [], []
                    rounds, _ = divmod(val_batch_size, batch_size * self.num_gpus)
                    for _round in range(rounds):
                        if self.opts['use_tf_data'] is True:
                            x, y, _, _ = self.sess.run(val_next_batch)
                        else:
                            # Get a batch of val samples and make them conform to the network's requirements
                            x, y, _ = self.ds.next_batch(batch_size * self.num_gpus, split='val')
                            # x: [batch_size * self.num_gpus,2,H,W,3] uint8 y: [batch_size,H,W,2] float32
                        x_adapt, _ = self.adapt_x(x)
                        y_adapt, _ = self.adapt_y(y)
                        # x_adapt: [batch_size * self.num_gpus,2,H,W,3] float32 y_adapt: [batch_size,H,W,2] float32

                        # Run the val samples through the network (loss and error rate ops)
                        feed_dict = {self.x_tnsr: x_adapt, self.y_tnsr: y_adapt}
                        y_hat = self.sess.run(self.y_hat_val_tnsr, feed_dict=feed_dict)
                        loss, epe = self.postproc_y_hat_val(y_hat)
                        val_loss.append(loss), val_epe.append(epe)

                    # Send the results to tensorboard
                    loss, epe = np.mean(val_loss), np.mean(val_epe)
                    ranking_value = epe
                    self.tb_val.log_scalar("losses/loss", loss, step)
                    self.tb_val.log_scalar("metrics/epe", epe, step)

                    # Print results, if requested
                    if self.opts['verbose']:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        # status = f"{ts} Iter {self.g_step_op.eval(session=self.sess)} [Val]: loss={loss:.2f}, epe={epe:.2f}"
                        # print(status)

                # Save a checkpoint every snapshot_step
                if step % self.opts['snapshot_step'] == 0 or step == self.opts['max_steps']:

                    # Log evolution of test images to Tensorboard, if requested
                    if self.opts['tb_test_imgs'] is not None:
                        # Get a batch of test samples and make them conform to the network's requirements
                        if tb_test_loaded is False:
                            x_tb_test, IDs_tb_test = self.ds.get_samples(
                                batch_size * self.num_gpus, split='test', simple_IDs=True)
                            x_tb_test_adapt, _ = self.adapt_x(x_tb_test)
                            # IDs_tb_test = self.ds.simplify_IDs(x_IDs)
                            tb_test_loaded = True

                        # Run the test samples through the network
                        feed_dict = {self.x_tnsr: x_tb_test_adapt}
                        y_hat = self.sess.run(self.y_hat_test_tnsr, feed_dict=feed_dict)
                        pred_flows, pred_flows_pyr = self.postproc_y_hat_test(y_hat)

                        # Only show batch_size results, no matter what the GPU count is
                        pred_flows, pred_flows_pyr = pred_flows[0:batch_size], pred_flows_pyr[0:batch_size]

                        # Send the results to tensorboard
                        if self.opts['tb_test_imgs'] == 'top_flow':
                            self.tb_test.log_imgs_w_flows('test/{}_flows', x_tb_test, None, 0, pred_flows,
                                                          None, step, IDs_tb_test)
                        else:
                            self.tb_test.log_imgs_w_flows('test/{}_flows_pyr', x_tb_test, pred_flows_pyr,
                                                          self.opts['pyr_lvls'] - self.opts['flow_pred_lvl'], pred_flows,
                                                          None, step, IDs_tb_test)

                    # Log evolution of val images, if requested
                    if self.opts['tb_val_imgs'] is not None:
                        # Get a batch of val samples and make them conform to the network's requirements
                        if tb_val_loaded is False:
                            x_tb_val, y_tb_val, IDs_tb_val = self.ds.get_samples(
                                batch_size * self.num_gpus, split='val', simple_IDs=True)
                            x_tb_val_adapt, _ = self.adapt_x(x_tb_val)
                            # IDs_tb_val = self.ds.simplify_IDs(x_IDs)
                            tb_val_loaded = True

                        # Run the val samples through the network (top flow and pyramid)
                        feed_dict = {self.x_tnsr: x_tb_val_adapt}
                        y_hat = self.sess.run(self.y_hat_test_tnsr, feed_dict=feed_dict)
                        pred_flows, pred_flows_pyr = self.postproc_y_hat_test(y_hat)

                        # Only show batch_size results, no matter what the GPU count is
                        x_tb_val, y_tb_val = x_tb_val[0:batch_size], y_tb_val[0:batch_size]
                        IDs_tb_val = IDs_tb_val[0:batch_size]
                        pred_flows, pred_flows_pyr = pred_flows[0:batch_size], pred_flows_pyr[0:batch_size]

                        # Send the results to tensorboard
                        if self.opts['tb_val_imgs'] == 'top_flow':
                            self.tb_val.log_imgs_w_flows('val/{}_flows', x_tb_val, None, 0, pred_flows,
                                                         y_tb_val, step, IDs_tb_val)
                        else:
                            self.tb_val.log_imgs_w_flows('val/{}_flows_pyr', x_tb_val[0:batch_size], pred_flows_pyr,
                                                         self.opts['pyr_lvls'] - self.opts['flow_pred_lvl'], pred_flows,
                                                         y_tb_val, step, IDs_tb_val)

                    # Save model
                    self.save_ckpt(ranking_value)

                step += 1

            if self.opts['verbose']:
                print("... done training.")

    ###
    # Evaluation helpers
    ###
    def setup_metrics_ops(self):
        """Setup metrics computations. Use the endpoint error metric to track progress.
        Note that, if the label flows come back from the network padded, it isn't a fair assessment of the performance
        of the model if we also measure the EPE in the padded area. This area is to be cropped out before returning
        the predicted flows to the caller, so exclude that area when computing the performance metric.
        """
        # Have the samples been padded to the nn's requirements? If so, crop flows back to original size.
        y_tnsr, flow_pred_tnsr = self.y_tnsr, self.flow_pred_tnsr
        if self.opts['adapt_info'] is not None:
            y_tnsr = y_tnsr[:, 0:self.opts['adapt_info'][1], 0:self.opts['adapt_info'][2], :]
            flow_pred_tnsr = flow_pred_tnsr[:, 0:self.opts['adapt_info'][1], 0:self.opts['adapt_info'][2], :]

        if self.opts['sparse_gt_flow'] is True:
            # Find the location of the zerod-out flows in the gt
            zeros_loc = tf.logical_and(tf.equal(y_tnsr[:, :, :, 0], 0.0), tf.equal(y_tnsr[:, :, :, 1], 0.0))
            zeros_loc = tf.expand_dims(zeros_loc, -1)

            # Zero out flow predictions at the same location so we only compute the EPE at the sparse flow points
            flow_pred_tnsr = tf_where(zeros_loc, tf.zeros_like(flow_pred_tnsr), flow_pred_tnsr)

        if self.mode in ['train_noval', 'train_with_val']:
            # In online evaluation mode, we only care about the average loss and metric for the batch:
            self.metric_op = tf.reduce_mean(tf.norm(y_tnsr - flow_pred_tnsr, ord=2, axis=3))

        if self.mode in ['val', 'val_notrain']:
            # In offline evaluation mode, we actually care about each individual prediction and metric -> axis=(1, 2)
            self.metric_op = tf.reduce_mean(tf.norm(y_tnsr - flow_pred_tnsr, ord=2, axis=3), axis=(1, 2))

    def eval(self, metric_name=None, save_preds=False):
        """Evaluation loop. Test the trained model on the validation split of the dataset.
        Args:
            save_preds: if True, the predictions are saved to disk
        Returns:
            Aaverage score for the entire dataset, a panda df with individual scores for further error analysis
        """
        with self.graph.as_default():
            # Use feed_dict from np or with tf.data.Dataset?
            batch_size = self.opts['batch_size']
            if self.opts['use_tf_data'] is True:
                # Create tf.data.Dataset manager
                tf_ds = self.ds.get_tf_ds(batch_size=batch_size, split='val', sess=self.sess)

                # Ops for initializing the iterator
                next_batch = tf_ds.make_one_shot_iterator().get_next()

            # Store results in a dataframe
            if metric_name is None:
                metric_name = 'Score'
            df = pd.DataFrame(columns=['ID', metric_name, 'Duration', 'Avg_Flow_Mag', 'Max_Flow_Mag'])

            # Chunk dataset
            rounds, rounds_left = divmod(self.ds.val_size, batch_size)
            if rounds_left:
                rounds += 1

            # Loop through samples and track their model performance
            desc = 'Measuring '+metric_name+' and saving preds' if save_preds else 'Measuring '+metric_name
            idx = 0
            for _round in trange(rounds, ascii=True, ncols=100, desc=desc):

                # Fetch and adapt sample
                if self.opts['use_tf_data'] is True:
                    x, y, y_hat_paths, IDs = self.sess.run(next_batch)
                    y_hat_paths = [y_hat_path.decode() for y_hat_path in y_hat_paths]
                    IDs = [ID.decode() for ID in IDs]
                else:
                    # Get a batch of samples and make them conform to the network's requirements
                    x, y, y_hat_paths, IDs = self.ds.next_batch(batch_size, split='val_with_pred_paths')
                    # x: [batch_size * self.num_gpus,2,H,W,3] uint8 y: [batch_size,H,W,2] float32

                x_adapt, _ = self.adapt_x(x)
                y_adapt, y_adapt_info = self.adapt_y(y)
                # x_adapt: [batch_size * self.num_gpus,2,H,W,3] float32 y_adapt: [batch_size,H,W,2] float32

                # Run the sample through the network (metric op)
                feed_dict = {self.x_tnsr: x_adapt, self.y_tnsr: y_adapt}
                start_time = time.time()
                y_hat = self.sess.run(self.y_hat_val_tnsr, feed_dict=feed_dict)
                duration = time.time() - start_time
                y_hats, metrics = self.postproc_y_hat_val(y_hat, y_adapt_info)

                # Save the individual results in df
                duration /= batch_size
                for y_hat, metric, y_hat_path, ID in zip(y_hats, metrics, y_hat_paths, IDs):
                    _, flow_mag_avg, flow_mag_max = flow_mag_stats(y_hat)
                    df.loc[idx] = (ID, metric, duration, flow_mag_avg, flow_mag_max)
                    if save_preds:
                        flow_write(y_hat, y_hat_path)
                        info="{"+metric_name+"}={"+metric+":.2f}"
                        flow_write_as_png(y_hat, y_hat_path.replace('.flo', '.png'), info=info)
                    idx += 1

            # Compute stats
            avg_metric, avg_duration = df.loc[:, metric_name].mean(), df.loc[:, 'Duration'].mean()

        # print(self.unique_y_shapes)
        return avg_metric, avg_duration, df

    ###
    # Inference helpers
    ###
    def predict(self, return_preds=False, save_preds=True):
        """Inference loop. Run the trained model on the test split of the dataset.
        The data samples are provided by the OpticalFlowDataset object associated with this ModelPWCNet instance.
        To predict flows for image pairs not provided by such object, use predict_from_img_pairs() instead.
        Args:
            return_preds: if True, the predictions are returned to the caller in list([2, H, W, 3]) format.
            save_preds: if True, the predictions are saved to disk in .flo and .png format
        Returns:
            if return_preds is True, the predictions and their IDs are returned (might require a lot of RAM...)
            if return_preds is False, return None
        """
        with self.graph.as_default():
            # Use feed_dict from np or with tf.data.Dataset?
            batch_size = self.opts['batch_size']
            if self.opts['use_tf_data'] is True:
                # Create tf.data.Dataset manager
                tf_ds = self.ds.get_tf_ds(batch_size=batch_size, split='test', sess=self.sess)

                # Ops for initializing the iterator
                next_batch = tf_ds.make_one_shot_iterator().get_next()

            # Chunk dataset
            rounds, rounds_left = divmod(self.ds.tst_size, batch_size)
            if rounds_left:
                rounds += 1

            # Loop through input samples and run inference on them
            if return_preds is True:
                preds, ids = [], []
            desc = 'Predicting flows and saving preds' if save_preds else 'Predicting flows'
            for _round in trange(rounds, ascii=True, ncols=100, desc=desc):

                # Fetch and adapt sample
                if self.opts['use_tf_data'] is True:
                    x, y_hat_paths, IDs = self.sess.run(next_batch)
                    y_hat_paths = [y_hat_path.decode() for y_hat_path in y_hat_paths]
                    IDs = [ID.decode() for ID in IDs]
                else:
                    # Get a batch of samples and make them conform to the network's requirements
                    x, y_hat_paths, IDs = self.ds.next_batch(batch_size, split='test_with_pred_paths')
                    # x: [batch_size,2,H,W,3] uint8; x_adapt: [batch_size,2,H,W,3] float32

                x_adapt, x_adapt_info = self.adapt_x(x)
                if x_adapt_info is not None:
                    y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
                else:
                    y_adapt_info = None

                # Run the sample through the network
                feed_dict = {self.x_tnsr: x_adapt}
                y_hat = self.sess.run(self.y_hat_test_tnsr, feed_dict=feed_dict)
                y_hats, _ = self.postproc_y_hat_test(y_hat, y_adapt_info)

                # Save the predicted flows to disk, if requested
                for y_hat, y_hat_path, ID in zip(y_hats, y_hat_paths, IDs):
                    if return_preds is True:
                        preds.append(y_hat)
                        ids.append(ID)
                    if save_preds is True:
                        flow_write(y_hat, y_hat_path)
                        flow_write_as_png(y_hat, y_hat_path.replace('.flo', '.png'))

        if return_preds is True:
            return preds[0:self.ds.tst_size], ids[0:self.ds.tst_size]
        else:
            return None

    def predict_from_img_pairs(self, img_pairs, batch_size=1, verbose=False):
        """Inference loop. Run inference on a list of image pairs.
        Args:
            img_pairs: list of image pairs/tuples in list((img_1, img_2),...,(img_n, img_nplusone)) format.
            batch_size: size of the batch to process (all images must have the same dimension, if batch_size>1)
            verbose: if True, show progress bar
        Returns:
            Predicted flows in list format
        """
        with self.graph.as_default():
            # Chunk image pair list
            batch_size = self.opts['batch_size']
            test_size = len(img_pairs)
            rounds, rounds_left = divmod(test_size, batch_size)
            if rounds_left:
                rounds += 1

            # Loop through input samples and run inference on them
            preds, test_ptr = [], 0
            rng = trange(rounds, ascii=True, ncols=100, desc='Predicting flows') if verbose else range(rounds)
            for _round in rng:
                # In batch mode, make sure to wrap around if there aren't enough input samples to process
                if test_ptr + batch_size < test_size:
                    new_ptr = test_ptr + batch_size
                    indices = list(range(test_ptr, test_ptr + batch_size))
                else:
                    new_ptr = (test_ptr + batch_size) % test_size
                    indices = list(range(test_ptr, test_size)) + list(range(0, new_ptr))
                test_ptr = new_ptr

                # Repackage input image pairs as np.ndarray
                x = np.array([img_pairs[idx] for idx in indices])

                # Make input samples conform to the network's requirements
                # x: [batch_size,2,H,W,3] uint8; x_adapt: [batch_size,2,H,W,3] float32
                x_adapt, x_adapt_info = self.adapt_x(x)
                if x_adapt_info is not None:
                    y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
                else:
                    y_adapt_info = None

                # Run the adapted samples through the network
                feed_dict = {self.x_tnsr: x_adapt}
                y_hat = self.sess.run(self.y_hat_test_tnsr, feed_dict=feed_dict)
                y_hats, _ = self.postproc_y_hat_test(y_hat, y_adapt_info)

                # Return flat list of predicted labels
                for y_hat in y_hats:
                    preds.append(y_hat)

        return preds[0:test_size]

    def predict_from_img_pairs_tf(self, img_pairs, batch_size=1, verbose=False):
        with self.graph.as_default():
            self.x_tnsr = img_pairs
        return self.y_hat_test_tnsr[0]

    ###
    # PWC-Net pyramid helpers
    ###
    def extract_features(self, x_tnsr, name='featpyr'):
        """Extract pyramid of features
        Args:
            x_tnsr: Input tensor (input pair of images in [batch_size, 2, H, W, 3] format)
            name: Variable scope name
        Returns:
            c1, c2: Feature pyramids
        Ref:
            Per page 3 of paper, section "Feature pyramid extractor," given two input images I1 and I2, we generate
            L-level pyramids of feature representations, with the bottom (zeroth) level being the input images,
            i.e., Ct<sup>0</sup> = It. To generate feature representation at the l-th layer, Ct<sup>l</sup>, we use
            layers of convolutional filters to downsample the features at the (l??)th pyramid level, Ct<sup>l-1</sup>,
            by a factor of 2. From the first to the sixth levels, the number of feature channels are respectively
            16, 32, 64, 96, 128, and 196. Also see page 15 of paper for a rendering of the network architecture.
            Per page 15, individual images of the image pair are encoded using the same Siamese network. Each
            convolution is followed by a leaky ReLU unit. The convolutional layer and the x2 downsampling layer at
            each level is implemented using a single convolutional layer with a stride of 2.
            Note that Figure 4 on page 15 differs from the PyTorch implementation in two ways:
            - It's missing a convolution layer at the end of each conv block
            - It shows a number of filters of 192 (instead of 196) at the end of the last conv block
        Ref PyTorch code:
            def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True), nn.LeakyReLU(0.1))
            [...]
            self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
            self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
            self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
            self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
            self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
            self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
            self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
            self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
            self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
            self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
            self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
            self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
            self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
            self.conv5aa = conv(128,128, kernel_size=3, stride=1)
            self.conv5b  = conv(128,128, kernel_size=3, stride=1)
            self.conv6aa = conv(128,196, kernel_size=3, stride=2)
            self.conv6a  = conv(196,196, kernel_size=3, stride=1)
            self.conv6b  = conv(196,196, kernel_size=3, stride=1)
            [...]
            c11 = self.conv1b(self.conv1aa(self.conv1a(im1))) # Higher-res
            c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
            c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
            c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
            c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
            c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
            c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
            c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
            c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
            c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
            c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
            c26 = self.conv6b(self.conv6a(self.conv6aa(c25))) # Lower-res
        Ref Caffee code:
            https://github.com/NVlabs/PWC-Net/blob/438ca897ae77e08f419ddce5f0d7fa63b0a27a77/Caffe/model/train.prototxt#L314-L1141
        """
        assert(1 <= self.opts['pyr_lvls'] <= 6)
        # Make the feature pyramids 1-based for better readability down the line
        num_chann = [None, 16, 32, 64, 96, 128, 196]
        c1, c2 = [None], [None]
        init = tf.keras.initializers.he_normal()
        with tf.variable_scope(name):
            for pyr, x, reuse, name in zip([c1, c2], [x_tnsr[:, 0], x_tnsr[:, 1]], [None, True], ['c1', 'c2']):
                for lvl in range(1, self.opts['pyr_lvls'] + 1):
                    # tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name, reuse)
                    # reuse is set to True because we want to learn a single set of weights for the pyramid
                    # kernel_initializer = 'he_normal' or tf.keras.initializers.he_normal(seed=None)
                    f = num_chann[lvl]
                    x = tf.layers.conv2d(x, f, 3, 2, 'same', kernel_initializer=init, name='conv'+str(lvl)+'a', reuse=reuse)
                    x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}a') # default alpha is 0.2 for TF
                    x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name='conv'+str(lvl)+'aa', reuse=reuse)
                    x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}aa')
                    x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name='conv'+str(lvl)+'b', reuse=reuse)
                    x = tf.nn.leaky_relu(x, alpha=0.1, name=name+str(lvl))
                    pyr.append(x)
        return c1, c2

    ###
    # PWC-Net warping helpers
    ###
    def warp(self, c2, sc_up_flow, lvl, name='warp'):
        """Warp a level of Image1's feature pyramid using the upsampled flow at level+1 of Image2's pyramid.
        Args:
            c2: The level of the feature pyramid of Image2 to warp
            sc_up_flow: Scaled and upsampled estimated optical flow (from Image1 to Image2) used for warping
            lvl: Index of that level
            name: Op scope name
        Ref:
            Per page 4 of paper, section "Warping layer," at the l-th level, we warp features of the second image toward
            the first image using the x2 upsampled flow from the l+1th level:
                C1w<sup>l</sup>(x) = C2<sup>l</sup>(x + Up2(w<sup>l+1</sup>)(x))
            where x is the pixel index and the upsampled flow Up2(w<sup>l+1</sup>) is set to be zero at the top level.
            We use bilinear interpolation to implement the warping operation and compute the gradients to the input
            CNN features and flow for backpropagation according to E. Ilg's FlowNet 2.0 paper.
            For non-translational motion, warping can compensate for some geometric distortions and put image patches
            at the right scale.
            Per page 3 of paper, section "3. Approach," the warping and cost volume layers have no learnable parameters
            and, hence, reduce the model size.
        Ref PyTorch code:
            # warp an image/tensor (im2) back to im1, according to the optical flow
            # x: [B, C, H, W] (im2)
            # flo: [B, 2, H, W] flow
            def warp(self, x, flo):
                B, C, H, W = x.size()
                # mesh grid
                xx = torch.arange(0, W).view(1,-1).repeat(H,1)
                yy = torch.arange(0, H).view(-1,1).repeat(1,W)
                xx = xx.view(1,1,H,W).repeat(B,1,1,1)
                yy = yy.view(1,1,H,W).repeat(B,1,1,1)
                grid = torch.cat((xx,yy),1).float()
                if x.is_cuda:
                    grid = grid.cuda()
                vgrid = Variable(grid) + flo
                # scale grid to [-1,1]
                vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
                vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
                vgrid = vgrid.permute(0,2,3,1)
                output = nn.functional.grid_sample(x, vgrid)
                mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
                mask = nn.functional.grid_sample(mask, vgrid)
                mask[mask<0.9999] = 0
                mask[mask>0] = 1
                return output*mask
            [...]
            warp5 = self.warp(c25, up_flow6*0.625)
            warp4 = self.warp(c24, up_flow5*1.25)
            warp3 = self.warp(c23, up_flow4*2.5)
            warp2 = self.warp(c22, up_flow3*5.0)
        Ref TF documentation:
            tf.contrib.image.dense_image_warp(image, flow, name='dense_image_warp')
            https://www.tensorflow.org/api_docs/python/tf/contrib/image/dense_image_warp
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/image/python/kernel_tests/dense_image_warp_test.py
        Other implementations:
            https://github.com/bryanyzhu/deepOF/blob/master/flyingChairsWrapFlow.py
            https://github.com/bryanyzhu/deepOF/blob/master/ucf101wrapFlow.py
            https://github.com/rajat95/Optical-Flow-Warping-Tensorflow/blob/master/warp.py
        """
        op_name = name+str(lvl)
        if self.dbg:
            msg = 'Adding '+op_name+' with inputs '+c2.op.name+' and '+sc_up_flow.op.name
            print(msg)
        with tf.name_scope(name):
            return dense_image_warp(c2, sc_up_flow, name=op_name)

    def deconv(self, x, lvl, name='up_flow'):
        """Upsample, not using a bilinear filter, but rather learn the weights of a conv2d_transpose op filters.
        Args:
            x: Level features or flow to upsample
            lvl: Index of that level
            name: Op scope name
        Ref PyTorch code:
            def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
                return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)
            [...]
            self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
            ...
            self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
            ...
            self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
            ...
            self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1)
            ...
            self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            [...]
            up_flow6 = self.deconv6(flow6)
            up_feat6 = self.upfeat6(x)
            ...
            up_flow5 = self.deconv5(flow5)
            up_feat5 = self.upfeat5(x)
            ...
            up_flow4 = self.deconv4(flow4)
            up_feat4 = self.upfeat4(x)
            ...
            up_flow3 = self.deconv3(flow3)
            up_feat3 = self.upfeat3(x)
        """
        op_name = name+str(lvl)
        if self.dbg:
            print('Adding '+op_name+' with input '+x.op.name)
        with tf.variable_scope('upsample'):
            # tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name)
            return tf.layers.conv2d_transpose(x, 2, 4, 2, 'same', name=op_name)

    ###
    # Cost Volume helpers
    ###
    def corr(self, c1, warp, lvl, name='corr'):
        """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
        Args:
            c1: The level of the feature pyramid of Image1
            warp: The warped level of the feature pyramid of image22
            lvl: Index of that level
            name: Op scope name
        Ref:
            Per page 3 of paper, section "Cost Volume," a cost volume stores the data matching costs for associating
            a pixel from Image1 with its corresponding pixels in Image2. Most traditional optical flow techniques build
            the full cost volume at a single scale, which is both computationally expensive and memory intensive. By
            contrast, PWC-Net constructs a partial cost volume at multiple pyramid levels.
            The matching cost is implemented as the correlation between features of the first image and warped features
            of the second image:
                CV<sup>l</sup>(x1,x2) = (C1<sup>l</sup>(x1))<sup>T</sup> . Cw<sup>l</sup>(x2) / N
            where where T is the transpose operator and N is the length of the column vector C1<sup>l</sup>(x1).
            For an L-level pyramid, we only need to compute a partial cost volume with a limited search range of d
            pixels. A one-pixel motion at the top level corresponds to 2**(L??) pixels at the full resolution images.
            Thus we can set d to be small, e.g. d=4. The dimension of the 3D cost volume is d**2 ? Hl ? Wl, where Hl
            and Wl denote the height and width of the L-th pyramid level, respectively.
            Per page 3 of paper, section "3. Approach," the warping and cost volume layers have no learnable parameters
            and, hence, reduce the model size.
            Per page 5 of paper, section "Implementation details," we use a search range of 4 pixels to compute the
            cost volume at each level.
        Ref PyTorch code:
        from correlation_package.modules.corr import Correlation
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        [...]
        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)
        ...
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        ...
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        ...
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        ...
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        """
        op_name = 'corr'+str(lvl)
        if self.dbg:
            print('Adding '+op_name+' with inputs '+c1.op.name+' and '+warp.op.name)
        with tf.name_scope(name):
            return cost_volume(c1, warp, self.opts['search_range'], op_name)

    ###
    # Optical flow estimator helpers
    ###
    def predict_flow(self, corr, c1, up_flow, up_feat, lvl, name='predict_flow'):
        """Estimate optical flow.
        Args:
            corr: The cost volume at level lvl
            c1: The level of the feature pyramid of Image1
            up_flow: An upsampled version of the predicted flow from the previous level
            up_feat: An upsampled version of the features that were used to generate the flow prediction
            lvl: Index of the level
            name: Op scope name
        Args:
            upfeat: The features used to generate the predicted flow
            flow: The predicted flow
        Ref:
            Per page 4 of paper, section "Optical flow estimator," the optical flow estimator is a multi-layer CNN. Its
            input are the cost volume, features of the first image, and upsampled optical flow and its output is the
            flow w<sup>l</sup> at the l-th level. The numbers of feature channels at each convolutional layers are
            respectively 128, 128, 96, 64, and 32, which are kept fixed at all pyramid levels. The estimators at
            different levels have their own parameters instead of sharing the same parameters. This estimation process
            is repeated until the desired level, l0.
            Per page 5 of paper, section "Implementation details," we use a 7-level pyramid and set l0 to be 2, i.e.,
            our model outputs a quarter resolution optical flow and uses bilinear interpolation to obtain the
            full-resolution optical flow.
            The estimator architecture can be enhanced with DenseNet connections. The inputs to every convolutional
            layer are the output of and the input to its previous layer. DenseNet has more direct connections than
            traditional layers and leads to significant improvement in image classification.
            Note that we do not use DenseNet connections in this implementation because a) they increase the size of the
            model, and, b) per page 7 of paper, section "Optical flow estimator," removing the DenseNet connections
            results in higher training error but lower validation errors when the model is trained on FlyingChairs
            (that being said, after the model is fine-tuned on FlyingThings3D, DenseNet leads to lower errors).
        Ref PyTorch code:
            def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.1))
            def predict_flow(in_planes):
                return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)
            [...]
            nd = (2*md+1)**2
            dd = np.cumsum([128,128,96,64,32])
            od = nd
            self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
            self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
            self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
            self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
            self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
            self.predict_flow6 = predict_flow(od+dd[4])
            [...]
            od = nd+128+4
            self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
            self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
            self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
            self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
            self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
            self.predict_flow5 = predict_flow(od+dd[4])
            [...]
            od = nd+96+4
            self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
            self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
            self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
            self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
            self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
            self.predict_flow4 = predict_flow(od+dd[4])
            [...]
            od = nd+64+4
            self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
            self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
            self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
            self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
            self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
            self.predict_flow3 = predict_flow(od+dd[4])
            [...]
            od = nd+32+4
            self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
            self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
            self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
            self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
            self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
            self.predict_flow2 = predict_flow(od+dd[4])
            [...]
            self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
            self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
            self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
            self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
            self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
            self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
            self.dc_conv7 = predict_flow(32)
            [...]
            x = torch.cat((self.conv6_0(corr6), corr6),1)
            x = torch.cat((self.conv6_1(x), x),1)
            x = torch.cat((self.conv6_2(x), x),1)
            x = torch.cat((self.conv6_3(x), x),1)
            x = torch.cat((self.conv6_4(x), x),1)
            flow6 = self.predict_flow6(x)
            ...
            x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
            x = torch.cat((self.conv5_0(x), x),1)
            x = torch.cat((self.conv5_1(x), x),1)
            x = torch.cat((self.conv5_2(x), x),1)
            x = torch.cat((self.conv5_3(x), x),1)
            x = torch.cat((self.conv5_4(x), x),1)
            flow5 = self.predict_flow5(x)
            ...
            x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
            x = torch.cat((self.conv4_0(x), x),1)
            x = torch.cat((self.conv4_1(x), x),1)
            x = torch.cat((self.conv4_2(x), x),1)
            x = torch.cat((self.conv4_3(x), x),1)
            x = torch.cat((self.conv4_4(x), x),1)
            flow4 = self.predict_flow4(x)
            ...
            x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
            x = torch.cat((self.conv3_0(x), x),1)
            x = torch.cat((self.conv3_1(x), x),1)
            x = torch.cat((self.conv3_2(x), x),1)
            x = torch.cat((self.conv3_3(x), x),1)
            x = torch.cat((self.conv3_4(x), x),1)
            flow3 = self.predict_flow3(x)
            ...
            x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
            x = torch.cat((self.conv2_0(x), x),1)
            x = torch.cat((self.conv2_1(x), x),1)
            x = torch.cat((self.conv2_2(x), x),1)
            x = torch.cat((self.conv2_3(x), x),1)
            x = torch.cat((self.conv2_4(x), x),1)
            flow2 = self.predict_flow2(x)
        """
        op_name = 'flow'+str(lvl)
        init = tf.keras.initializers.he_normal()
        with tf.variable_scope(name):
            if c1 is None and up_flow is None and up_feat is None:
                if self.dbg:
                    print('Adding '+op_name+' with input '+corr.op.name)
                x = corr
            else:
                if self.dbg:
                    msg = 'Adding '+op_name+' with inputs '+corr.op.name+', '+c1.op.name+', '+up_flow.op.name+', '+up_feat.op.name
                    print(msg)
                x = tf.concat([corr, c1, up_flow, up_feat], axis=3)

            conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name='conv'+str(lvl)+'_0')
            act = tf.nn.leaky_relu(conv, alpha=0.1)  # default alpha is 0.2 for TF
            x = tf.concat([act, x], axis=3) if self.opts['use_dense_cx'] else act

            conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name='conv'+str(lvl)+'_1')
            act = tf.nn.leaky_relu(conv, alpha=0.1)
            x = tf.concat([act, x], axis=3) if self.opts['use_dense_cx'] else act

            conv = tf.layers.conv2d(x, 96, 3, 1, 'same', kernel_initializer=init, name='conv'+str(lvl)+'_2')
            act = tf.nn.leaky_relu(conv, alpha=0.1)
            x = tf.concat([act, x], axis=3) if self.opts['use_dense_cx'] else act

            conv = tf.layers.conv2d(x, 64, 3, 1, 'same', kernel_initializer=init, name='conv'+str(lvl)+'_3')
            act = tf.nn.leaky_relu(conv, alpha=0.1)
            x = tf.concat([act, x], axis=3) if self.opts['use_dense_cx'] else act

            conv = tf.layers.conv2d(x, 32, 3, 1, 'same', kernel_initializer=init, name='conv'+str(lvl)+'_4')
            act = tf.nn.leaky_relu(conv, alpha=0.1)  # will also be used as an input by the context network
            upfeat = tf.concat([act, x], axis=3, name='upfeat'+str(lvl)) if self.opts['use_dense_cx'] else act

            flow = tf.layers.conv2d(upfeat, 2, 3, 1, 'same', name=op_name)

            return upfeat, flow

    ###
    # PWC-Net context network helpers
    ###
    def refine_flow(self, feat, flow, lvl, name='ctxt'):
        """Post-ptrocess the estimated optical flow using a "context" nn.
        Args:
            feat: Features of the second-to-last layer from the optical flow estimator
            flow: Estimated flow to refine
            lvl: Index of the level
            name: Op scope name
        Ref:
            Per page 4 of paper, section "Context network," traditional flow methods often use contextual information
            to post-process the flow. Thus we employ a sub-network, called the context network, to effectively enlarge
            the receptive field size of each output unit at the desired pyramid level. It takes the estimated flow and
            features of the second last layer from the optical flow estimator and outputs a refined flow.
            The context network is a feed-forward CNN and its design is based on dilated convolutions. It consists of
            7 convolutional layers. The spatial kernel for each convolutional layer is 3?3. These layers have different
            dilation constants. A convolutional layer with a dilation constant k means that an input unit to a filter
            in the layer are k-unit apart from the other input units to the filter in the layer, both in vertical and
            horizontal directions. Convolutional layers with large dilation constants enlarge the receptive field of
            each output unit without incurring a large computational burden. From bottom to top, the dilation constants
            are 1, 2, 4, 8, 16, 1, and 1.
        Ref PyTorch code:
            def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
                return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.1))
            def predict_flow(in_planes):
                return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)
            [...]
            self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
            self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
            self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
            self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
            self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
            self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
            self.dc_conv7 = predict_flow(32)
            [...]
            x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
            x = torch.cat((self.conv2_0(x), x),1)
            x = torch.cat((self.conv2_1(x), x),1)
            x = torch.cat((self.conv2_2(x), x),1)
            x = torch.cat((self.conv2_3(x), x),1)
            x = torch.cat((self.conv2_4(x), x),1)
            flow2 = self.predict_flow2(x)
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
            flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        """
        op_name = 'refined_flow'+str(lvl)
        if self.dbg:
            print('Adding '+op_name+' sum of dc_convs_chain('+feat.op.name+') with '+flow.op.name)
        init = tf.keras.initializers.he_normal()
        with tf.variable_scope(name):
            x = tf.layers.conv2d(feat, 128, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv'+str(lvl)+'1')
            x = tf.nn.leaky_relu(x, alpha=0.1)  # default alpha is 0.2 for TF
            x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=2, kernel_initializer=init, name='dc_conv'+str(lvl)+'2')
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=4, kernel_initializer=init, name='dc_conv'+str(lvl)+'3')
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 96, 3, 1, 'same', dilation_rate=8, kernel_initializer=init, name='dc_conv'+str(lvl)+'4')
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 64, 3, 1, 'same', dilation_rate=16, kernel_initializer=init, name='dc_conv'+str(lvl)+'5')
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 32, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv'+str(lvl)+'6')
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 2, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv'+str(lvl)+'7')

            return tf.add(flow, x, name=op_name)

    ###
    # PWC-Net nn builder
    ###
    def nn(self, x_tnsr, name='pwcnet', reuse=False):
        """Defines and connects the backbone neural nets
        Args:
            inputs: TF placeholder that contains the input frame pairs in [batch_size, 2, H, W, 3] format
            name: Name of the nn
        Returns:
            net: Output tensors of the backbone network
        Ref:
            RE: the scaling of the upsampled estimated optical flow, per page 5, section "Implementation details," we
            do not further scale the supervision signal at each level, the same as the FlowNet paper. As a result, we
            need to scale the upsampled flow at each pyramid level for the warping layer. For example, at the second
            level, we scale the upsampled flow from the third level by a factor of 5 (=20/4) before warping features
            of the second image.
        Based on:
            - https://github.com/daigo0927/PWC-Net_tf/blob/master/model.py
            Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
            MIT License
        """
        with tf.variable_scope(name, reuse=reuse):

            # Extract pyramids of CNN features from both input images (1-based lists))
            c1, c2 = self.extract_features(x_tnsr)

            flow_pyr = []

            for lvl in range(self.opts['pyr_lvls'], self.opts['flow_pred_lvl'] - 1, -1):

                if lvl == self.opts['pyr_lvls']:
                    # Compute the cost volume
                    corr = self.corr(c1[lvl], c2[lvl], lvl)

                    # Estimate the optical flow
                    upfeat, flow = self.predict_flow(corr, None, None, None, lvl)
                else:
                    # Warp level of Image1's using the upsampled flow
                    scaler = 20. / 2**lvl  # scaler values are 0.625, 1.25, 2.5, 5.0
                    warp = self.warp(c2[lvl], up_flow * scaler, lvl)

                    # Compute the cost volume
                    corr = self.corr(c1[lvl], warp, lvl)

                    # Estimate the optical flow
                    upfeat, flow = self.predict_flow(corr, c1[lvl], up_flow, up_feat, lvl)

                _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(c1[lvl]))

                if lvl != self.opts['flow_pred_lvl']:
                    if self.opts['use_res_cx']:
                        flow = self.refine_flow(upfeat, flow, lvl)

                    # Upsample predicted flow and the features used to compute predicted flow
                    flow_pyr.append(flow)

                    up_flow = self.deconv(flow, lvl, 'up_flow')
                    up_feat = self.deconv(upfeat, lvl, 'up_feat')
                else:
                    # Refine the final predicted flow
                    flow = self.refine_flow(upfeat, flow, lvl)
                    flow_pyr.append(flow)

                    # Upsample the predicted flow (final output) to match the size of the images
                    scaler = 2**self.opts['flow_pred_lvl']
                    if self.dbg:
                        print('Upsampling '+flow.op.name+' by '+scaler+' in each dimension.')
                    size = (lvl_height * scaler, lvl_width * scaler)
                    flow_pred = tf.image.resize_bilinear(flow, size, name="flow_pred", align_corners=True) * scaler
                    break

            return flow_pred, flow_pyr