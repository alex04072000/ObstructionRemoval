"""
model_base.py

Model base class.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ckpt_mgr import BestCheckpointSaver
from logger import OptFlowTBLogger
from dataset_base import _DBG_TRAIN_VAL_TEST_SETS
from lr import lr_multisteps_long, lr_multisteps_fine, lr_cyclic_long, lr_cyclic_fine
from mixed_precision import float32_variable_storage_getter

_DEBUG_USE_REF_IMPL = False


class ModelBase:
    def __init__(self, name='base', mode='train_with_val', session=None, options=None):
        """Initialize the ModelBase object
        Args:
            mode: Must be in ['train_noval', 'val', 'train_with_val', 'test']
            session: optional TF session
            options: see _DEFAULT_PWCNET_TRAIN_OPTIONS comments
        Mote:
            As explained [here](https://stackoverflow.com/a/36282423), you don't need to use with blocks if you only
            have one default graph and one default session. However, we sometimes create notebooks where we pit the
            performance of models against each other. Because of that, we need the with block.
            # tf.reset_default_graph()
            # self.graph = tf.Graph()
            # with self.graph.as_default():
        """
        assert(mode in ['train_noval', 'train_with_val', 'val', 'val_notrain', 'test'])
        self.mode, self.sess, self.opts = mode, session, options
        self.y_hat_train_tnsr = self.y_hat_val_tnsr = self.y_hat_test_tnsr = None
        self.name = name
        self.num_gpus = len(self.opts['gpu_devices'])
        self.dbg = False  # Set this to True for a detailed log of operation

        if _DBG_TRAIN_VAL_TEST_SETS != -1:  # Debug mode only
            if self.mode in ['train_noval', 'train_with_val']:
                self.opts['display_step'] = 10  # show progress every 10 training batches
                self.opts['snapshot_step'] = 100  # save trained model every 100 training batches
                self.opts['val_step'] = 100  # Test trained model on validation split every 1000 training batches
                if self.opts['lr_boundaries'] == 'multisteps':
                    self.opts['lr_boundaries'] = [int(boundary / 1000) for boundary in self.opts['lr_boundaries']]
                    self.opts['max_steps'] = self.opts['lr_boundaries'][-1]
                else:
                    self.opts['cyclic_lr_stepsize'] = 50
                    self.opts['max_steps'] = 500  # max number of training iterations (i.e., batches to run)

        # tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Configure a TF session, if one doesn't already exist
            self.config_session(session)

            # Build the TF graph
            self.build_graph()

    ###
    # Session mgmt
    ###
    def config_session(self, sess):
        """Configure a TF session, if one doesn't already exist.
        Args:
            sess: optional TF session
        """
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            if self.dbg:
                config.log_device_placement = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

        tf.logging.set_verbosity(tf.logging.INFO)

    ###
    # Training-specific helpers
    ###
    def config_train_ops(self):
        """Configure training ops. Override this to train your model.
        Called by the base class when building the TF graph to setup all the training ops, including:
            - setting up loss computations,
            - setting up metrics computations,
            - selecting an optimizer,
            - creating a training schedule.
        """
        raise NotImplementedError

    def config_loggers(self):
        """Configure train logger and, optionally, val logger.
        """
        if self.mode == 'train_with_val':
            self.tb_train = OptFlowTBLogger(self.opts['ckpt_dir'], 'train')
            self.tb_val = OptFlowTBLogger(self.opts['ckpt_dir'], 'val')
        elif self.mode == 'train_noval':
            self.tb_train = OptFlowTBLogger(self.opts['ckpt_dir'], 'train')

    ###
    # Checkpoint mgmt
    ###
    def init_saver(self):
        """Creates a default saver to load/save model checkpoints. Override, if necessary.
        """
        if self.mode in ['train_noval', 'train_with_val']:
            self.saver = BestCheckpointSaver(self.opts['ckpt_dir'], self.name, self.opts['max_to_keep'], maximize=False)
        else:
            self.saver = tf.train.Saver()

    def save_ckpt(self, ranking_value=0):
        """Save a model checkpoint
        Args:
            ranking_value: The ranking value by which to rank the checkpoint.
        """
        assert(self.mode in ['train_noval', 'train_with_val'])
        if self.opts['verbose']:
            print("Saving model...")

        # save_path = self.saver.save(self.sess, self.opts['ckpt_dir'] + self.name, self.g_step_op)
        save_path = self.saver.save(ranking_value, self.sess, self.g_step_op)

        if self.opts['verbose']:
            if save_path is None:
                msg = "... model wasn't saved -- its score ({ranking_value:.2f}) doesn't outperform other checkpoints"
            else:
                msg = "... model saved in {save_path}"
            print(msg)

    def load_ckpt(self):
        """Load a model checkpoint
        In train mode, load the latest checkpoint from the checkpoint folder if it exists; otherwise, run initializer.
        In other modes, load from the specified checkpoint file.
        """
        if self.mode in ['train_noval', 'train_with_val']:
            self.last_ckpt = None
            if self.opts['train_mode'] == 'fine-tune':
                # In fine-tuning mode, we just want to load the trained params from the file and that's it...
                assert(tf.train.checkpoint_exists(self.opts['ckpt_path']))
                if self.opts['verbose']:
                    print("Initializing from pre-trained model at {self.opts['ckpt_path']} for finetuning...\n")
                # ...however, the AdamOptimizer also stores variables in the graph, so reinitialize them as well
                self.sess.run(tf.variables_initializer(self.optim.variables()))
                # Now initialize the trained params with actual values from the checkpoint
                _saver = tf.train.Saver(var_list=tf.trainable_variables())
                _saver.restore(self.sess, self.opts['ckpt_path'])
                if self.opts['verbose']:
                    print("... model initialized")
                self.last_ckpt = self.opts['ckpt_path']
            else:
                # In training mode, we either want to start a new training session or resume from a previous checkpoint
                self.last_ckpt = self.saver.best_checkpoint(self.opts['ckpt_dir'], maximize=False)
                if self.last_ckpt is None:
                    self.last_ckpt = tf.train.latest_checkpoint(self.opts['ckpt_dir'])

                if self.last_ckpt:
                    # We're resuming a session -> initialize the graph with the content of the checkpoint
                    if self.opts['verbose']:
                        print("Initializing model from previous checkpoint {self.last_ckpt} to resume training...\n")
                    self.saver.restore(self.sess, self.last_ckpt)
                    if self.opts['verbose']:
                        print("... model initialized")
                else:
                    # Initialize all the variables of the graph
                    if self.opts['verbose']:
                        print("Initializing model with random values for initial training...\n")
                    assert (self.mode in ['train_noval', 'train_with_val'])
                    self.sess.run(tf.global_variables_initializer())
                    if self.opts['verbose']:
                        print("... model initialized")
        else:
            # Initialize the graph with the content of the checkpoint
            self.last_ckpt = self.opts['ckpt_path']
            assert(self.last_ckpt is not None)
            if self.opts['verbose']:
                print("Loading model checkpoint {self.last_ckpt} for eval or testing...\n")
            self.saver.restore(self.sess, self.last_ckpt)
            if self.opts['verbose']:
                print("... model loaded")

    ###
    # Model mgmt
    ###
    def build_model(self):
        """Build model. Override this.
        """
        raise NotImplementedError

    def set_output_tnsrs(self):
        """Initialize output tensors. Override this.
        """
        raise NotImplementedError

    ###
    # Graph mgmt
    ###
    def config_placeholders(self):
        """Configure input and output tensors
        Args:
            x_dtype, x_shape:  type and shape of elements in the input tensor
            y_dtype, y_shape:  shape of elements in the input tensor
        """
        # Increase the batch size with the number of GPUs dedicated to computing TF ops
        batch_size = self.num_gpus * self.opts['batch_size']
        self.x_tnsr = tf.placeholder(self.opts['x_dtype'], [batch_size] + self.opts['x_shape'], 'x_tnsr')
        self.y_tnsr = tf.placeholder(self.opts['y_dtype'], [batch_size] + self.opts['y_shape'], 'y_tnsr')

    def build_graph(self):
        """ Build the complete graph in TensorFlow
        """
        # with tf.device(self.main_device):
        # Configure input and output tensors
        self.config_placeholders()

        # Build the backbone network, then:
        # In training mode, configure training ops (loss, metrics, optimizer, and lr schedule)
        # Also, config train logger and, optionally, val logger
        # In validation mode, configure validation ops (loss, metrics)
        if self.mode in ['train_noval', 'train_with_val']:
            if self.opts['use_mixed_precision'] is True:
                with tf.variable_scope('fp32_vars', custom_getter=float32_variable_storage_getter):
                    if self.num_gpus == 1:
                        self.build_model()
                        self.config_train_ops()
                    else:
                        self.build_model_towers()
            else:
                if self.num_gpus == 1:
                    self.build_model()
                    self.config_train_ops()
                else:
                    self.build_model_towers()

            self.config_loggers()

        elif self.mode in ['val', 'val_notrain']:
            if self.opts['use_mixed_precision'] is True:
                with tf.variable_scope('fp32_vars', custom_getter=float32_variable_storage_getter):
                    self.build_model()
                    self.setup_metrics_ops()
            else:
                self.build_model()
                self.setup_metrics_ops()

        else:  # inference mode
            if self.opts['use_mixed_precision'] is True:
                with tf.variable_scope('fp32_vars', custom_getter=float32_variable_storage_getter):
                    self.build_model()
            else:
                self.build_model()

        # Set output tensors
        self.set_output_tnsrs()

        # Init saver (override if you wish) and load checkpoint if it exists
        self.init_saver()
        self.load_ckpt()

    ###
    # Sample mgmt (preprocessing and postprocessing)
    ###
    def adapt_x(self, x):
        """Preprocess the input samples to adapt them to the network's requirements
        Here, x, is the actual data, not the x TF tensor. Override as necessary.
        Args:
            x: input samples
        Returns:
            Samples ready to be given to the network (w. same shape as x) and companion adaptation info
        """
        return x, None

    def adapt_y(self, y):
        """Preprocess the labels to adapt them to the loss computation requirements of the network
        Here, y, is the actual data, not the y TF tensor. Override as necessary.
        Args:
            y: training labels
        Returns:
            Labels ready to be used by the network's loss function (w. same shape as y) and companion adaptation inf
        """
        return y, None

    def postproc_y_hat(self, y_hat):
        """Postprocess the predictions coming from the network. Override as necessary.
        Here, y_hat, is the actual data, not the y_hat TF tensor.
        Args:
            y_hat: predictions
        Returns:
            Postprocessed labels
        """
        return y_hat

    ###
    # Learning rate helpers
    ###
    def setup_lr_sched(self):
        """Setup a learning rate training schedule and setup the global step. Override as necessary.
        """
        assert (self.opts['lr_policy'] in [None, 'multisteps', 'cyclic'])
        self.g_step_op = tf.train.get_or_create_global_step()

        # Use a set learning rate, if requested
        if self.opts['lr_policy'] is None:
            self.lr = tf.constant(self.opts['init_lr'])
            return

        # Use a learning rate schedule, if requested
        assert (self.opts['train_mode'] in ['train', 'fine-tune'])
        if self.opts['lr_policy'] == 'multisteps':
            boundaries = self.opts['lr_boundaries']
            values = self.opts['lr_values']
            if self.opts['train_mode'] == 'train':
                self.lr = lr_multisteps_long(self.g_step_op, boundaries, values)
            else:
                self.lr = lr_multisteps_fine(self.g_step_op, boundaries, values)
        else:
            lr_base = self.opts['cyclic_lr_base']
            lr_max = self.opts['cyclic_lr_max']
            lr_stepsize = self.opts['cyclic_lr_stepsize']
            if self.opts['train_mode'] == 'train':
                self.lr = lr_cyclic_long(self.g_step_op, lr_base, lr_max, lr_stepsize)
            else:
                self.lr = lr_cyclic_fine(self.g_step_op, lr_base, lr_max, lr_stepsize)

    ###
    # Debug utils
    ###
    def summary(self):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def print_config(self):
        """Display configuration values.
        Ref:
            - How to count total number of trainable parameters in a tensorflow model?
            https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
        """
        with self.graph.as_default():
            print("\nModel Configuration:")
            for k, v in self.opts.items():
                if self.mode in ['train_noval', 'train_with_val']:
                    if self.opts['lr_policy'] == 'multisteps':
                        if k in ['init_lr', 'cyclic_lr_max', 'cyclic_lr_base', 'cyclic_lr_stepsize']:
                            continue
                    if self.opts['lr_policy'] == 'cyclic':
                        if k in ['init_lr', 'lr_boundaries', 'lr_values']:
                            continue
                print("  {k:22} {v}")
            print("  {'mode':22} {self.mode}")
            # if self.mode in ['train_noval', 'train_with_val']:
            if self.dbg:
                self.summary()
            print("  {'trainable params':22} {np.sum([np.prod(v.shape) for v in tf.trainable_variables()])}")
