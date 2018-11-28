"""Hyperparameters and ranges common to multiple models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from utils import text_reader


def transformer_params():
    """A set of basic hyperparameters."""
    return tf.contrib.training.HParams(
        batching_mantissa_bits=3,
        kernel_height=3,
        kernel_width=1,
        compress_steps=0,
        dropout=0.0,
        clip_grad_norm=0.0,
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        label_smoothing=0.1,
        optimizer="Adam",
        optimizer_adam_epsilon=1e-9,
        optimizer_adam_beta1=0.9,
        optimizer_adam_beta2=0.998,
        optimizer_momentum_momentum=0.9,
        weight_decay=0.0,
        weight_noise=0.0,
        learning_rate_decay_scheme="noam",
        learning_rate_warmup_steps=16000,
        learning_rate=0.1,
        sampling_method="argmax",  # "argmax" or "random"
        multiply_embedding_mode="sqrt_depth",
        symbol_modality_num_shards=16,
        num_sampled_classes = 0,
        shared_source_embedding_and_softmax_weights = int(True),
        shared_target_embedding_and_softmax_weights = int(True),
        pos = "timing",
        ffn_layer = "conv_hidden_relu",
        attention_key_channels = 0,
        attention_value_channels = 0,
      
        hidden_size = 256,
        batch_size = 4096,
        max_length = 256,
        filter_size = 1024,
        num_heads = 4,
        attention_dropout = 0.0,
        relu_dropout = 0.0,
        residual_dropout = 0.1,
        nbr_decoder_problems = 1,
        num_hidden_layers = 6,
        num_hidden_layers_src = 6,
        num_hidden_layers_tgt = 6,
      
        # problem hparams
        loss_multiplier=1.4,
        batch_size_multiplier=1,
        max_expected_batch_size_per_shard=64,
        input_modality=None,
        target_modality=None,
        vocab_src_size = 30000,
        vocab_tgt_size = 30000,
        vocabulary = {
        },
    )

def transformer_params_big(data_dir, vocab_src_name, vocab_tgt_name):
    """A set of basic hyperparameters."""
    hparams = transformer_params()
    hparams.vocabulary = {
        "inputs": text_reader.TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_src_name)),
        "targets": text_reader.TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_tgt_name))}
    hparams.hidden_size = 1024
    hparams.filter_size = 4096
    hparams.num_heads = 16
    hparams.batching_mantissa_bits = 3
    return hparams

def transformer_params_base(data_dir, vocab_src_name, vocab_tgt_name):
    """A set of basic hyperparameters."""
    hparams = transformer_params()
    hparams.vocabulary = {
        "inputs": text_reader.TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_src_name)),
        "targets": text_reader.TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_tgt_name))}
    hparams.hidden_size = 512
    hparams.filter_size = 2048
    hparams.num_heads = 8
    hparams.batching_mantissa_bits = 2
    return hparams


