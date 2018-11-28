"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.util import nest
import time
import sys

from models import common_hparams
from models import common_attention
from models import common_layers
from utils import inference
from utils import parallel


class Transformer(object):
    """Attention net.  See file docstring."""

    def __init__(self,
                hparams,
                mode,
                data_parallelism=None):
    
        hparams = copy.copy(hparams)
        hparams.add_hparam("mode", mode)
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            for key in hparams.values():
                if key[-len("dropout"):] == "dropout":
                    setattr(hparams, key, 0.0)
        self._hparams = hparams
        self._data_parallelism = data_parallelism
        self._num_datashards = data_parallelism.n   
        ##source side
        self._hparams.input_modality = SymbolModality(hparams, hparams.vocab_src_size)
        ## target side 
        self._hparams.target_modality = SymbolModality(hparams, hparams.vocab_tgt_size)
        
    def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0):
        """A inference method.
        """
        local_features = {}
        local_features["_num_datashards"] = self._num_datashards
        local_features["_data_parallelism"] = self._data_parallelism
        local_features["_hparams"] = self._hparams
        local_features["_shard_features"] = self._shard_features
        local_features["encode"] = self.encode
        local_features["decode"] = self.decode

        if beam_size == 1:
            tf.logging.info("Greedy Decoding")
            return inference._greedy_infer(features, decode_length, local_features)
        else:
            tf.logging.info("Beam Decoding with beam size %d" % beam_size)
            return inference._beam_decode(features, decode_length, beam_size, top_beams, alpha, local_features)


    def _shard_features(self, features):  # pylint: disable=missing-docstring
        sharded_features = dict()
        for k, v in six.iteritems(features):
            v = tf.convert_to_tensor(v)
            if not v.shape.as_list():
                v = tf.expand_dims(v, axis=-1)
                v = tf.tile(v, [self._num_datashards])
            sharded_features[k] = self._data_parallelism(tf.identity, tf.split(v, self._num_datashards, 0))

        return sharded_features

    def model_fn(self, features, skip=False, last_position_only=False):
        """Computes the entire model and produces sharded logits and training loss.
        """
    
        start_time = time.time()
        dp = self._data_parallelism
        sharded_features = self._shard_features(features)    
        transformed_features = {}
    
        # source embedding
        with tf.variable_scope(self._hparams.input_modality.name, reuse=False):
            transformed_features["inputs"] = self._hparams.input_modality.bottom_sharded(
                sharded_features["inputs"], dp)
    
        # target embedding
        with tf.variable_scope(self._hparams.target_modality.name, reuse=False):
            transformed_features["targets_l2r"] = self._hparams.target_modality.targets_bottom_sharded(
                    sharded_features["targets_l2r"], dp)
            transformed_features["targets_r2l"] = self._hparams.target_modality.targets_bottom_sharded(
                    sharded_features["targets_r2l"], dp)

        # Construct the model body.
        with tf.variable_scope("body", reuse=False):
            with tf.name_scope("model"):
                datashard_to_features = [{
                    k: v[d] for k, v in six.iteritems(transformed_features)
                    } for d in xrange(self._num_datashards)]
                body_outputs = self._data_parallelism(self.model_fn_body, datashard_to_features)
                extra_loss = 0.
    
        body_outputs_l2r = []
        body_outputs_r2l = []
        ## for multi-gpus
        for output in body_outputs:
            body_outputs_l2r.append(output[0])
            body_outputs_r2l.append(output[1])
    
        # target linear transformation and compute loss
        with tf.variable_scope(self._hparams.target_modality.name, reuse=False):  ## = target_reuse
            sharded_logits, training_loss_l2r = (self._hparams.target_modality.top_sharded(
                body_outputs_l2r, sharded_features["targets_l2r"], self._data_parallelism))    
            sharded_logits_r2l, training_loss_r2l = (self._hparams.target_modality.top_sharded(
                body_outputs_r2l, sharded_features["targets_r2l"], self._data_parallelism))    
            training_loss = training_loss_l2r + training_loss_r2l
      
            training_loss *= self._hparams.loss_multiplier

        tf.logging.info("This model_fn took %.3f sec." % (time.time() - start_time))
        return sharded_logits, training_loss, extra_loss


    def model_fn_body(self, features):
        hparams = copy.copy(self._hparams)
        inputs = features.get("inputs")

        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, hparams)

        targets_l2r = features["targets_l2r"]
        targets_r2l = features["targets_r2l"]
        targets_l2r = common_layers.flatten4d3d(targets_l2r)
        targets_r2l = common_layers.flatten4d3d(targets_r2l)
        (decoder_input, decoder_self_attention_bias) = transformer_prepare_decoder(
                targets_l2r, targets_r2l, hparams)

        decode_output = self.decode(decoder_input, encoder_output, encoder_decoder_attention_bias,
                decoder_self_attention_bias, hparams)

        return decode_output

    def encode(self, inputs, hparams):
        inputs = common_layers.flatten4d3d(inputs)

        (encoder_input, self_attention_bias, encoder_decoder_attention_bias) = \
            transformer_prepare_encoder(inputs, hparams)

        encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.residual_dropout)
        encoder_output = transformer_encoder(encoder_input, self_attention_bias, hparams)

        return encoder_output, encoder_decoder_attention_bias


    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias,
                decoder_self_attention_bias, hparams, batch_size=None, beam_size=None, cache=None):
        decoder_input = tf.nn.dropout(decoder_input, 1.0 - hparams.residual_dropout)
      
        if cache is None: ##training
            decoder_output = transformer_decoder(
                    decoder_input, encoder_output, decoder_self_attention_bias,
                    encoder_decoder_attention_bias, hparams, cache=cache)
            return tf.expand_dims(decoder_output, axis=3)
        else: ##inference
            decoder_output = transformer_decoder_for_decoding(
                    decoder_input, encoder_output, decoder_self_attention_bias,
                    encoder_decoder_attention_bias, hparams, batch_size, beam_size, cache=cache)
            return tf.expand_dims(decoder_output, axis=2)

def transformer_prepare_encoder(inputs, hparams):
    """Prepare one shard of the model for the encoder.
    """
    # Flatten inputs.
    ishape_static = inputs.shape.as_list()
    encoder_input = inputs
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    
    ##remove
    emb_target_space = common_layers.embedding(
        9, 32, ishape_static[-1], name="target_space_embedding")
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input += emb_target_space

    if hparams.pos == "timing":
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
    return (encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias)


def transformer_prepare_decoder(targets_l2r, targets_r2l, hparams):
    """Prepare one shard of the model for the decoder.
    """
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(tf.shape(targets_l2r)[1])) ## [1, 1, length, length]
    decoder_input_l2r = common_layers.shift_left_3d(targets_l2r)
    decoder_input_r2l = common_layers.shift_left_3d(targets_r2l)
    if hparams.pos == "timing":
        decoder_input_l2r = common_attention.add_timing_signal_1d(decoder_input_l2r)
        decoder_input_r2l = common_attention.add_timing_signal_1d(decoder_input_r2l)
    decoder_input = tf.concat([tf.expand_dims(decoder_input_l2r, 0), tf.expand_dims(decoder_input_r2l, 0)], axis=0) ## [2, batch, length, hidden_size]
    return (decoder_input, decoder_self_attention_bias)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder"):
    """A stack of transformer layers.
    """
    x = encoder_input
    # Summaries don't work in multi-problem setting yet.
    summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
    with tf.variable_scope(name):
        for layer in xrange(hparams.num_hidden_layers_src):
            with tf.variable_scope("layer_%d" % layer):
                y = common_attention.multihead_attention(
                    x,
                    None,
                    encoder_self_attention_bias,
                    hparams.attention_key_channels or hparams.hidden_size,
                    hparams.attention_value_channels or hparams.hidden_size,
                    hparams.hidden_size,
                    hparams.num_heads,
                    hparams.attention_dropout,
                    summaries=summaries,
                    name="encoder_self_attention")
                x = common_attention.residual_fn(x, y, hparams) ###
                y = transformer_ffn_layer(x, hparams)
                x = common_attention.residual_fn(x, y, hparams)
    return x


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder"):
    """A stack of transformer layers.
    """
    x = decoder_input
    # Summaries don't work in multi-problem setting yet.
    summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
    with tf.variable_scope(name):
        for layer in xrange(hparams.num_hidden_layers_tgt):
            layer_name = "layer_%d" % layer
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                y = common_attention.sb_multihead_attention(
                    x,
                    None,
                    decoder_self_attention_bias,
                    hparams.attention_key_channels or hparams.hidden_size,
                    hparams.attention_value_channels or hparams.hidden_size,
                    hparams.hidden_size,
                    hparams.num_heads,
                    hparams.attention_dropout,
                    cache=layer_cache,
                    summaries=summaries,
                    name="decoder_self_attention")
                x = common_attention.residual_fn(x, y, hparams)
                y = common_attention.sb_multihead_attention(
                    x,
                    encoder_output,
                    encoder_decoder_attention_bias,
                    hparams.attention_key_channels or hparams.hidden_size,
                    hparams.attention_value_channels or hparams.hidden_size,
                    hparams.hidden_size,
                    hparams.num_heads,
                    hparams.attention_dropout,
                    summaries=summaries,
                    name="encdec_attention")
                x = common_attention.residual_fn(x, y, hparams)
                y = transformer_ffn_layer(x, hparams)
                x = common_attention.residual_fn(x, y, hparams)
    return x

def transformer_decoder_for_decoding(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        batch_size=None,
                        beam_size=None,
                        cache=None,
                        name="decoder"):
    """A stack of transformer layers.
    """
    x = decoder_input
    # Summaries don't work in multi-problem setting yet.
    summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
    with tf.variable_scope(name):
        for layer in xrange(hparams.num_hidden_layers_tgt):
            layer_name = "layer_%d" % layer
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                y = common_attention.sb_multihead_attention_for_decoding(
                        x,
                        None,
                        decoder_self_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        batch_size,
                        beam_size,
                        cache=layer_cache,
                        summaries=summaries,
                        name="decoder_self_attention")
                x = common_attention.residual_fn(x, y, hparams)
                y = common_attention.sb_multihead_attention_for_decoding(
                        x,
                        encoder_output,
                        encoder_decoder_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        summaries=summaries,
                        name="encdec_attention")
                x = common_attention.residual_fn(x, y, hparams)
                y = transformer_ffn_layer(x, hparams)
                x = common_attention.residual_fn(x, y, hparams)
    return x


def transformer_ffn_layer(x, hparams):
    """Feed-forward layer in the transformer.
    [batch_size, length, hparams.hidden_size] -->  [batch_size, length, hparams.hidden_size]
    """
    if hparams.ffn_layer == "conv_hidden_relu":
        return common_layers.conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            dropout=hparams.relu_dropout)
    else:
        assert hparams.ffn_layer == "none"
        return x


####################################################################

class SymbolModality(object):
    """Modality for sets of discrete symbols.
    Input: Embedding.
    Output: Linear transformation + softmax.
    """

    def __init__(self, model_hparams, vocab_size=None):
        self._model_hparams = model_hparams
        self._vocab_size = vocab_size

    @property
    def name(self):
        return "symbol_modality_%d_%d" % (self._vocab_size, self._body_input_depth)

    @property
    def top_dimensionality(self):
        return self._vocab_size
  
    @property
    def _body_input_depth(self):
        return self._model_hparams.hidden_size

    def _get_weights(self):
        """Create or get concatenated embedding or softmax variable.
        Returns: a list of self._num_shards Tensors.
        """
        num_shards = self._model_hparams.symbol_modality_num_shards
        shards = []
        for i in xrange(num_shards):
            shard_size = (self._vocab_size // num_shards) + (
                1 if i < self._vocab_size % num_shards else 0)
            var_name = "weights_%d" % i
            shards.append(
                tf.get_variable(
                    var_name, [shard_size, self._body_input_depth],
                    initializer=tf.random_normal_initializer(
                        0.0, self._body_input_depth**-0.5)))
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = parallel.ConvertGradientToTensor(ret)
        return ret

    def bottom_simple(self, x, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            # Squeeze out the channels dimension.
            x = tf.squeeze(x, axis=3)
            var = self._get_weights()
            ret = tf.gather(var, x)
            if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
                ret *= self._body_input_depth**0.5
            ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
            return ret

    def bottom(self, x):
        if self._model_hparams.shared_source_embedding_and_softmax_weights:
            return self.bottom_simple(x, "shared", reuse=None)
        else:
            return self.bottom_simple(x, "input_emb", reuse=None)

    def targets_bottom(self, x):
        if self._model_hparams.shared_target_embedding_and_softmax_weights:
        #return self.bottom_simple(x, "shared", reuse=True)
            return self.bottom_simple(x, "shared", reuse=tf.AUTO_REUSE)
        else:
            return self.bottom_simple(x, "target_emb", reuse=None)

    def top(self, body_output, targets):
        """Generate logits.
        Args:
            body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
            targets: A Tensor with shape [batch, p0, p1, 1]
        Returns:
            logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
        """
        if self._model_hparams.shared_target_embedding_and_softmax_weights:
            scope_name = "shared"
            reuse = True
        else:
            scope_name = "softmax"
            reuse = False
        with tf.variable_scope(scope_name, reuse=reuse):
            var = self._get_weights()
            shape = tf.shape(body_output)[:-1]
            body_output = tf.reshape(body_output, [-1, self._body_input_depth])
            logits = tf.matmul(body_output, var, transpose_b=True)
            logits = tf.reshape(logits, tf.concat([shape, [self._vocab_size]], 0))
            # insert a channels dimension
            return tf.expand_dims(logits, 3)

    def bottom_sharded(self, xs, data_parallelism):
        """Transform the inputs.
            [batch, p0, p1, depth --> [batch, p0, p1, body_input_depth].
        """
        return data_parallelism(self.bottom, xs)

    def targets_bottom_sharded(self, xs, data_parallelism):
        """Transform the targets.
            [batch, p0, p1, target_channels] --> [batch, p0, p1, body_input_depth].
        """
        return data_parallelism(self.targets_bottom, xs)

    def top_sharded(self,
                  sharded_body_output,
                  sharded_targets,
                  data_parallelism,
                  weights_fn=common_layers.weights_nonzero):
        """Transform all shards of targets.
        Classes with cross-shard interaction will override this function.
        """
        sharded_logits = data_parallelism(self.top, sharded_body_output,
                                      sharded_targets)
        if sharded_targets is None:
            return sharded_logits, 0

        loss_num, loss_den = data_parallelism(
            common_layers.padded_cross_entropy,
            sharded_logits,
            sharded_targets,
            self._model_hparams.label_smoothing,
            weights_fn=weights_fn)
        loss = tf.add_n(loss_num) / tf.maximum(1.0, tf.add_n(loss_den))
        return sharded_logits, loss


