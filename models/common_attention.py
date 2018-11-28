"""Utilities for attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from models import common_layers
import tensorflow as tf

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return x + signal


def embedding_to_padding(emb):
    """Input embeddings -> is_padding.
    """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.equal(emb_sum, 0.0)


def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.
    """
    lower_triangle = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
    ret = -1e9 * (1.0 - lower_triangle)
    return tf.reshape(ret, [1, 1, length, length])


def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.
        input: [batch, memory_length], return: [batch, 1, 1, memory_length].
    """
    ret = tf.to_float(memory_padding) * -1e9
    return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    note: [..., m] --> [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return ret


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    note: [..., a, b] --> [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret


def split_heads(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).
    note: [batch, length, channels] -> [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])

def sb_split_heads(x, num_heads):
    return tf.transpose(split_last_dimension(x, num_heads), [0, 1, 3, 2, 4])

def combine_heads(x):
    """Inverse of split_heads.
    note: [batch, num_heads, length, channels / num_heads] -> [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def sb_combine_heads(x):
    return combine_last_two_dimensions(tf.transpose(x, [0, 1, 3, 2, 4]))

def shape_list(x):
    #x = tf.conver_to_tensor(x)
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in xrange(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def residual_fn(x, y, hparams):
    return common_layers.layer_norm(x + tf.nn.dropout(
            y, 1.0 - hparams.residual_dropout))

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        return tf.matmul(weights, v)


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        cache=None,
                        summaries=False,
                        image_shapes=None,
                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
    """
    with tf.variable_scope(
        name,
        default_name="multihead_attention",
        values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
            # self attention
            combined = common_layers.conv1d(
                    query_antecedent,
                    total_key_depth * 2 + total_value_depth,
                    1,
                    name="qkv_transform")
            q, k, v = tf.split(
                    combined, [total_key_depth, total_key_depth, total_value_depth],
                    axis=2)
        else:
            q = common_layers.conv1d(
                    query_antecedent, total_key_depth, 1, name="q_transform")
            combined = common_layers.conv1d(
                    memory_antecedent,
                    total_key_depth + total_value_depth,
                    1,
                    name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
    
        if cache is not None:
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                             "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)

        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        x = dot_product_attention(
                q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = combine_heads(x)
        x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
        return x

def sb_dot_product_attention_for_decoding(q,
                          k,
                          v,
                          bias,
                          batch_size=None,
                          beam_size=None,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="sb_dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights_l2r")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        #if summaries and not tf.get_variable_scope().reuse:
        #    attention_image_summary(weights, image_shapes)
        final_l2r = tf.matmul(weights, v) ## [batch*beam, num_heads, length_tmp, hidden_size/num_heads]

        ## calculate final_r2l
        shape = shape_list(k)
        new_shape = [batch_size]+[2]+[tf.cast(beam_size/2,tf.int32)]+shape[1:]
        k_ = tf.reshape(k, new_shape) ## [batch, 2, beam/2, num_heads, length_tmp, hidden_size/num_heads]
        k_ = tf.reverse(k_,[1])
        v_ = tf.reshape(v, new_shape)
        v_ = tf.reverse(v_,[1])

        shape_ = shape_list(k_)
        new_shape_ = [batch_size*beam_size]+shape_[3:]
        k_ = tf.reshape(k_, new_shape_) ## [batch*beam, num_heads, length_tmp, hidden_size/num_heads]
        v_ = tf.reshape(v_, new_shape_)
        logits_ = tf.matmul(q, k_, transpose_b=True)
        logits_ += bias
        weights_ = tf.nn.softmax(logits_, name="attention_weights_r2l")
        weights_ = tf.nn.dropout(weights_, 1.0 - dropout_rate)
        final_r2l = tf.matmul(weights_, v_)

        final_all = final_l2r + 0.1*tf.tanh(final_r2l) ## [batch*beam, num_heads, length_tmp, hidden_size/num_heads]
        return final_all


def sb_dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="sb_dot_product_attention", values=[q, k, v]):
        # [2, batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        bias = tf.expand_dims(bias, axis=0)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights_l2r")
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        #if summaries and not tf.get_variable_scope().reuse:
        #    attention_image_summary(weights[0], image_shapes)
        final_l2r = tf.matmul(weights, v) ## [2, batch, num_heads, length, hidden_size/num_heads]

        ## calculate final_r2l
        k_ = tf.reverse(k, [0])
        v_ = tf.reverse(v, [0])
        logits_ = tf.matmul(q, k_, transpose_b=True)
        logits_ += bias ### modify err, logits --> logits_
        weights_ = tf.nn.softmax(logits_, name="attention_weights_r2l")
        weights_ = tf.nn.dropout(weights_, 1.0 - dropout_rate)
        final_r2l = tf.matmul(weights_, v_)

        final_all = final_l2r + 0.1*tf.tanh(tf.nn.dropout(final_r2l, 1-0.3))
        return final_all ## [2, batch, num_heads, length, hidden_size/num_heads]

def sb_multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        cache=None,
                        summaries=False,
                        image_shapes=None,
                        name=None,
                        is_decoding=False):
    """Multihead scaled-dot-product attention with input/output transformations.
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
    """
    
    with tf.variable_scope(
                name,
                default_name="sb_multihead_attention",
                values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
        # self attention
            combined = common_layers.sb_conv1d(
                query_antecedent,
                total_key_depth * 2 + total_value_depth,
                1,
                name="qkv_transform")
            q, k, v = tf.split(
                combined, [total_key_depth, total_key_depth, total_value_depth], axis=3) ## 2-->3
        else:
            q = common_layers.sb_conv1d(
                query_antecedent, total_key_depth, 1, name="q_transform")
            combined = common_layers.conv1d(
                memory_antecedent,
                total_key_depth + total_value_depth,
                1,
                name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
			
            k = tf.concat([tf.expand_dims(k,0), tf.expand_dims(k,0)], axis=0) ## [2, batch, length, hidden_size]
            v = tf.concat([tf.expand_dims(v,0), tf.expand_dims(v,0)], axis=0) 
    
        if cache is not None:
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                             "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)

        q = sb_split_heads(q, num_heads)
        k = sb_split_heads(k, num_heads)
        v = sb_split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        if memory_antecedent is None: ## decoder self attention (synchronous bidirectional att)
            x = sb_dot_product_attention( ## for training
	        q, k, v, bias, dropout_rate, summaries, image_shapes) ## q: [2, num_heads, length_tmp, lenght]
        else: ## enc-dec attention
            x = dot_product_attention(
                q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = sb_combine_heads(x)
        x = common_layers.sb_conv1d(x, output_depth, 1, name="output_transform")
        return x

def sb_multihead_attention_for_decoding(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        batch_size=None,
                        beam_size=None,
                        cache=None,
                        summaries=False,
                        image_shapes=None,
                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
    """
    
    with tf.variable_scope(
                name,
                default_name="sb_multihead_attention",
                values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
            # self attention
            combined = common_layers.conv1d(
                query_antecedent,
                total_key_depth * 2 + total_value_depth,
                1,
                name="qkv_transform")
            q, k, v = tf.split(
                combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
        else:
            q = common_layers.conv1d(
                query_antecedent, total_key_depth, 1, name="q_transform")
            combined = common_layers.conv1d(
                memory_antecedent,
                total_key_depth + total_value_depth,
                1,
                name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
			 
        if cache is not None:
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                             "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)

        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        if memory_antecedent is None: ## decoder self attention (synchronous bidirectional att)
            x = sb_dot_product_attention_for_decoding( ## for decoding
	        q, k, v, bias, batch_size, beam_size, dropout_rate, summaries, image_shapes) ## q: [batch, num_heads, length_tmp, lenght]
        else: ## enc-dec attention
            x = dot_product_attention(
                q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = combine_heads(x)
        x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
        return x

