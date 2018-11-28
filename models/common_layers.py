"""Layers common to multiple models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from utils import parallel

import tensorflow as tf
from tensorflow.python.framework import function

# This is a global setting. When turned off, no @function.Defun is used.
allow_defun = True

def flatten4d3d(x):
    """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
    xshape = tf.shape(x)
    result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
    # Preserve static shapes when available.
    xshape_static = x.get_shape()
    result.set_shape([xshape_static[0], None, xshape_static[3]])
    return result


def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors, reducing to max 4 dimensions."""
    with tf.variable_scope(
            name, default_name="embedding", values=[x], reuse=reuse):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        # On the backwards pass, we want to convert the gradient from
        # an indexed-slices to a regular tensor before sending it back to the
        # parameter server. This avoids excess computation on the parameter server.
        embedding_var = parallel.ConvertGradientToTensor(embedding_var)
        emb_x = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            emb_x *= multiplier
        shape, static_shape = tf.shape(emb_x), emb_x.shape.as_list()
        if not static_shape or len(static_shape) < 5:
            return emb_x
        # If we had extra channel dimensions, assume it's 1, i.e. shape[3] == 1.
        assert len(static_shape) == 5
        return tf.reshape(emb_x, [shape[0], shape[1], shape[2], static_shape[4]])


def shift_left(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    if pad_value is None:
        shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
    else:
        shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
    return shifted_targets


def shift_left_3d(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    if pad_value is None:
        shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    else:
        shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
    return shifted_targets


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
    """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
        raise ValueError("Inputs to conv must have statically known rank 4.")
    # Add support for left padding.
    if "padding" in kwargs and kwargs["padding"] == "LEFT":
        dilation_rate = (1, 1)
        if "dilation_rate" in kwargs:
            dilation_rate = kwargs["dilation_rate"]
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
        cond_padding = tf.cond(
                tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
                lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
        inputs = tf.pad(inputs, padding)
        # Set middle two dimensions to None to prevent convolution from complaining
        inputs.set_shape([static_shape[0], None, None, static_shape[3]])
        kwargs["padding"] = "VALID"

    def conv2d_kernel(kernel_size_arg, name_suffix):
        """Call conv2d but add suffix to name."""
        if "name" in kwargs:
            original_name = kwargs["name"]
            name = kwargs.pop("name") + "_" + name_suffix
        else:
            original_name = None
            name = "conv_" + name_suffix
        original_force2d = None
        if "force2d" in kwargs:
            original_force2d = kwargs.pop("force2d")
        result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
        if original_name is not None:
            kwargs["name"] = original_name  # Restore for other calls.
        if original_force2d is not None:
            kwargs["force2d"] = original_force2d
        return result

    return conv2d_kernel(kernel_size, "single")


def conv(inputs, filters, kernel_size, **kwargs):
    return conv_internal(tf.layers.conv2d, inputs, filters, kernel_size, **kwargs)


def conv1d(inputs, filters, kernel_size, **kwargs):
    return tf.squeeze(
            conv(tf.expand_dims(inputs, 2), filters, (kernel_size, 1), **kwargs), 2)
def sb_conv1d(inputs, filters, kernel_size, **kwargs):
  return conv(inputs, filters, (kernel_size, 1), **kwargs)


def separable_conv(inputs, filters, kernel_size, **kwargs):
    return conv_internal(tf.layers.separable_conv2d, inputs, filters, kernel_size, **kwargs)

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


@function.Defun(compiled=True)
def layer_norm_compute_grad(x, epsilon, scale, bias, dy):
    y = layer_norm_compute_python(x, epsilon, scale, bias)
    dx = tf.gradients(ys=[y], xs=[x, epsilon, scale, bias], grad_ys=[dy])
    return dx


@function.Defun(
    compiled=True,
    separate_compiled_gradients=True,
    grad_func=layer_norm_compute_grad)
def layer_norm_compute(x, epsilon, scale, bias):
    return layer_norm_compute_python(x, epsilon, scale, bias)


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(
            name, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
                "layer_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable(
                "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
        if allow_defun:
            result = layer_norm_compute(x, tf.constant(epsilon), scale, bias)
            result.set_shape(x.get_shape())
        else:
            result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def residual_function(hparams):
    """Returns a function for combining layer input and layer output.
    """

    def residual_fn(x, y):
        return hparams.norm_function(x + tf.nn.dropout(
            y, 1.0 - hparams.residual_dropout))

    return residual_fn

def relu_density_logit(x, reduce_dims):
    """logit(density(x)).
    """
    frac = tf.reduce_mean(tf.to_float(x > 0.0), reduce_dims)
    scaled = tf.log(frac + math.exp(-10)) - tf.log((1.0 - frac) + math.exp(-10))
    return scaled


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     second_kernel_size=(1, 1),
                     summaries=True,
                     dropout=0.0,
                     **kwargs):
    """Hidden layer with RELU activation followed by linear projection."""
    name = kwargs.pop("name") if "name" in kwargs else None
    with tf.variable_scope(name, "conv_hidden_relu", [inputs]):
        if inputs.get_shape().ndims == 3:
            is_3d = True
            inputs = tf.expand_dims(inputs, 2)
        else:
            is_3d = False
        conv_f1 = conv if kernel_size == (1, 1) else separable_conv
        h = conv_f1(
                inputs,
                hidden_size,
                kernel_size,
                activation=tf.nn.relu,
                name="conv1",
                **kwargs)
        if dropout != 0.0:
            h = tf.nn.dropout(h, 1.0 - dropout)
        conv_f2 = conv if second_kernel_size == (1, 1) else separable_conv
        ret = conv_f2(h, output_size, second_kernel_size, name="conv2", **kwargs)
        if is_3d:
            ret = tf.squeeze(ret, 2)
        return ret

def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
    """Pad tensors x and y on axis 1 so that they have the same length."""
    if axis not in [1, 2]:
        raise ValueError("Only axis=1 and axis=2 supported for now.")
    with tf.name_scope("pad_to_same_length", [x, y]):
        x_length = tf.shape(x)[axis]
        y_length = tf.shape(y)[axis]
        max_length = tf.maximum(x_length, y_length)
        if final_length_divisible_by > 1:
            # Find the nearest larger-or-equal integer divisible by given number.
            max_length += final_length_divisible_by - 1
            max_length //= final_length_divisible_by
            max_length *= final_length_divisible_by
        length_diff1 = max_length - x_length
        length_diff2 = max_length - y_length

        def padding_list(length_diff, arg):
            if axis == 1:
                return [[[0, 0], [0, length_diff]],
                        tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
            return [[[0, 0], [0, 0], [0, length_diff]],
                    tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

        paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
        paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
        res_x = tf.pad(x, paddings1)
        res_y = tf.pad(y, paddings2)
        # Static shapes are the same except for axis=1.
        x_shape = x.shape.as_list()
        x_shape[axis] = None
        res_x.set_shape(x_shape)
        y_shape = y.shape.as_list()
        y_shape[axis] = None
        res_y.set_shape(y_shape)
        return res_x, res_y


def pad_with_zeros(logits, labels):
    """Pad labels on the length dimension to match logits length."""
    with tf.name_scope("pad_with_zeros", [logits, labels]):
        logits, labels = pad_to_same_length(logits, labels)
        if len(labels.shape.as_list()) == 3:  # 2-d labels.
            logits, labels = pad_to_same_length(logits, labels, axis=2)
        return logits, labels


def weights_nonzero(labels):
    """Assign weight 1.0 to all labels except for padding (id=0)."""
    return tf.to_float(tf.not_equal(labels, 0))


def padded_cross_entropy(logits,
                         labels,
                         label_smoothing,
                         weights_fn=weights_nonzero,
                         reduce_sum=True):
    """Compute cross-entropy assuming 0s are padding.

    Computes a loss numerator (the sum of losses), and loss denominator
    (the number of non-padding tokens).

    Args:
        logits: a `Tensor` with shape `[batch, timesteps, vocab_size]`.
        labels: an integer `Tensor` with shape `[batch, timesteps]`.
        label_smoothing: a floating point `Scalar`.
        weights_fn: A function from labels to weights.
        reduce_sum: a Boolean, whether to sum at the end or not.

    Returns:
        loss_numerator: a `Scalar`.  Sum of losses.
        loss_denominator: a `Scalar.  The number of non-padding target tokens.
    """
    confidence = 1.0 - label_smoothing
    vocab_size = tf.shape(logits)[-1]
    with tf.name_scope("padded_cross_entropy", [logits, labels]):
        pad_logits, pad_labels = pad_with_zeros(logits, labels)
        xent = smoothing_cross_entropy(pad_logits, pad_labels, vocab_size, confidence)
        weights = weights_fn(pad_labels)
        if not reduce_sum:
            return xent * weights, weights
        return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def smoothing_cross_entropy(logits, labels, vocab_size, confidence):
    """Cross entropy with label smoothing to limit over-confidence."""
    with tf.name_scope("smoothing_cross_entropy", [logits, labels]):
        # Low confidence is given to all non-true labels, uniformly.
        low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
        # Normalizing constant is the best cross-entropy value with soft targets.
        # We subtract it just for readability, makes no difference on learning.
        normalizing = -(confidence * tf.log(confidence) + tf.to_float(
                vocab_size - 1) * low_confidence * tf.log(low_confidence + 1e-20))
        # Soft targets.
        soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=soft_targets)
        return xentropy - normalizing

def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


