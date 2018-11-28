"""Utilities for creating Sparsely-Gated Mixture-of-Experts Layers.

See the most recent draft of our ICLR paper:
https://openreview.net/pdf?id=B1ckMDqlg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.framework import function



@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def ConvertGradientToTensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `ConvertGradientToTensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


class Parallelism(object):
  """Helper class for creating sets of parallel function calls.

  The purpose of this class is to replace this code:

      e = []
      f = []
      for i in xrange(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)

  with this code:

      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
  """

  def __init__(self,
               device_names_or_functions,
               reuse=None,
               caching_devices=None,
               daisy_chain_variables=False):
    """Create a Parallelism.

    Args:
      device_names_or_functions: A list of of length n, containing device names
        or device functions (see `tf.device`)
      reuse: True or None.  Whether to reuse variables created in the first
        replica in the subsequent replicas.
      caching_devices: Either `None`, or a list of length n containing device
        names.
      daisy_chain_variables: a boolean - if true, then copies variables in a
        daisy chain between devices.

    Returns:
      a Parallelism.
    """
    assert device_names_or_functions
    self._devices = device_names_or_functions
    self._n = len(device_names_or_functions)
    self._reuse = reuse
    self._caching_devices = self._MaybeRepeat(caching_devices)
    self._daisy_chain_variables = daisy_chain_variables

  def __call__(self, fn, *args, **kwargs):
    """A parallel set of function calls (using the specified devices).

    Args:
      fn: a function or a list of n functions.
      *args: additional args.  Each arg should either be not a list, or a list
         of length n.
      **kwargs: additional keyword args.  Each arg should either be not a
         list, or a list of length n.

    Returns:
      either a single list of length n (if fn does not return a tuple), or a
      tuple of lists of length n (if fn returns a tuple).
    """
    # Construct lists or args and kwargs for each function.
    if args:
      my_args = TransposeListOfLists([self._MaybeRepeat(arg) for arg in args])
    else:
      my_args = [[] for _ in xrange(self.n)]
    my_kwargs = [{} for _ in xrange(self.n)]
    for k, v in six.iteritems(kwargs):
      vals = self._MaybeRepeat(v)
      for i in xrange(self.n):
        my_kwargs[i][k] = vals[i]

    # Construct lists of functions.
    fns = self._MaybeRepeat(fn)

    # Now make the parallel call.
    outputs = []
    cache = {}
    for i in xrange(self.n):

      def DaisyChainGetter(getter, name, *args, **kwargs):
        """Get a variable and cache in a daisy chain."""
        device_var_key = (self._devices[i], name)
        if device_var_key in cache:
          # if we have the variable on the correct device, return it.
          return cache[device_var_key]
        if name in cache:
          # if we have it on a different device, copy it from the last device
          v = tf.identity(cache[name])
        else:
          var = getter(name, *args, **kwargs)
          v = tf.identity(var._ref())  # pylint: disable=protected-access
        # update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

      # Variable scope will not reset caching_device on reused variables,
      # so we make a custom getter that uses identity to cache the variable.
      # pylint: disable=cell-var-from-loop
      def CachingGetter(getter, name, *args, **kwargs):
        v = getter(name, *args, **kwargs)
        key = (self._caching_devices[i], name)
        if key in cache:
          return cache[key]
        with tf.device(self._caching_devices[i]):
          ret = tf.identity(v._ref())  # pylint: disable=protected-access
        cache[key] = ret
        return ret

      if self._daisy_chain_variables:
        custom_getter = DaisyChainGetter
      elif self._caching_devices:
        custom_getter = CachingGetter
      else:
        custom_getter = None
      # pylint: enable=cell-var-from-loop
      with tf.name_scope('parallel_%d' % i):
        with tf.variable_scope(
            tf.get_variable_scope(),
            reuse=True if i > 0 and self._reuse else None,
            caching_device=self._caching_devices[i],
            custom_getter=custom_getter):
          with tf.device(self._devices[i]):
            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
    if isinstance(outputs[0], tuple):
      outputs = list(zip(*outputs))
      outputs = tuple([list(o) for o in outputs])
    return outputs

  @property
  def n(self):
    return self._n

  @property
  def devices(self):
    return self._devices

  def _MaybeRepeat(self, x):
    """Utility function for processing arguments that are singletons or lists.

    Args:
      x: either a list of self.n elements, or not a list.

    Returns:
      a list of self.n elements.
    """
    if isinstance(x, list):
      assert len(x) == self.n
      return x
    else:
      return [x] * self.n


def Parallel(device_names_or_functions, fn, *args):
  """Deprecated interface.

  Use `Parallelism(device_names_or_functions)(fn, *args)` instead.

  Args:
    device_names_or_functions: A list of length n.
    fn: a function or a list of n functions.
    *args: additional args.  Each arg should either be not a list, or a list
       of length n.

  Returns:
    either a single list of length n (if fn does not return a tuple), or a
    tuple of lists of length n (if fn returns a tuple).
  """
  return Parallelism(device_names_or_functions)(fn, *args)

def TransposeListOfLists(lol):
  """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  """
  assert lol, 'cannot pass the empty list'
  return [list(x) for x in zip(*lol)]

