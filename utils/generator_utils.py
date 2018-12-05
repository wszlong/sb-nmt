"""Utilities for data generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import io
import os
import tarfile

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import six.moves.urllib_request as urllib  # Imports urllib on Python2, urllib.request on Python3
import tensorflow as tf

from utils import text_reader
PAD = 0
EOS = 1
L2R = 2
R2L = 3

def token_generator_three(source_path, target_path_l2r, target_path_r2l, token_vocab_src, token_vocab_tgt, eos=None):
    """Generator for sequence-to-sequence tasks that uses tokens.
    """
    eos_list = [] if eos is None else [eos]
    pad_list = [] if PAD is None else [PAD]
    l2r_list = [] if L2R is None else [L2R]
    r2l_list = [] if R2L is None else [R2L]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path_l2r, mode="r") as target_file_l2r:
            with tf.gfile.GFile(target_path_r2l, mode="r") as target_file_r2l:

                source, target_l2r, target_r2l = source_file.readline(), target_file_l2r.readline(), target_file_r2l.readline()
                while source and target_l2r and target_r2l:
                    source_ints = token_vocab_src.encode(source.strip()) + eos_list 
        
                    t_l2r = token_vocab_tgt.encode(target_l2r.strip())
                    t_r2l = token_vocab_tgt.encode(target_r2l.strip())
                    t_length_max = max(len(t_l2r),len(t_r2l))
                    t_l2r_add_len = t_length_max - len(t_l2r)
                    t_r2l_add_len = t_length_max - len(t_r2l)
                    ## let len(target_ints_l2r)==len(target_ints_r2l)
                    target_ints_l2r = l2r_list + t_l2r + t_l2r_add_len*pad_list + eos_list
                    target_ints_r2l = r2l_list + t_r2l + t_r2l_add_len*pad_list + eos_list

                    yield {"inputs": source_ints, "targets_l2r": target_ints_l2r, "targets_r2l": target_ints_r2l}
                    source, target_l2r, target_r2l = source_file.readline(), target_file_l2r.readline(), target_file_r2l.readline()


def translation_token_generator(data_dir, tmp_dir, train_src_name, train_tgt_name, vocab_src_name, vocab_tgt_name):
  
    train_src_path = os.path.join(tmp_dir, train_src_name)
    train_tgt_path_l2r = os.path.join(tmp_dir, train_tgt_name + ".l2r")
    train_tgt_path_r2l = os.path.join(tmp_dir, train_tgt_name + ".r2l")

    token_vocab_src_dir = os.path.join(data_dir, vocab_src_name)
    token_vocab_tgt_dir = os.path.join(data_dir, vocab_tgt_name)
    if not tf.gfile.Exists(token_vocab_src_dir):
        tf.gfile.Copy(os.path.join(tmp_dir, vocab_src_name), token_vocab_src_dir)
    if not tf.gfile.Exists(token_vocab_tgt_dir):
        tf.gfile.Copy(os.path.join(tmp_dir, vocab_tgt_name), token_vocab_tgt_dir)

    token_vocab_src = text_reader.TokenTextEncoder(vocab_filename=token_vocab_src_dir)
    token_vocab_tgt = text_reader.TokenTextEncoder(vocab_filename=token_vocab_tgt_dir)
    return token_generator_three(train_src_path, train_tgt_path_l2r, train_tgt_path_r2l, token_vocab_src, token_vocab_tgt, 1)


###########################################################

def to_example(dictionary):
    """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
    features = {}
    for (k, v) in six.iteritems(dictionary):
        if not v:
            raise ValueError("Empty generated field: %s", str((k, v)))
        if isinstance(v[0], six.integer_types):
            features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
        elif isinstance(v[0], float):
            features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
        elif isinstance(v[0], six.string_types):
            if not six.PY2:  # Convert in python 3.
                v = [bytes(x, "utf-8") for x in v]
            features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
        elif isinstance(v[0], bytes):
            features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
        else:
            raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                    (k, str(v[0]), str(type(v[0]))))
    return tf.train.Example(features=tf.train.Features(feature=features))


def generate_files(generator,
                   output_name,
                   output_dir,
                   num_shards=1,
                   max_cases=None):
    """Generate cases from a generator and save as TFRecord files.
    """
    writers = []
    output_files = []
    for shard in xrange(num_shards):
        output_filename = "%s-%.5d-of-%.5d" % (output_name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        output_files.append(output_file)
        writers.append(tf.python_io.TFRecordWriter(output_file))

    counter, shard = 0, 0
    for case in generator:
        if counter > 0 and counter % 100000 == 0:
            tf.logging.info("Generating case %d for %s." % (counter, output_name))
        counter += 1
        if max_cases and counter > max_cases:
            break
        sequence_example = to_example(case)
        writers[shard].write(sequence_example.SerializeToString())
        shard = (shard + 1) % num_shards

    for writer in writers:
        writer.close()

    return output_files

def read_records(filename):
    reader = tf.python_io.tf_record_iterator(filename)
    records = []
    for record in reader:
        records.append(record)
    if len(records) % 100000 == 0:
        tf.logging.info("read: %d", len(records))
    return records


def write_records(records, out_filename):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for count, record in enumerate(records):
        writer.write(record)
    if count > 0 and count % 100000 == 0:
        tf.logging.info("write: %d", count)
    writer.close()

