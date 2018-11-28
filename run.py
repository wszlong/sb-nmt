#!/usr/bin/env python

"""
Author: Long Zhou. 
Email: long.zhou@nlpr.ia.ac.cn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys
import random
import tempfile
import numpy as np

import tensorflow as tf

from utils import generator_utils
from utils import trainer_utils as trainer_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("generate_data", False, "Generate data before training?")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen", "Temporary storage directory.")
flags.DEFINE_integer("num_shards", 10, "How many shards to use.")
flags.DEFINE_integer("max_cases", 0, "Maximum number of cases to generate (unbounded if 0).")
flags.DEFINE_integer("random_seed", 429459, "Random seed to use.")

UNSHUFFLED_SUFFIX = "-unshuffled"

_SUPPORTED_PROBLEM_GENERATORS = {
    "translation": (
        lambda: generator_utils.translation_token_generator(FLAGS.data_dir, FLAGS.tmp_dir, 
            FLAGS.train_src_name, FLAGS.train_tgt_name, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name) )
}

def set_random_seed():
    """Set the random seed from flag everywhere."""
    tf.set_random_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

def generate_data():
    data_dir = os.path.expanduser(FLAGS.data_dir)
    tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
    tf.gfile.MakeDirs(data_dir)
    tf.gfile.MakeDirs(tmp_dir)
  
    problem = list(sorted(_SUPPORTED_PROBLEM_GENERATORS))[0]
    set_random_seed()

    training_gen = _SUPPORTED_PROBLEM_GENERATORS[problem]

    tf.logging.info("Generating training data for %s.", problem)
    train_output_files = generator_utils.generate_files(
            training_gen(), problem + UNSHUFFLED_SUFFIX + "-train",
            FLAGS.data_dir, FLAGS.num_shards, FLAGS.max_cases)

    train_output_files = []
    output_dir = FLAGS.data_dir
    for shard in xrange(FLAGS.num_shards):
        output_filename = "%s-%.5d-of-%.5d" % ('translation-unshuffled-train', shard, FLAGS.num_shards)
        output_file = os.path.join(output_dir, output_filename)
        train_output_files.append(output_file)

    tf.logging.info("Shuffling data...")
    #for fname in train_output_files + dev_output_files:
    for fname in train_output_files:
        records = generator_utils.read_records(fname)
        random.shuffle(records)
        out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
        generator_utils.write_records(records, out_fname)
        tf.gfile.Remove(fname)
    tf.logging.info("Data Process Over")


############

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    #utils.validate_flags()

    if FLAGS.generate_data:
        generate_data()
        if FLAGS.model != "transformer":  ## no train
            return
  
    trainer_utils.run(
            data_dir=FLAGS.data_dir,
            model=FLAGS.model, ##transformer
            output_dir=FLAGS.output_dir,
            train_steps=FLAGS.train_steps)

if __name__ == "__main__":
    tf.app.run()
