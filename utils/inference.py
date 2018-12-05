"""Implemetation of beam seach with penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from models import common_layers
from models import common_attention
from utils import parallel

import tensorflow as tf
from tensorflow.python.util import nest
import operator
import time
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.training import saver


# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7

def decode_from_file(predict_func, hparams, decode_from_file, decode_to_file, decode_batch_size, decode_beam_size, decode_return_beams):
    """Compute predictions on entries in filename and write them out."""
    problem_id = 0
    inputs_vocab = hparams.vocabulary["inputs"]
    targets_vocab = hparams.vocabulary["targets"]
    sorted_inputs, sorted_keys = _get_sorted_inputs(decode_from_file)
    num_decode_batches = (len(sorted_inputs) - 1) // decode_batch_size + 1
    input_fn = _decode_batch_input_fn(problem_id, num_decode_batches,
            sorted_inputs, inputs_vocab, decode_batch_size)

    def _save_until_eos(hyp):  #  pylint: disable=missing-docstring
        ret = []
        index = 1
        # until you reach <EOS> id
        while index < len(hyp) and hyp[index] != 1:
            ret.append(hyp[index])
            index += 1
        ##If a R2L hypothesis wins, we reverse the tokens before presenting it
        if hyp[0] == 3: ## 3:<r2l>, 2:<l2r>
            ret.reverse()
        return np.array(ret)

    decodes = []
    scores = []

    start = time.clock()
    for input_x in input_fn:
        result_iter = predict_func(feed_fn=input_x) # !!!

        for result in result_iter:
            def log_fn(inputs, outputs):
                decoded_inputs = inputs_vocab.decode(_save_until_eos(inputs.flatten()))
                tf.logging.info("Inference results INPUT: %s" % decoded_inputs)

                decoded_outputs = targets_vocab.decode(
                    _save_until_eos(outputs.flatten()))
                tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
                return decoded_outputs

            if decode_return_beams:
                beam_decodes = []
                output_beams = np.split(
                    result["outputs"], decode_beam_size, axis=0)
                for k, beam in enumerate(output_beams):
                    tf.logging.info("BEAM %d:" % k)
                    beam_decodes.append(log_fn(result["inputs"], beam))
                decodes.append(str.join("\n", beam_decodes))

                result["scores"] = [str(m) for m in result["scores"]]
                scores.append("\n".join(result["scores"]))
            else:
                decodes.append(log_fn(result["inputs"], result["outputs"]))
                #scores.append(result["scores"][0])

    sorted_inputs.reverse()
    decodes.reverse()

    decode_filename = decode_to_file
    decode_filename_score = decode_to_file + ".score" ##!!

    tf.logging.info("Writing decodes into %s" % decode_filename)
    outfile = tf.gfile.Open(decode_filename, "w")
    #outfile_score = tf.gfile.Open(decode_filename_score, "w")
    for index in range(len(sorted_inputs)):
        outfile.write("%s\n" % (decodes[sorted_keys[index]]))
        #outfile_score.write("%s\n" % (scores[sorted_keys[index]]))

    elapsed = (time.clock() - start)
    print("Time used:",elapsed)

def _decode_batch_input_fn(problem_id, num_decode_batches, sorted_inputs,
                           vocabulary, decode_batch_size):
    tf.logging.info(" batch %d" % num_decode_batches)
    sorted_inputs.reverse()
    for b in range(num_decode_batches):
        tf.logging.info("Deocding batch %d" % b)
        batch_length = 0
        batch_inputs = []
        for inputs in sorted_inputs[b * decode_batch_size:(
                b + 1) * decode_batch_size]:
            input_ids = vocabulary.encode(inputs, replace_oov="UNK")
            input_ids.append(1)  # Assuming EOS=1.
            batch_inputs.append(input_ids)
            if len(input_ids) > batch_length:
                batch_length = len(input_ids)
        final_batch_inputs = []
        for input_ids in batch_inputs:
            assert len(input_ids) <= batch_length
            x = input_ids + [0] * (batch_length - len(input_ids))
            final_batch_inputs.append(x)
        yield {
            "inputs": np.array(final_batch_inputs).astype(np.int32),
            "problem_choice": np.array(problem_id).astype(np.int32),
        }

def _get_sorted_inputs(filename):
    """Returning inputs sorted according to length.
    """
    tf.logging.info("Getting sorted inputs")
    decode_filename = filename

    inputs = [line.strip() for line in tf.gfile.Open(decode_filename)]
    input_lens = [(i, len(line.strip().split())) for i, line in enumerate(inputs)]
    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
    # We'll need the keys to rearrange the inputs back into their original order
    sorted_keys = {}
    sorted_inputs = []
    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i
    return sorted_inputs, sorted_keys


##############################################
#beam search #

def shape_list(x):
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

def _merge_beam_dim(tensor):
    """Reshapes first two dimensions in to single dimension.
        tensor: Tensor to reshape of shape [A, B, ...] --> [A*B, ...]
    """
    shape = shape_list(tensor)
    batch_size = shape[0]
    beam_size = shape[1]
    newshape = [batch_size*beam_size]+shape[2:]
    return tf.reshape(tensor, [batch_size*beam_size]+shape[2:])

def _unmerge_beam_dim(tensor, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].
        [batch_size*beam_size, ...] --> [batch_size, beam_size, ...]
    """
    shape = shape_list(tensor)
    new_shape = [batch_size]+[beam_size]+shape[1:]
    return tf.reshape(tensor, new_shape)

def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.
        tensor: tensor to tile [batch_size, ...] --> [batch_size, beam_size, ...]
    """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size
    return tf.tile(tensor, tile_dims)

def log_prob_from_logits(logits):
    return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)

def compute_batch_indices(batch_size, beam_size):
    """Computes the i'th coodinate that contains the batch index for gathers.
    like [[0,0,0,0,],[1,1,1,1],..]
    """
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos

def get_state_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


def beam_search(predict_next_symbols,
                initial_ids,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                states=None,
                eos_id=EOS_ID,
                stop_early=True):
    """Beam search with length penalties.
    """
    batch_size = shape_list(initial_ids)[0]
    #initial_log_probs = tf.constant([[0.] + [-float("inf")] * (beam_size - 1)])
    initial_log_probs = tf.constant([[0.] + (int(beam_size/2)-1)*[-float("inf")] + [0.] + (int(beam_size/2)-1)*[-float("inf")]])
    alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1]) # (batch_size, beam_size)
  
    #alive_seq = _expand_to_beam_size(initial_ids, beam_size) # (batch, beam)
    #alive_seq = tf.expand_dims(alive_seq, axis=2)  # (batch_size, beam_size, 1)

    initial_ids_1 = 2*tf.ones([batch_size,1], dtype=tf.int32) ## index 2 == <l2r>
    initial_ids_2 = 3*tf.ones([batch_size,1], dtype=tf.int32) ## index 3 == <r2l>

    initial_ids_1 = tf.concat([initial_ids, initial_ids_1], axis=1) ## [batch, 2]
    initial_ids_2 = tf.concat([initial_ids, initial_ids_2], axis=1)

    alive_seq_1 = tf.tile(tf.expand_dims(initial_ids_1, 1), [1, tf.cast(beam_size/2, tf.int32), 1]) ## [batch, beam/2, 2]
    alive_seq_2 = tf.tile(tf.expand_dims(initial_ids_2, 1), [1, tf.cast(beam_size/2, tf.int32), 1])
    alive_seq = tf.concat([alive_seq_1, alive_seq_2], axis=1) ## [batch, beam, 2]

    states = nest.map_structure(
        lambda state: _expand_to_beam_size(state, beam_size), states)
    finished_seq = tf.zeros(shape_list(alive_seq), tf.int32)
    finished_scores = tf.ones([batch_size, beam_size]) * -INF
    finished_flags = tf.zeros([batch_size, beam_size], tf.bool)
    finished_flags = tf.zeros([batch_size, beam_size], tf.bool)

    def _beam_search_step(i, alive_seq, alive_log_probs, finished_seq, finished_scores,
                 finished_flags, states):
        """Inner beam seach loop.
        """
        ## 1. Get the current topk items.
        flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])
        flat_states = nest.map_structure(_merge_beam_dim, states)
        flat_logits, flat_states = predict_next_symbols(flat_ids, i, batch_size, beam_size, flat_states) # !!
        states = nest.map_structure(
                lambda t: _unmerge_beam_dim(t, batch_size, beam_size), flat_states)

        logits = tf.reshape(flat_logits, [batch_size, beam_size, -1]) # (batch, beam, vocab)
        candidate_log_probs = log_prob_from_logits(logits) # softmax
        log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)
        length_penalty = tf.pow(((5. + tf.to_float(i + 1)) / 6.), alpha)
        curr_scores = log_probs / length_penalty
        #flat_curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size]) # (batch, beam*vocab)
        flat_curr_scores = tf.reshape(curr_scores, [-1, 2, tf.cast(beam_size/2, tf.int32) * vocab_size]) ## [batch, 2, (beam/2) * vocab]
        #topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size * 2) # (batch, 2*beam)
        topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size) ## [batch, 2, beam]
        topk_log_probs = topk_scores * length_penalty
        topk_log_probs = tf.reshape(topk_log_probs, [-1, 2*beam_size]) ## add; [batch, 2*beam]
        topk_scores = tf.reshape(topk_scores, [-1, 2*beam_size]) ## add;
        topk_beam_index = topk_ids // vocab_size ## like [[0,1,1,0],[1,1,0,0],[1,0,0,0],...], e.g. beam=2
        topk_ids %= vocab_size  # Unflatten the ids

        ##add;
        topk_beam_index_1 = tf.concat([tf.expand_dims(topk_beam_index[:,0,:],1), tf.expand_dims(topk_beam_index[:,1,:]+tf.cast(beam_size/2,tf.int32),1)], axis=1)
        topk_beam_index = tf.reshape(topk_beam_index_1, [-1, beam_size*2])
        topk_ids = tf.reshape(topk_ids, [-1, beam_size*2])

        batch_pos = compute_batch_indices(batch_size, beam_size * 2) # like [[0,0,0,0,],[1,1,1,1],[2,2,2,2],...] (batch, 2*beam) 
        topk_coordinates = tf.stack([batch_pos, topk_beam_index], axis=2) # like [[[0,0],[0,1],[0,1],[0,0]], [[1,1],[1,1],[1,0],[1,0]], [[2,1],[2,0],[2,0],[2,0]],...]  (batch, 2*beam, 2)
        topk_seq = tf.gather_nd(alive_seq, topk_coordinates) # (batch, 2*beam, lenght)
        states = nest.map_structure(
                lambda state: tf.gather_nd(state, topk_coordinates), states)
        topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2) # (batch, 2*beam, length+1)
        topk_finished = tf.equal(topk_ids, eos_id) # (batch, 2*beam)

        # 2. Extract the ones that have finished and haven't finished
        curr_scores = topk_scores + tf.to_float(topk_finished) * -INF # (batch, 2*beam)
        curr_scores = tf.reshape(curr_scores, [batch_size, 2, beam_size])
        #_, topk_indexes = tf.nn.top_k(curr_scores, k=beam_size) # (batch, beam)
        _, topk_indexes = tf.nn.top_k(curr_scores, k=tf.cast(beam_size/2, tf.int32)) ## [batch, 2, beam/2]
        topk_indexes_tmp = topk_indexes[:,1,:]+beam_size
        topk_indexes = tf.concat([tf.expand_dims(topk_indexes[:,0,:],1), tf.expand_dims(topk_indexes_tmp,1)], axis=1)
        topk_indexes = tf.reshape(topk_indexes, [batch_size, beam_size])

        batch_pos_2 = compute_batch_indices(batch_size, beam_size)
        top_coordinates = tf.stack([batch_pos_2, topk_indexes], axis=2) # (batch, beam, 2) 
        alive_seq = tf.gather_nd(topk_seq, top_coordinates)
        alive_log_probs = tf.gather_nd(topk_log_probs, top_coordinates)
        alive_states = nest.map_structure(
                lambda state: tf.gather_nd(state, top_coordinates), states)
     
        # 3. Recompute the contents of finished based on scores.
        finished_seq = tf.concat(
            [finished_seq,
            tf.zeros([batch_size, beam_size, 1], tf.int32)], axis=2)
        curr_scores = topk_scores + (1. - tf.to_float(topk_finished)) * -INF
        curr_finished_seq = tf.concat([finished_seq, topk_seq], axis=1)
        curr_finished_scores = tf.concat([finished_scores, curr_scores], axis=1)
        curr_finished_flags = tf.concat([finished_flags, topk_finished], axis=1)
    
        _, topk_indexes = tf.nn.top_k(curr_finished_scores, k=beam_size)
        top_coordinates = tf.stack([batch_pos_2, topk_indexes], axis=2)
        finished_seq = tf.gather_nd(curr_finished_seq, top_coordinates)
        finished_flags = tf.gather_nd(curr_finished_flags, top_coordinates)
        finished_scores = tf.gather_nd(curr_finished_scores, top_coordinates)

        return (i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores,
                finished_flags, alive_states)

    def _is_finished(i, unused_alive_seq, alive_log_probs, unused_finished_seq,
                   finished_scores, finished_flags, unused_states):
        """Checking termination condition.
        """
        if not stop_early:
            return tf.less(i, decode_length)
        max_length_penalty = tf.pow(((5. + tf.to_float(decode_length)) / 6.), alpha)
        lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty
        lowest_score_of_finished = tf.reduce_min(
                finished_scores * tf.to_float(finished_flags), axis=1)
        lowest_score_of_finished += (
                (1. - tf.to_float(tf.reduce_any(finished_flags, 1))) * -INF)
        bound_is_met = tf.reduce_all( # return True when lowest_score_of_finished > lower_bound_alive_scores
                tf.greater(lowest_score_of_finished, lower_bound_alive_scores))

        return tf.logical_and( # return True(do not finish) when i<decode_length and lowest_score_of_finished<lower_bound_alive_scores 
                tf.less(i, decode_length), tf.logical_not(bound_is_met))

    (_, alive_seq, alive_log_probs, finished_seq, finished_scores,
            finished_flags, _) = tf.while_loop(
        _is_finished, # termination when return False
        _beam_search_step, [
            tf.constant(0), alive_seq, alive_log_probs, finished_seq,
            finished_scores, finished_flags, states],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, None, None]),
            alive_log_probs.get_shape(),
            tf.TensorShape([None, None, None]),
            finished_scores.get_shape(),
            finished_flags.get_shape(),
            nest.map_structure(
                #lambda tensor: tf.TensorShape(tensor.shape), states)],
                lambda tensor: get_state_shape_invariants(tensor), states)],
        parallel_iterations=1,
        back_prop=False)

    alive_seq.set_shape((None, beam_size, None)) # (batch, beam, lenght)
    finished_seq.set_shape((None, beam_size, None))

    finished_seq = tf.where(
        tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
    finished_scores = tf.where(
        tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
    return finished_seq, finished_scores


################################## infer

def _beam_decode(features, decode_length, beam_size, top_beams, alpha, local_features):

    decoded_ids, scores = _fast_decode(features, decode_length,
                                            beam_size, top_beams, alpha, local_features)
    return {"outputs": decoded_ids, "scores": scores}

def _greedy_infer(features, decode_length, local_features):
    
    decoded_ids, _ = _fast_decode(features, decode_length, 1, 1, 1, local_features)
    return decoded_ids


def _fast_decode(features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0,
                   local_features=None):
    """Fast decoding.
    """
    if local_features["_num_datashards"] != 1:
        raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = local_features["_data_parallelism"]
    hparams = local_features["_hparams"]

    inputs = features["inputs"]
    batch_size = tf.shape(features["inputs"])[0]
    target_modality = local_features["_hparams"].target_modality
    decode_length = tf.shape(features["inputs"])[1] + tf.constant(decode_length)

    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
    s = tf.shape(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match
    inputs = local_features["_shard_features"]({"inputs": inputs})["inputs"]
    input_modality = local_features["_hparams"].input_modality
    with tf.variable_scope(input_modality.name):
        inputs = input_modality.bottom_sharded(inputs, dp)
    with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
                local_features["encode"], inputs, hparams)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

    if hparams.pos == "timing": #####
        timing_signal = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
        """Performs preprocessing steps on the targets to prepare for the decoder.
        Returns: Processed targets [batch_size, 1, hidden_dim]
        """
        # _shard_features called to ensure that the variable names match
        targets = local_features["_shard_features"]({"targets": targets})["targets"]
        with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
        targets = common_layers.flatten4d3d(targets)

        # TODO(llion): Explain! Is this even needed?
        #targets = tf.cond(
        #        tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)
        targets = tf.cond(
            tf.equal(i, 0), lambda: tf.concat([tf.zeros_like(targets)[:,:1,:],targets[:,1:,:]], axis=1), lambda: targets)
        
        #if hparams.pos == "timing":
        #    targets += timing_signal[:, i:i + 1]
        if hparams.pos == "timing":
            timing_signal_1 = tf.cond(
                        tf.equal(i, 0), lambda: timing_signal[:, i:i + 2], lambda: timing_signal[:, i+1:i + 2])
            targets += timing_signal_1
        return targets

    decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length+1))

    def predict_next_symbols(ids, i, batch_size, beam_size, cache):
        """Go from ids to logits for next symbol."""
        #tf.logging.info("target id's shape is {0}".format(ids.shape))
        #ids = ids[:, -1:]    # only need the last time target input
        ids = tf.cond(
                    tf.equal(i, 0), lambda: ids[:, -2:], lambda: ids[:, -1:])
        #tf.logging.info("after target id's shape is {0}".format(ids.shape))
        targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
        targets = preprocess_targets(targets, i)

        #bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
        bias_1 = decoder_self_attention_bias[:, :, i:i + 2, :i + 2]
        bias_2 = decoder_self_attention_bias[:, :, i+1:i + 2, :i + 2]
        bias = tf.cond(
                    tf.equal(i, 0), lambda: bias_1, lambda: bias_2)
        
        s = tf.shape(cache['encoder_output'])
        cache['encoder_output'] = tf.reshape(cache['encoder_output'],[s[0],s[1],hparams.hidden_size])
        #tf.logging.info('encoder_output is {0}'.format(cache['encoder_output'].shape))
        with tf.variable_scope("body"):
            body_outputs = dp(
                local_features["decode"], targets, cache["encoder_output"],
                    cache["encoder_decoder_attention_bias"], bias, hparams, batch_size, beam_size, cache)

        #tf.logging.info("body_output before softmax shape is {0}".format(body_outputs[0].shape))
        with tf.variable_scope(target_modality.name):
            logits = target_modality.top_sharded(body_outputs, None, dp)[0]
            
        tf.logging.info("logits's shape is {0}".format(logits[0].shape))
        #return tf.squeeze(logits, axis=[0, 2, 3]), cache
        return tf.squeeze(logits, axis=[0, 3])[:,-1,:], cache

    key_channels = hparams.hidden_size
    value_channels = hparams.hidden_size
    num_layers = hparams.num_hidden_layers

    cache = {
        "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, key_channels]),
                "v": tf.zeros([batch_size, 0, value_channels]),
        }
        for layer in range(num_layers)
    }

    for layer in cache:
        #cache[layer]["k"]._shape = tf.TensorShape([None, None, key_channels])
        #cache[layer]["v"]._shape = tf.TensorShape([None, None, value_channels])
        cache[layer]["k"].set_shape = tf.TensorShape([None, None, key_channels])
        cache[layer]["v"].set_shape = tf.TensorShape([None, None, value_channels])
    # pylint: enable=protected-access
    cache["encoder_output"] = encoder_output

    #tf.logging.info("cache['encoder_output'] is {0}".format(cache['encoder_output'].shape))
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    if beam_size > 1:  # Beam Search
        target_modality = (
                local_features["_hparams"].target_modality)
        vocab_size = target_modality.top_dimensionality
        #initial_ids = tf.zeros([batch_size], dtype=tf.int32)
        initial_ids = tf.zeros([batch_size,1], dtype=tf.int32)
        decoded_ids, scores = beam_search(
                predict_next_symbols,
                initial_ids,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                states=cache,
                stop_early=(top_beams == 1))

        if top_beams == 1:
            decoded_ids = decoded_ids[:, 0, 1:]
        else:
            decoded_ids = decoded_ids[:, :top_beams, 1:]
    else:  # Greedy
          #tf.logging.info("##########greedy_infer!!!##########")
        def inner_loop(i, next_id, decoded_ids, cache):
            logits, cache = predict_next_symbols(next_id, i, cache)

            next_id = tf.argmax(logits, -1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            #print ("decoded_ids: ", decoded_ids)
            return i + 1, next_id, decoded_ids, cache

        decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
        scores = None
        next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
        _, _, decoded_ids, _ = tf.while_loop(
                # TODO(llion): Early stopping.
                lambda i, *_: tf.less(i, decode_length),
                inner_loop,
                [tf.constant(0), next_id, decoded_ids, cache],
                shape_invariants=[
                    tf.TensorShape([]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    #nest.map_structure(lambda t: tf.TensorShape(t.shape), cache),
                    nest.map_structure(lambda t: get_state_shape_invariants(t), cache),
                ])

    return decoded_ids, scores

