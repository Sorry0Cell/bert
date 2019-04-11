#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import numpy as np
import tensorflow as tf
import optimization
from data_preparation import bert_model


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_length", 64, "the max ad records a user have")
flags.DEFINE_integer("image_embedding_size", None, "image embedding size")
flags.DEFINE_integer("text_embedding_size", None, "text embedding size")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")


def _truncate_seq_pair(image_a, text_a, image_b, text_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(image_a) + len(image_b)
        if total_length <= max_length:
            break
        if len(image_a) > len(image_b):
            image_a.pop()
            text_a.pop()
        else:
            image_b.pop()
            text_b.pop()


def pad_embedding(embedding, max_seq_length):
    """ ad-lists have different size, padding them to max length """
    embed_num, embed_size = embedding.shape
    zero_embedding = np.zeros(shape=[max_seq_length-embed_num, embed_size])
    return np.concatenate([embedding, zero_embedding], axis=0)


def truncate_embedding(embedding, max_seq_length):
    return embedding[:max_seq_length, :]


class EmbeddingFileProcessor(object):

    @classmethod
    def _read_txt(cls, embedding_file):
        """ a single line is an example """
        with tf.gfile.Open(embedding_file, "r") as f:
            lines = f.readlines()

        examples = []
        for line in lines:
            image_embedding_list = []
            text_embedding_list = []

            # TODO(andy), convert the string to float numbers, assumed file format
            guid = line[0]
            label = line[1]
            embeddings = line[2, -1].split(";")

            for embedding_pair in embeddings:
                image_embedding = embedding_pair[0]
                text_embedding = embedding_pair[1]

                image_embedding_list.append(image_embedding)
                text_embedding_list.append(text_embedding)
            input_example = InputExample(guid, image_embedding_list, text_embedding_list, label)
            examples.append(input_example)
        return examples


class InputExample(object):
    """single ad example, including image embeddings & text embeddings, read from .txt file directly"""
    def __init__(self, guid, image_embedding_a, text_embedding_a,
                 image_embedding_b=None, text_embedding_b=None, label=None):
        assert len(image_embedding_a) == len(text_embedding_a), "text, image embedding must be same size"
        assert len(image_embedding_b) == len(text_embedding_b), "text, image embedding must be same size"
        self.guid = guid
        self.image_embedding_a = image_embedding_a    # with shape (num_examples, ad_size, image_embedding_size)
        self.image_embedding_b = image_embedding_b
        self.text_embedding_a = text_embedding_a      # with shape (num_examples, ad_size, text_embedding_size)
        self.text_embedding_b = text_embedding_b
        self.label = label


class PaddingInputExample(object):
    """ Fake example so the num input examples is a multiple of the batch size """


class InputFeatures(object):
    """single ad feature"""
    def __init__(self, image_embedding, text_embedding, input_masks, segment_ids, label_id, combine_type="concat",
                 is_real_example=True):
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.combine_type = combine_type
        self.label_id = label_id
        self.is_real_example = is_real_example

    def get_embedding(self):
        embedding = self.combine_embedding()
        return embedding

    def combine_embedding(self):
        if self.combine_type == "concat":
            return np.concatenate([self.image_embedding, self.text_embedding], axis=1)
        else:
            raise TypeError("combine type only support concat")


def convert_single_example(ex_index, example):
    """ convert the example to features"""
    if ex_index % 1000 == 0:
        tf.logging.info("convert %d example" % ex_index)

    max_seq_length = FLAGS.max_length
    image_embed_size = FLAGS.image_embedding_size
    text_embed_size = FLAGS.text_embedding_size

    if isinstance(example, PaddingInputExample):

        return InputFeatures(
            image_embedding=np.zeros(shape=[max_seq_length, image_embed_size]),
            text_embedding=np.zeros(shape=[max_seq_length, text_embed_size]),
            input_masks=np.zeros(shape=[max_seq_length]),
            segment_ids=np.zeros(shape=[max_seq_length]),
            label_id=0,
            is_real_example=False
        )

    # A example has 2 ad-embedding, each ad-embedding includes image and text,  pair task
    image_embedding_a = example.image_embedding_a
    text_embedding_a = example.text_embedding_a

    image_embedding_b = None
    text_embedding_b = None
    if example.image_embedding_b is not None and example.text_embedding_b is not None:
        image_embedding_b = example.image_embedding_b
        text_embedding_b = example.text_embedding_b

    if image_embedding_b and text_embedding_b:
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(image_embedding_a, text_embedding_a, image_embedding_b, text_embedding_b, max_seq_length-3)
    elif len(image_embedding_a) > max_seq_length-2:
        image_embedding_a = image_embedding_a[: (max_seq_length-2)]
        text_embedding_a = text_embedding_a[: (max_seq_length-2)]

    # [CLS], represent it by zeros
    image_embeddings = [[0]*image_embed_size]
    text_embeddings = [[0]*text_embed_size]
    segment_ids = [0]

    for i in range(len(image_embedding_a)):
        image_embeddings.append(image_embedding_a[i])
        text_embeddings.append(text_embedding_a[i])
        segment_ids.append(0)
    # [SEP], represent it by zeros
    image_embeddings.append([0]*image_embed_size)
    text_embeddings.append([0]*text_embed_size)
    segment_ids.append(0)

    if image_embedding_b and text_embedding_b:
        for i in range(len(image_embedding_b)):
            image_embeddings.append(image_embedding_b[i])
            text_embeddings.append(text_embedding_b[i])
            segment_ids.append(1)
        # [SEP], represent it by zeros
        image_embeddings.append([0] * image_embed_size)
        text_embeddings.append([0] * text_embed_size)
        segment_ids.append(1)

    current_length = len(image_embeddings)
    input_masks = [1] * current_length
    while current_length < max_seq_length:
        input_masks.append(0)
        segment_ids.append(0)

    image_embeddings = np.array(image_embeddings)
    text_embeddings = np.array(text_embeddings)
    input_masks = np.array(input_masks)
    segment_ids = np.array(segment_ids)

    if image_embeddings.shape[0] < max_seq_length:
        image_embeddings = pad_embedding(image_embeddings, max_seq_length)
        text_embeddings = pad_embedding(text_embeddings, max_seq_length)

    assert image_embeddings.shape[0] == max_seq_length
    assert text_embeddings.shape[0] == max_seq_length
    assert input_masks.shape[0] == max_seq_length
    assert segment_ids.shape[0] == max_seq_length

    feature = InputFeatures(image_embeddings, text_embeddings, input_masks, segment_ids, example.label)
    return feature


def convert_examples_to_features(examples):
    features = []
    for ex_index, example in enumerate(examples):
        feature = convert_single_example(ex_index, example)
        features.append(feature)
    return features


def file_based_convert_examples_to_features(examples, output_file):
    """write examples to a TFRecordDataset"""
    writer = tf.python_io.TFRecordWriter(output_file)

    for ex_index, example in enumerate(examples):
        feature = convert_single_example(ex_index, example)

        def create_feature(feature_values, feature_name="input_embeddings"):
            f = None
            if feature_name in ["input_masks", "segment_ids", "label_id", "is_real_example"]:
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature_values)))
            elif feature_name == "input_embeddings":
                f = tf.train.Feature(float_list=tf.train.FloatList(value=list(feature_values)))
            else:
                raise TypeError("the feature name is wrong")
            return f
        features = collections.OrderedDict()
        embedding_value = feature.get_embedding().reshape(shape=[-1])
        features["input_embeddings"] = create_feature(embedding_value, "input_embeddings")
        features["input_masks"] = create_feature(feature.input_masks, "input_masks")
        features["segment_ids"] = create_feature(feature.segment_ids, "segment_ids")
        features["label_id"] = create_feature([feature.label_id], "label_id")
        features["is_real_example"] = create_feature([int(feature.is_real_example)], "is_real_example")
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()
    tf.logging.info("Write all examples to %s done" % output_file)


def input_fn_builder(features, seq_length, embed_size, is_training, drop_remainder):
    """for large dataset, use tf.data.TFRecordDataset instead"""
    all_input_embedding = []
    all_input_masks = []
    all_segment_ids = []
    all_label = []

    for feature in features:
        all_input_embedding.append(feature.get_embedding())
        all_input_masks.append(feature.input_masks)
        all_segment_ids.append(feature.segment_ids)
        all_label.append(feature.label_id)

    def input_fn(params):
        batch_size = params["batch_size"]
        num_examples = len(features)
        # embed_size = FLAGS.image_embedding_size + FLAGS.text_embedding_size
        d = tf.data.Dataset.from_tensor_slices({
            "input_embeddings": tf.constant(all_input_embedding, shape=[num_examples, seq_length, embed_size],
                                            dtype=tf.float32),
            "input_masks": tf.constant(all_input_masks, shape=[num_examples, seq_length],
                                       dtype=tf.int64),
            "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length],
                                       dtype=tf.int64),
            "label_id": tf.constant(all_label, shape=[num_examples], dtype=tf.int64),
        })
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def file_based_input_fn_builder(input_file, seq_length, embed_size, is_training, drop_reminder):
    """create an `input_fn` closure to be passed to TPUEstimator"""
    name_to_features = {
        "input_embeddings": tf.FixedLenFeature(shape=[seq_length, embed_size], dtype=tf.float32),
        "input_masks": tf.FixedLenFeature(shape=[seq_length], dtype=tf.int64),
        "segment_ids": tf.FixedLenFeature(shape=[seq_length], dtype=tf.int64),
        "label_id": tf.FixedLenFeature(shape=[], dtype=tf.int64),
        "is_real_example": tf.FixedLenFeature(shape=[], dtype=tf.int64)
    }

    # tf.example only support tf.int64, but TPU only support tf.int32
    def _decode_record(record, features):
        example = tf.parse_single_example(record, features=features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_reminder=drop_reminder))
        return d
    return input_fn


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu):

    def model_fn(features, labels, mode, params):

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name=%s, shape=%s" % (name, features[name].shape))

        input_embeddings = features["input_embeddings"]
        input_masks = features["input_masks"]
        segment_ids = features["segment_ids"]
        label_id = features["label_id"]
        is_real_example = None

        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(shape=tf.shape(label_id), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # TODO(andy), BertModel constructor need to be modified
        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_embeddings, input_masks, segment_ids, label_id,
            num_labels)

        tvars = tf.trainable_variables()
        initalized_variable_names = {}
        scaffold_fn = None

        if init_checkpoint:
            assignment_map, initialized_variable_names = bert_model.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ***")
        for var in tvars:
            init_string = ""
            # show which variables are initialized from checkpoint
            if var.name in initalized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(" name = %s, shape = %s%s" % (var.name, var.shape, init_string))

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_id, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_id, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "loss": loss
                }
            eval_metrics = (metric_fn, [per_example_loss, label_id, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def create_model(bert_config, is_training, input_embeddings, input_masks, segment_ids, labels,
                 num_labels):
    model = bert_model.BertModel(
        config=bert_config,
        is_training=is_training,
        input_embeddings=input_embeddings,
        input_masks=input_masks,
        token_type_ids=segment_ids)

    outpue_layer = model.get_pooled_output()
    hidden_size = outpue_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.truncated_normal_initializer(stddev=0.02))

    with tf.variable_scope("loss"):
        if is_training:
            outpue_layer = tf.nn.dropout(outpue_layer, keep_prob=0.9)

        logits = tf.matmul(outpue_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_label = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_label * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

    return loss, per_example_loss, logits, probabilities


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == '__main__':
    flags.mark_flag_as_required("image_embedding_size")
    flags.mark_flag_as_required("text_embedding_size")
    tf.app.run()


