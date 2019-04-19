#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import collections
import numpy as np
import tensorflow as tf
import optimization
from tag_proj import bert_model

flags = tf.flags
FLAGS = flags.FLAGS

# required params
flags.DEFINE_string("bert_config_file", None, "the bert config file")
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the data files for the task.")
flags.DEFINE_string("output_dir", None, "output dir")

flags.DEFINE_integer("max_seq_length", None, "the max ad records a user have")
flags.DEFINE_integer("image_embedding_size", None, "image embedding size")
flags.DEFINE_integer("text_embedding_size", None, "text embedding size")

# other params
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 5, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 2, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 2, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 500,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")


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


def pad_embedding(embedding, max_length):
    """ ad-lists have different size, padding them to max length """
    # andy: for debug
    # tf.logging.info(embedding.shape)
    embed_num, embed_size = embedding.shape
    zero_embedding = np.zeros(shape=[max_length-embed_num, embed_size])
    return np.concatenate([embedding, zero_embedding], axis=0)


def truncate_embedding(embedding, max_length):
    return embedding[:max_length, :]


class EmbeddingFileProcessor(object):
    """read file, get examples"""
    @classmethod
    def _read_txt(cls, embedding_file):
        """ a single line is an example """
        with tf.gfile.Open(embedding_file, "r") as f:
            lines = f.readlines()

        examples = []
        for ex_index, line in enumerate(lines):
            image_embedding_list = []
            text_embedding_list = []

            # TODO(andy), convert the string to float numbers, assumed file format
            guid = ex_index

            elements = line.split(";")
            label = elements[0].split(",")
            label = [eval(x) for x in label]
            for embedding_pair in elements[1:]:

                image_embedding = embedding_pair.split("##")[0]
                text_embedding = embedding_pair.split("##")[1]
                image_embedding = image_embedding.split(",")
                text_embedding = text_embedding.split(",")
                image_embedding = [eval(x) for x in image_embedding]
                text_embedding = [eval(x) for x in text_embedding]

                image_embedding_list.append(image_embedding)
                text_embedding_list.append(text_embedding)
            input_example = InputExample(guid, image_embedding_list, text_embedding_list, label=label)
            examples.append(input_example)
        return examples

    @classmethod
    def get_train_examples(cls, data_dir):
        return cls._read_txt(data_dir)

    @classmethod
    def get_eval_examples(cls, data_dir):
        return cls._read_txt(data_dir)

    @classmethod
    def get_test_examples(cls, data_dir):
        return cls._read_txt(data_dir)

    @classmethod
    def get_labels(cls):
        """See base class."""
        return [0, 1]


class InputExample(object):
    """single ad example, including image embeddings & text embeddings, read from .txt file directly"""
    def __init__(self, guid, image_embedding_a, text_embedding_a,
                 image_embedding_b=None, text_embedding_b=None, label=None):
        assert len(image_embedding_a) == len(text_embedding_a), "text, image embedding must be same size"
        if image_embedding_b and text_embedding_b:
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


def convert_single_example(ex_index, example, label_list):
    """ convert the example to features"""
    if ex_index % 10 == 0:
        tf.logging.info("convert %d example" % ex_index)

    max_seq_length = FLAGS.max_seq_length
    image_embed_size = FLAGS.image_embedding_size
    text_embed_size = FLAGS.text_embedding_size

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            image_embedding=np.zeros(shape=[max_seq_length, image_embed_size]),
            text_embedding=np.zeros(shape=[max_seq_length, text_embed_size]),
            input_masks=np.zeros(shape=[max_seq_length]),
            segment_ids=np.zeros(shape=[max_seq_length]),
            # TODO(andy)
            label_id=np.zeros(shape=[len(label_list)]),
            is_real_example=False
        )

    # avoid some labels are string
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

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
        current_length += 1

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

    feature_label = np.array(example.label)
    feature = InputFeatures(image_embeddings, text_embeddings, input_masks, segment_ids, feature_label)
    return feature


def convert_examples_to_features(examples, label_list):
    features = []
    for ex_index, example in enumerate(examples):
        feature = convert_single_example(ex_index, example, label_list)
        features.append(feature)
    return features


def file_based_convert_examples_to_features(examples, label_list, output_file):
    """write examples to a TFRecordFile"""
    writer = tf.python_io.TFRecordWriter(output_file)

    for ex_index, example in enumerate(examples):
        feature = convert_single_example(ex_index, example, label_list)

        def create_feature(feature_values, feature_name="input_embeddings"):
            f = None
            if feature_name in ["input_masks", "segment_ids", "is_real_example", "label_id"]:
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feature_values)))
            elif feature_name == "input_embeddings":
                f = tf.train.Feature(float_list=tf.train.FloatList(value=list(feature_values)))
            else:
                raise ValueError("the feature name is wrong")
            return f
        features = collections.OrderedDict()

        embedding_value = feature.get_embedding().reshape([-1])
        label_value = feature.label_id.reshape([-1])

        features["input_embeddings"] = create_feature(embedding_value, "input_embeddings")
        features["input_masks"] = create_feature(feature.input_masks, "input_masks")
        features["segment_ids"] = create_feature(feature.segment_ids, "segment_ids")
        features["label_id"] = create_feature(label_value, "label_id")
        features["is_real_example"] = create_feature([int(feature.is_real_example)], "is_real_example")
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()
    tf.logging.info("Write all examples to %s done" % output_file)


def input_fn_builder(features, seq_length, embed_size, num_labels, is_training, drop_remainder):
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
            # TODO(andy) the label dimension should support multi-label classification, i.e. [num_examples, num_labels]
            "label_id": tf.constant(all_label, shape=[num_examples, num_labels], dtype=tf.int64),
        })
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def file_based_input_fn_builder(input_file, seq_length, embed_size, num_labels, is_training, drop_remainder):
    """create an `input_fn` closure to be passed to TPUEstimator"""
    name_to_features = {
        "input_embeddings": tf.FixedLenFeature(shape=[seq_length, embed_size], dtype=tf.float32),
        "input_masks": tf.FixedLenFeature(shape=[seq_length], dtype=tf.int64),
        "segment_ids": tf.FixedLenFeature(shape=[seq_length], dtype=tf.int64),

        # TODO(andy) the label dimension should support multi-label classification, i.e. [num_labels]
        "label_id": tf.FixedLenFeature(shape=[num_labels], dtype=tf.int64),
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
                drop_remainder=drop_remainder))
        return d

    return input_fn


def direct_input_fn_builder(hdfs_address=""):
    """given a hdfs address, read and return batch data immediately"""

    def input_fn(params):
        batch_size = params["batch_size"]
        processor = EmbeddingFileProcessor()
        label_list = processor.get_labels()
        examples = processor.get_train_examples(FLAGS.data_dir)
        batch_embedding = []
        batch_mask = []
        batch_segment = []
        batch_label = []
        example_num = 0
        features = {}
        for ex_index, example in enumerate(examples):
            feature = convert_single_example(ex_index, example, label_list)

            batch_embedding.append(feature.get_embedding())
            batch_mask.append(feature.input_masks)
            batch_segment.append(feature.segment_ids)
            batch_label.append(feature.label_id)
            example_num += 1
            if example_num % batch_size == 0:
                features["input_embeddings"] = tf.constant(np.array(batch_embedding), shape=[batch_size, 64, 24], dtype=tf.float32)
                features["input_masks"] = tf.constant(np.array(batch_mask), shape=[batch_size, 64], dtype=tf.int64)
                features["segment_ids"] = tf.constant(np.array(batch_segment), shape=[batch_size, 64], dtype=tf.int64)
                features["label_id"] = tf.constant(np.array(batch_label), shape=[batch_size, 2], dtype=tf.int64)
                batch_embedding = []
                batch_mask = []
                batch_segment = []
                batch_label = []
                yield features
                features = {}
                
    return input_fn


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu):

    def model_fn(features, labels, mode, params):

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name=%s, shape=%s" % (name, features[name].shape))

        # batch data
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

        # create model to get loss, logits, probabilities
        # TODO(andy) label_id should support multi-label classification, i.e. [batch, num_labels]
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

        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        # one_hot_label = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        # TODO(andy) calc multi-label probabilities
        probabilities = tf.sigmoid(logits)

        # tf.nn.sigmoid_cross_entropy_with_logits op, both labels and logits are float
        labels = tf.cast(labels, dtype=tf.float32)

        # multi-label classification, logits and label must be same shape
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

        per_example_loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

    return loss, per_example_loss, logits, probabilities


def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("at least one of `do_train`, `do_eval` and `do_predict` is true")

    bert_config = bert_model.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = EmbeddingFileProcessor()
    label_list = processor.get_labels()

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,

        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    embed_size = FLAGS.image_embedding_size + FLAGS.text_embedding_size

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

        # consider how to read data not via TFRecord
        file_based_convert_examples_to_features(train_examples, label_list, train_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Epoch = %d", FLAGS.num_train_epochs)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        # consider how to read data not via TFRecord
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            embed_size=embed_size,
            num_labels=len(label_list),
            is_training=True,
            drop_remainder=True)

        # train_input_fn = direct_input_fn_builder()
        tf.logging.info("Begin training")
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(eval_examples, label_list, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            embed_size=embed_size,
            num_labels=len(label_list),
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            embed_size=embed_size,
            num_labels=len(label_list),
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]

                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == '__main__':
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("max_seq_length")
    flags.mark_flag_as_required("image_embedding_size")
    flags.mark_flag_as_required("text_embedding_size")
    tf.app.run()

