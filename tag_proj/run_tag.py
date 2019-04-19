#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tag_proj.bert_model import BertModel
from tag_proj.bert_model import BertConfig
from tag_proj.bert_model import get_assignment_map_from_checkpoint

from datetime import datetime
from queue import Queue

import os
import json
import logging
import threading
import numpy as np
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

# required params
flags.DEFINE_string("out_dir", "out_dir", "Path to save results")


flags.DEFINE_string("bert_config", None, "BertConfig file")

flags.DEFINE_integer("max_ads", None, "Max ads num per example")
flags.DEFINE_integer("min_ads", None, "Min ads num per example")
flags.DEFINE_integer("max_tags", None, "Max tags num per ad")

# other params
flags.DEFINE_integer("max_iter", 5000, "Max iter for training")

flags.DEFINE_integer("train_batch_size", 64, "Train batch size")
flags.DEFINE_integer("eval_batch_size", 32, "Eval batch size")
flags.DEFINE_integer("predict_batch_size", 32, "Predict batch size")

flags.DEFINE_float("init_lr", 5e-5, "Init learning rate")
flags.DEFINE_bool("do_train", False, "Whether to train")
flags.DEFINE_bool("do_eval", True, "Whether to eval")
flags.DEFINE_bool("do_predict", False, "Whether to predict")

flags.DEFINE_string("train_file_dir", None, "Training ad files dir path")
flags.DEFINE_string("eval_file_dir", None, "Eval ad files dir path")
flags.DEFINE_string("predict_file_dir", None, "Predict ad files dir path")

flags.DEFINE_string("init_checkpoint", None, "Path to init_checkpoint file")


def load_tag2idx(path="ad_data/dict.json"):
    return json.load(open(path, "r"))


def get_file_list(file_dir):
    ad_file_list = []
    ad_files = tf.gfile.ListDirectory(file_dir)

    for file in ad_files:
        if file.startswith("part"):
            ad_file_list.append(os.path.join(file_dir, file))
    return ad_file_list


def parse_ad_line(line, is_training):
    """
    each line is a sample, multi ads, each ad has multi tags
    each line: date; user; clk time; ad tag and ad weight
    :param line:
    :param is_training:
    :return:
    """
    # date, user_id, ad_series
    ads = line.strip().split("\t")[2]
    ads = ads.split(";")
    # remove empty ads
    ads = [ad for ad in ads if ad.split("#")[2] != ""]

    if len(ads) < FLAGS.min_ads or len(ads) > FLAGS.max_ads:
        return []

    # negative sample
    if is_training and np.random.rand() < 0.2:
        return []

    vocab_size = len(tag_2_idx)

    # padding with vocab_size+1
    x_ad_tag_ids = np.ones([FLAGS.max_ads, FLAGS.max_tags]) * (vocab_size+1)    # initialize with [PAD]
    x_ad_tag_num = np.zeros([FLAGS.max_ads])
    y_ad_label = np.zeros([vocab_size+2])       # [OOV] & [PAD]

    for i, ad in enumerate(ads):

        _, _, tags = ad.split("#")

        if tags == "":
            x_ad_tag_ids[i:0] = vocab_size
        else:
            tags = tags.split(",")
            for j, tag in enumerate(tags):
                tag_w = tag.split("_")
                if len(tag_w) == 2:
                    word, weight = tag_w
                    word_idx = tag_2_idx.get(word, vocab_size)  # OOV is represented by vocab_size
                else:
                    word_idx = vocab_size
                if i != len(ads)-1:
                    x_ad_tag_ids[i, j] = word_idx
                else:
                    y_ad_label[word_idx] = 1      # multi label classification task
            # the last ad is label, do not count it
            if i < len(ads)-1:
                x_ad_tag_num[i] = len(tags)

    x_ad_num = len(ads)-1
    x_ad_mask = [1] * x_ad_num
    while len(x_ad_mask) < FLAGS.max_ads:
        x_ad_mask.append(0)
    x_ad_mask = np.array(x_ad_mask)

    return [x_ad_tag_ids, x_ad_tag_num, x_ad_mask, x_ad_num, y_ad_label]


class DataLoader(object):
    """
    offer huge data for train or eval
    """
    def __init__(self, file_list, batch_size, line_func,
                 is_training=True, max_queue_size=512, num_workers=1):
        """
        :param file_list:
        :param batch_size:
        :param line_func:
        :param max_queue_size:
        :param num_workers:
        """
        self.file_list = file_list
        self.batch_size = batch_size
        # line_func return a parsed training example
        self.line_func = line_func
        self.is_training = is_training
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers

    def parse_file(self):
        """
        parse each file into training examples
        :return:
        """
        while True:
            filename = self.filename_queue.get()
            if filename is None:
                return

            logging.info("Processing FILE: %s" % filename)

            with tf.gfile.Open(filename, 'r') as f:
                for line in f:
                    parsed_example = self.line_func(line, self.is_training)
                    if parsed_example:
                        self.examples_queue.put(parsed_example)
            self.examples_queue.put(None)       # flag, finish parsing a file

    def get_batch(self, files_num):
        """
        :param files_num:
        :return: batch training examples
        """
        count_end = 0
        batch_example = []
        while True:
            example = self.examples_queue.get()

            if example is None:
                count_end += 1
                if count_end == files_num:
                    break
            else:
                batch_example.append(example)
                if len(batch_example) == self.batch_size:

                    yield self.feed_data(batch_example)
                    batch_example = []

        # remainder
        if len(batch_example) > 0:
            logging.info("Batch Data from remainder.")
            yield self.feed_data(batch_example)

    @classmethod
    def feed_data(cls, example_list):
        """
        :param example_list: each element is [x_ad_tag_ids, y_ad_label, x_ad_tag_num, x_ad_num]
        :return: feed_dict to train or eval
        """
        x_ad_tag_ids = []
        x_ad_tag_num = []
        x_ad_mask = []
        x_ad_num = []
        y_ad_label = []

        for example in example_list:

            x_ad_tag_ids.append(example[0])
            x_ad_tag_num.append(example[1])
            x_ad_mask.append(example[2])
            x_ad_num.append(example[3])
            y_ad_label.append(example[4])

        feed_dict = {
            "x_ad_tag_ids": np.array(x_ad_tag_ids),
            "x_ad_tag_num": np.array(x_ad_tag_num),
            "x_ad_mask": np.array(x_ad_mask),
            "x_ad_num": np.array(x_ad_num),
            "y_ad_label": np.array(y_ad_label),
        }

        return feed_dict

    def iteration(self):
        self.filename_queue = Queue()
        self.examples_queue = Queue(maxsize=self.max_queue_size)

        thread_pool = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self.parse_file)
            t.daemon = True
            t.start()
            thread_pool.append(t)

        for filename in self.file_list:
            self.filename_queue.put(filename)

        # denote the end
        for i in range(self.num_workers):
            self.filename_queue.put(None)

        for batch_example in self.get_batch(len(self.file_list)):
            yield batch_example

        for thread in thread_pool:
            thread.join()

    # support multi epoch
    def infinite_iteration(self, epoch=1):
        while epoch > 0:
            for batch_example in self.iteration():
                yield batch_example
            epoch -= 1


def main(_):

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one mode (Train, Eval, Predict) is specified ")

    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)

    log_file = os.path.join(FLAGS.out_dir, "BertTag_DEBUG_{}.log".format(
        datetime.now().strftime("%Y-%m-%d %H%M%S")))
    logging.basicConfig(format='%(asctime)s - %(filename)s - %(levelname)s : %(message)s', level=logging.DEBUG,
                        filename=log_file)

    bert_config = BertConfig.from_json_file(FLAGS.bert_config)
    model = BertModel(bert_config, FLAGS.max_ads, FLAGS.max_tags, is_training=False)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}

    if FLAGS.init_checkpoint:
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
            tvars=tvars, init_checkpoint=FLAGS.init_checkpoint
        )
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

    tf.logging.info("*** Trainable variables ***")
    for var in tvars:
        init_str = ""
        if var.name in initialized_variable_names:
            init_str = "*INIT_FROM_CKPT*"
        tf.logging.info("var name: {}, var shape: {}{}".format(var.name, var.shape, init_str))

    if FLAGS.do_train:
        if FLAGS.train_file_dir is None:
            raise ValueError("Please give a train files directory")

        train_file_list = get_file_list(FLAGS.train_file_dir)
        assert len(train_file_list) > 0
        train_loader = DataLoader(train_file_list, batch_size=FLAGS.train_batch_size,
                                  line_func=parse_ad_line, num_workers=4)

        train_data = train_loader.infinite_iteration(epoch=5)
        model.train(train_data, out_dir=FLAGS.out_dir, init_lr=FLAGS.init_lr, max_iter=FLAGS.max_iter)

        # TODO(andy), when training, also need eval

    if FLAGS.do_eval:
        if FLAGS.init_checkpoint is None:
            raise ValueError("Init checkpoint must be specified in EVAL MODE")
        if FLAGS.eval_file_dir is None:
            raise ValueError("Please give an eval files directory")

        eval_file_list = get_file_list(FLAGS.eval_file_dir)
        assert len(eval_file_list) > 0
        eval_loader = DataLoader(eval_file_list, batch_size=FLAGS.eval_batch_size,
                                 line_func=parse_ad_line, is_training=False, num_workers=1)

        eval_data = eval_loader.iteration()
        precision, recall = model.eval(eval_data=eval_data, out_dir=FLAGS.out_dir, k=5)
        logging.info("Precision: {}, Recall: {}".format(precision, recall))

    if FLAGS.do_predict:
        if FLAGS.init_checkpoint is None:
            raise ValueError("init checkpoint must be specified in PREDICT MODE")
        if FLAGS.predict_file_dir is None:
            raise ValueError("please give an predict directory")

        predict_file_list = get_file_list(FLAGS.predict_file_dir)
        assert len(predict_file_list) > 0
        predict_loader = DataLoader(predict_file_list, batch_size=FLAGS.predict_batch_size,
                                    line_func=parse_ad_line, is_training=False, num_workers=1)

        predict_data = predict_loader.iteration()
        model.predict(predict_data=predict_data, out_dir=FLAGS.out_dir)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)

    flags.mark_flag_as_required("out_dir")

    flags.mark_flag_as_required("train_file_dir")
    flags.mark_flag_as_required("bert_config")

    flags.mark_flag_as_required("max_ads")
    flags.mark_flag_as_required("min_ads")
    flags.mark_flag_as_required("max_tags")

    tag_2_idx = load_tag2idx()

    tf.app.run()
