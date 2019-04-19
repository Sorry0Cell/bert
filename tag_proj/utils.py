
import logging
import threading
import datetime
import os
import json
from queue import Queue
import numpy as np
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

# required params

# flags.DEFINE_string("file_dir", None, "Ad files path")
#
# flags.DEFINE_integer("max_ads", None, "Max ads num per example")
# flags.DEFINE_integer("min_ads", None, "Min ads num per example")
# flags.DEFINE_integer("max_tags", None, "Max tags num per ad")
#
# flags.DEFINE_integer("batch_size", 15, "batch size")
# flags.DEFINE_string("log_file", "ad_data/20190417_DEBUG_utils.log", "Log file path")


def load_vocab(path="ad_data/dict.txt"):
    """
    load vocabulary
    :param path:
    :return: vocab dict, word with its occurance
    """
    vocab = {}
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            word, occur = line.strip().split('\t')
            occur = int(occur)
            # if occur > 10000:
            #     continue
            vocab[word] = occur
    return vocab


def load_tag2idx(path="ad_data/dict.json"):
    return json.load(open(path, "r"))


def parse_ad_line(line):
    """
    each line is a sample, multi ads, each ad has multi tags
    each line: date; user; clk time; ad tag and ad weight
    :param line:
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
    if np.random.rand() < 0.2:
        return []

    vocab_size = len(tag_2_idx)

    # padding with vocab_size+1
    x_ad_tag_ids = np.ones([FLAGS.max_ads, FLAGS.max_tags]) * (vocab_size+1)
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
    def __init__(self, file_list, batch_size, vocab_size, line_func,
                 max_queue_size=512, num_workers=1):
        """
        :param file_list:
        :param batch_size:
        :param vocab_size:
        :param line_func:
        :param max_queue_size:
        :param num_workers:
        """

        self.max_queue_size = max_queue_size
        self.file_list = file_list
        self.num_workers = num_workers
        self.batch_size = batch_size
        # line_func return a parsed training example
        self.line_func = line_func
        self.vocab_size = vocab_size

    def parse_file(self):
        while True:
            filename = self.filename_queue.get()
            if filename is None:
                return

            logging.info("Processing FILE: %s" % filename)

            with tf.gfile.Open(filename, 'r') as f:
                for line in f:
                    parsed_example = self.line_func(line)
                    if parsed_example:
                        self.examples_queue.put(parsed_example)
            self.examples_queue.put(None)

    def get_batch(self, files_num):
        count_end = 0
        batch_example = []
        while True:
            example = self.examples_queue.get()
            # example is a list
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
            logging.info("batch from remainder ...")
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

        logging.info("example_list: %d " % len(example_list))

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

    def infinite_iteration(self):
        while True:
            for batch_example in self.iteration():
                yield batch_example


def main(_):
    file_list = []
    ad_files = tf.gfile.ListDirectory(FLAGS.file_dir)
    for file in ad_files:
        if file.startswith("part"):
            file_list.append(os.path.join(FLAGS.file_dir, file))

    loader = DataLoader(file_list, batch_size=FLAGS.batch_size,
                        vocab_size=len(tag_2_idx), line_func=parse_ad_line)
    loader = loader.infinite_iteration()

    while True:
        try:
            feed_dict = next(loader)
            logging.info("x_ad_tag_ids shape {}".format(feed_dict["x_ad_tag_ids"].shape))
            logging.info("y_ad_label shape {}".format(feed_dict["y_ad_label"].shape))
            logging.info("x_ad_tag_num shape {}".format(feed_dict["x_ad_tag_num"].shape))
            logging.info("x_ad_num shape     {}".format(feed_dict["x_ad_num"].shape))
        except StopIteration:
            break


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                        filename=FLAGS.log_file)

    flags.mark_flag_as_required("file_dir")
    flags.mark_flag_as_required("max_ads")
    flags.mark_flag_as_required("min_ads")
    flags.mark_flag_as_required("max_tags")

    tag_2_idx = load_tag2idx()

    tf.app.run()


