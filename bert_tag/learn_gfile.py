#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import json


def main():
    with tf.gfile.Open("test.txt", "w") as f:
        f.write("123")


if __name__ == '__main__':
    main()
