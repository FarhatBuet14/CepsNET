from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import six

import numpy as np
import time
import json
import io
import sys

from collections import OrderedDict


class CSVLogger():

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.logs = None
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        super(CSVLogger, self).__init__()

    def on_train_begin(self):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs, keys = None, ):

        if self.keys is None:
            self.keys = list(logs.keys())

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, logs[key]) for key in list(self.keys))
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self):
        self.csv_file.close()
        self.writer = None