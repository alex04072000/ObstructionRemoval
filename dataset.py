"""Implements a dataset class for handling image data"""

from image_utils import imread


class Dataset(object):
    def __init__(self, data_list_file=None, process_func=None):
        """
          Args:
        """
        self.data_list_file = data_list_file

    def read_data_list_file(self):
        """Reads the data list_file into python list
        """
        f = open(self.data_list_file)
        data_list = [line.rstrip() for line in f]
        self.data_list = data_list
        return data_list

    def process_func(self, example_line):
        return imread(example_line)