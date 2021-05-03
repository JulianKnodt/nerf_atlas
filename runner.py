# Global runner for all NeRF methods.
# For convenience, we want all methods using NeRF to use this one file.

import argparse

def arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data", help="path to data")
  parser.add_argument("--device", help="device to use for processing")
  # TODO add more arguments here
  return parser.parse_args()


# loads the dataset
def load(kind="original"):
  ...

def main():
  args = arguments()


