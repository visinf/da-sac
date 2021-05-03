"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0

Description:
This script will read all masks recursively
and save to a file a dictionary
    
    mask_basename -> pixel_dictionary,

where pixel_dictionary is 
    
    class_id -> number of class pixels in the mask / total number of class pixels

Example run:
    python tools/compute_IS_weights.py --labels <path/to/masks/png> --out <output_weights.data>
"""

import torch
import fnmatch
import os
import sys
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm

#
# Arguments
#

parser = argparse.ArgumentParser(description="Count pixels")

parser.add_argument("--labels", type=str, default='./data/labels',
                    help="Path to directory with labels")
parser.add_argument("--ext", type=str, default='*labelIds.png',
                    help="File extension")
parser.add_argument("--out", type=str, default='./data/weights.data',
                    help="Path to file list")

def count(path, ext, out):

    if os.path.isfile(out):
        print("Output file already exists: {}".format(out))
        sys.exit(1)

    # search for all PNGs
    matches = []
    filenames = os.listdir(path)
    for filename in fnmatch.filter(filenames, ext):
        matches.append(os.path.join(path, filename))

    print("Found {} masks".format(len(matches)))
    mask_stats = {}
    unique_labels = {}
    pixel_count = {} # number of pixels per class
    num_images = {}  # number of images with the class
    for filepath in tqdm(matches):
        image = np.array(Image.open(filepath))
        labels = np.unique(image)
        label_stats = {}

        for label in labels:

            if label == 255:
                continue
            
            if not label in unique_labels:
                unique_labels[label] = True

            label_stats[label] = (label == image).astype(np.float).sum()

            if not label in pixel_count:
                pixel_count[label] = 0.

            pixel_count[label] += label_stats[label]

            if not label in num_images:
                num_images[label] = 0

            num_images[label] += 1

        sample_id = os.path.basename(filepath)
        mask_stats[sample_id] = label_stats

    # reporting pixel count
    print("Pixel count / # of Images: ")
    for key in sorted(pixel_count):
        print("Class {:02d}: {:2.1f} {}".format(key, pixel_count[key], num_images[key]))

    # normalising
    for sample_id, label_stats in mask_stats.items():

        for label in label_stats.keys():
            label_stats[label] /= pixel_count[label]

    torch.save(mask_stats, out)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    count(args.labels, args.ext, args.out)
