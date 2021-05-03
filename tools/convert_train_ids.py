"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import sys
import os
import argparse
import numpy as np
import fnmatch
import tqdm

from PIL import Image
from id_mappers import get_mapper

#
# Arguments
#

parser = argparse.ArgumentParser(description="Label ID pre-processing")

parser.add_argument("--dataset", choices=['cs', 'gta', 'synthia'],
                    help="Dataset name. One of cs|gta|synthia")
parser.add_argument("--ann-data", type=str, default='./data/labels',
                    help="The path to the annotation file")
parser.add_argument("--ann-out", type=str, default='./data/annotation_out',
                    help="The path where to save the filtered filepaths")

def check_dir(path):
    if not os.path.isdir(path):
        print("Creating {}".format(path))
        os.makedirs(path)

def convert_to_target(filepath, data_ann_path_out, target_map):
    """
    Saves the image as a .png in the data path
    """

    if not os.path.isfile(filepath):
        print("No such file found: ", filepath)
        return False

    ann_file = os.path.basename(filepath)

    # open the image
    mask = target_map.read(filepath)
    source_labels = set(np.unique(mask))

    num_added = 0
    new_mask = np.ones_like(mask) * target_map.UNLABELLED
    for cat_id in source_labels:

        # skip the category if not in the mapping
        if not cat_id in target_map.MAP:
            continue

        cat_mask = mask == cat_id
        new_mask[cat_mask] = target_map.MAP[cat_id]
        num_added += 1

    if num_added == 0:
        return False

    im = Image.fromarray(new_mask).convert("L")
    im.save(os.path.join(data_ann_path_out, ann_file))

    return True
    

def preprocess(args):
    """Iterate through annotations;
        read Synthia classes, convert to .png,
        substitute with classes in the target set
    """

    mapper = get_mapper(args.dataset)

    # checking dirs
    check_dir(args.ann_out)

    # searching for files .png
    filelist_in = []
    for root, dirnames, filenames in os.walk(args.ann_data):
        for filename in fnmatch.filter(filenames, mapper.ext()):
            subdir = root.replace(args.ann_data, '').lstrip("/")
            filelist_in.append((os.path.join(root, filename), subdir))
    
    print("Found {:d} files".format(len(filelist_in)))

    filelist = []
    num_processed = 0
    for full_path, subdir in tqdm.tqdm(filelist_in):

        ann_out = os.path.join(args.ann_out, subdir)
        check_dir(ann_out)
        if convert_to_target(full_path, ann_out, mapper):
            num_processed += 1

    print("Processed {} files".format(num_processed))

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    preprocess(args)
