#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:52:19 2023

@author: eo
"""


#%% Imports

import os
import os.path as osp
import gzip
import json

import cv2
import numpy as np

from collections import defaultdict


#%% Data handling functions

def jsongz_to_dict(load_path):
    data_dict = None
    with gzip.open(load_path, "rt", encoding="UTF-8") as in_file:
        data_dict = json.load(in_file)
    return data_dict

def json_to_dict(load_path):
    data_dict = None
    with open(load_path, "r") as in_file:
        data_dict = json.load(in_file)
    return data_dict

def load_data_dict(load_path):
    is_compressed = load_path.endswith(".json.gz")        
    return jsongz_to_dict(load_path) if is_compressed else json_to_dict(load_path)

def load_one_metadata(data_folder_path, filename_no_ext, target_file_ext = ".json.gz"):
    
    load_path = osp.join(data_folder_path, "{}{}".format(filename_no_ext, target_file_ext))
    if not osp.exists(load_path):
        raise FileNotFoundError("No metadata @ {}".format(load_path))
    
    return load_data_dict(load_path)


#%% Data-specific functions

def get_report_folder_paths(locations_folder_path, camera_select):
    
    report_folder_path = osp.expanduser(osp.join(locations_folder_path, camera_select, "report"))
    
    snap_img_folder_path = osp.join(report_folder_path, "images", "snapshots")
    bg_img_folder_path = osp.join(report_folder_path, "images", "backgrounds")
    obj_md_folder_path = osp.join(report_folder_path, "metadata", "objects")
    
    return snap_img_folder_path, bg_img_folder_path, obj_md_folder_path

def load_trail_data(data_folder_path, target_file_exts = {".gz", ".json"}):
    
    # Build loading paths for every file we're interested in
    all_files = os.listdir(data_folder_path)
    is_target_file = lambda path: osp.splitext(path)[1] in target_file_exts
    all_target_files = (each_file for each_file in all_files if is_target_file(each_file))
    all_target_paths = (osp.join(data_folder_path, each_file) for each_file in all_target_files)
    
    # Store each object data using the object ID as a key
    trails_by_class_and_id = defaultdict(dict)
    for each_path in all_target_paths:
        each_obj_data_dict = load_data_dict(each_path)
        obj_id = each_obj_data_dict["_id"]
        obj_bdb_class = each_obj_data_dict["bdb_classifier"]
        class_label = obj_bdb_class if type(obj_bdb_class) is str else "unclassified"
        trails_by_class_and_id[class_label][obj_id] = {"xy_center": each_obj_data_dict["tracking"]["xy_center"],
                                                       "first_epoch_ms": each_obj_data_dict["first_epoch_ms"],
                                                       "final_epoch_ms": each_obj_data_dict["final_epoch_ms"]}
    
    ex_obj = load_data_dict(each_path)
    return trails_by_class_and_id, ex_obj

def load_snapshot_image(image_folder_path, snap_ems, target_image_ext = ".jpg"):
    load_path = osp.join(image_folder_path, "{}{}".format(snap_ems, target_image_ext))
    if not osp.exists(load_path):
        print("", "Error with snapshot loading path", "@ {}".format(load_path), "", sep = "\n")
        raise FileNotFoundError()
    return cv2.imread(load_path)

def get_final_snap(snaps_ems_array, target_ems):
    closest_idx = np.where(np.sort(snaps_ems_array) > target_ems)[0][0]
    return snaps_ems_array[closest_idx]
