#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:50:00 2023

@author: eo
"""


#%% Imports

import json
import os.path as osp


#%% Functions

def make_history_path(current_script_folder_path, history_file_name = "selection_history.json"):
    parent_dir = osp.dirname(current_script_folder_path)
    history_file_path = osp.join(parent_dir, history_file_name)
    return history_file_path
    
def load_history_file(dunder_file):
    
    # Initialize output
    history_dict = {}
    
    # If the file doesn't exist, create it
    load_path = make_history_path(dunder_file)
    if not osp.exists(load_path):
        with open(load_path, "w") as out_file:
            json.dump(history_dict, out_file)
    
    # Load existing file, which should exit, since we just created it if it didn't
    with open(load_path, "r") as in_file:
        history_dict = json.load(in_file)
    
    return history_dict

def save_history_file(dunder_file, history_dict):
    
    save_path = make_history_path(dunder_file)
    with open(save_path, "w") as out_file:
        json.dump(history_dict, out_file)
    
    return save_path
