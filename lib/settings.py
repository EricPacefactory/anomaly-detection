#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:50:00 2023

@author: eo
"""


#%% Imports

import json
import os.path as osp


#%% Classes

class SettingsDict:
    
    ''' Helper used to make dictionary that stores default values of missing keys '''
    
    def __init__(self, dunder_file = None):
        self._dict = {}
        if dunder_file is not None:
            self.load(dunder_file)
        pass
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, value):
        self._dict[key] = value
    
    def get(self, key_name, default_value):
        
        # Get the key value if possible, otherwise use default and store for re-use
        value = self._dict.get(key_name, None)
        if value is None:
            value = default_value
        self._dict[key_name] = value
        
        return value
    
    def load(self, dunder_file):
        
        # If the file doesn't exist, create it
        load_path = self.make_file_path(dunder_file)
        if not osp.exists(load_path):
            with open(load_path, "w") as out_file:
                json.dump(self._dict, out_file)
        
        # Load existing file, which should exist, since we just created it if it didn't
        with open(load_path, "r") as in_file:
            self._dict = json.load(in_file)
        
        return self
    
    def save(self, dunder_file):
        save_path = self.make_file_path(dunder_file)
        with open(save_path, "w") as out_file:
            json.dump(self._dict, out_file, indent = 2)
        return save_path
    
    @staticmethod
    def make_file_path(dunder_file, settingsfile_name = "settings.json"):
        parent_dir = osp.dirname(dunder_file)
        file_path = osp.join(parent_dir, settingsfile_name)
        return file_path

