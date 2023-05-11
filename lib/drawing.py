#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:50:18 2023

@author: eo
"""


#%% Imports

import cv2
import numpy as np


#%%

class MouseClickCB:
    
    ''' Helper class used to add callback to window that reports click position '''
    
    def __init__(self, xy_min, xy_max):
        self.xy = (0, 0)
        self._prev_xy = (-1, -1)
        self._xmin, self._ymin = xy_min
        self._xmax, self._ymax = xy_max
        self._is_down = False
        
    def __call__(self, event, x, y, flags, param):
        
        # Don't respond to out-of-bounds events
        oob_x = (x >= self._xmax) or (x < self._xmin)
        oob_y = (y >= self._ymax) or (y < self._ymin)
        if oob_x or oob_y:
            return
        
        # Keep track of mouse positioning
        self._prev_xy = self.xy
        self.xy = (x,y)
        
        # Record new click event + xy
        if event == cv2.EVENT_LBUTTONDOWN:
            self._is_down = True
        
        # End click/drag events
        if event == cv2.EVENT_LBUTTONUP:
            self._is_down = False
        
        return
    
    def is_down(self):
        return self._is_down, self.xy
    
    def get_line_points(self):
        return np.int32([self._prev_xy, self.xy])

#%% Functions

def scale_to_max_side_length(image, max_side_length = 800):
    img_h, img_w = image.shape[0:2]
    max_dim = max(image.shape)
    if max_dim < max_side_length:
        return image
    
    scale_factor = max_side_length / max_dim
    new_w = round(img_w * scale_factor)
    new_h = round(img_h * scale_factor)
    return cv2.resize(image, dsize = [new_w, new_h])

def draw_centered_text(frame, text_str, scale, color, use_bg = True):
    
    # For clarity
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    # Figure out how big our text will be, so we can properly offset it when trying to center on frame
    (text_w, text_h), text_baseline = cv2.getTextSize(text_str, font, scale, thickness)
    
    # Figure out text positioning
    frame_h, frame_w = frame.shape[0:2]
    text_x = round((frame_w - text_w) * 0.5)
    text_y = round(text_baseline + frame_h * 0.5)
    text_pos = (text_x, text_y)
    
    # Draw text, with background if needed
    if use_bg:
        bg_color = (0, 0, 0)
        bg_thickness = 2 + thickness
        cv2.putText(frame, text_str, text_pos, font, scale, bg_color, bg_thickness, cv2.LINE_AA)
    cv2.putText(frame, text_str, text_pos, font, scale, color, thickness, cv2.LINE_AA)
    
    return frame

def add_header_image(display_image, header_text,
                     header_height = 40, header_bg_value = (40,40,40), text_color = (255,255,255), text_scale = 0.5):
    
    disp_w = display_image.shape[1]
    header_img = np.full((header_height, disp_w, 3), header_bg_value, dtype = np.uint8)
    cv2.rectangle(header_img, (-1,0), (disp_w+1, header_height), (0,0,0), 5)
    draw_centered_text(header_img, header_text, text_scale, text_color)
    
    return np.row_stack((header_img, display_image))

def build_tiled_display(frames_list, sort_idx_list = None, num_rows = None, num_cols = None):
    
    # Figure out how many row/col tiles we need for display
    num_frames = len(frames_list)
    if (num_rows is None) or (num_cols is None):
        edge_1 = round(np.sqrt(num_frames))
        edge_2 = round(num_frames / edge_1)
        is_complete = ((edge_1 * edge_2) == num_frames)
        if not is_complete:
            edge_1 = 2
            edge_2 = int(np.ceil(num_frames / edge_1))
    
        # Prefer more columns to rows, to give wide aspect ratio
        num_rows, num_cols = sorted([edge_1, edge_2])
    
    # If a sorting order isn't provided, just display in list order
    if sort_idx_list is None:
        sort_idx_list = range(num_frames)
    
    row_list = []
    col_list = []
    for idx in sort_idx_list:
        each_frame = frames_list[idx]
        col_list.append(each_frame)
        if len(col_list) >= num_cols:
            row_list.append(np.column_stack(col_list))
            col_list = []
    
    # If we don't have enough entries on the last row, pad with blank frames
    need_check_padding = len(col_list) > 0
    while need_check_padding:
        last_row_is_short = (len(col_list) < num_cols)
        if not last_row_is_short:
            row_list.append(np.column_stack(col_list))
            col_list = []
            break
        blank_frame = np.zeros_like(frames_list[0], dtype=np.uint8)
        col_list.append(blank_frame)
    
    return np.row_stack(row_list)
