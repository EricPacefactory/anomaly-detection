#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:57:47 2023

@author: eo
"""


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp
import cv2
import numpy as np

from time import perf_counter

from collections import defaultdict

from lib.history import load_history_file, save_history_file
from lib.loading import load_one_metadata, load_trail_data, load_snapshot_image, get_final_snap, get_report_folder_paths
from lib.colormap import make_inferno_colormap, apply_cmap
from lib.drawing import MouseClickCB
from lib.drawing import scale_to_max_side_length, draw_centered_text, add_header_image, build_tiled_display


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def print_time_taken_ms(t1, t2):
    print("-> Took {} ms".format(round(1000*(t2 - t1))))
    return

def prompt_for_folder_select(parent_folder_path):
    
    # Get list of available folders to pick from
    child_paths = [osp.join(parent_folder_path, name) for name in os.listdir(parent_folder_path)]
    folder_names_list = sorted([osp.basename(path) for path in child_paths if osp.isdir(path)])
    folder_paths_list = [osp.join(parent_folder_path, name) for name in folder_names_list]
    
    # If there is only one option, choose it automatically without prompting
    only_one_option = len(folder_names_list) == 1
    if only_one_option:
        return folder_paths_list[0]
    
    # Ask for location selection
    print("", "Options:", sep = "\n")
    for idx, name in enumerate(folder_names_list):
        print("  {} - {}".format(idx + 1, name))
    print("", flush = True)
    user_idx = input("Select by index: ")
    
    try:
        user_idx = int(user_idx.strip()) - 1
    except ValueError:
        print("",
              "Couldn't convert index input into a number!",
              "Got:", 
              user_idx, sep = "\n")
        raise SystemExit()

    return folder_paths_list[user_idx]

def nullfunc(x):
    ''' For trackbar callbacks '''
    return None

# ---------------------------------------------------------------------------------------------------------------------
#%% Config

# Set minimum trail count. Classes with fewer trails are discarded from further processing
MIN_NUM_TRAILS_FOR_HEATMAP = 50

# Heatmap controls. Trail thickness has a very significant impact on results!
HEATMAP_SIDE_LENGTH_PX = 360
HEATMAP_TRAIL_THICKNESS = 2
HEATMAP_BLUR_SIZE = 7

# The set of keycodes that will close a window (for display windows only)
WINDOW_CLOSE_KEYCODES = set([27, 113])  # esc or q


# ---------------------------------------------------------------------------------------------------------------------
#%% Get pathing to data

# Clear any open/bugged windows
cv2.destroyAllWindows()

# Figure out where the locations folder holding all camera data is
history_dict = load_history_file(__file__)
all_loc_folder_path = history_dict.get("locations_folder_path", None)
missing_all_loc_folder_path = (all_loc_folder_path is None)
if missing_all_loc_folder_path:
    print("", flush = True)
    all_loc_folder_path = input("Enter path to scv2 locations folder: ")

# Clean up path & display for user
all_loc_folder_path = osp.expanduser(all_loc_folder_path)
print("", "Using locations folder:", "@ {}".format(all_loc_folder_path), sep = "\n")

# Get user data selections
location_select_path = prompt_for_folder_select(all_loc_folder_path)
camera_select_path = prompt_for_folder_select(location_select_path)
camera_select = osp.basename(camera_select_path)

# Get important report pathing
snap_img_folder_path, bg_img_folder_path, obj_md_folder_path = \
    get_report_folder_paths(location_select_path, camera_select)

# Grab list of all snaps (for display, later) and backgrounds, if this works, we probably have good pathing
all_snap_img_files = os.listdir(snap_img_folder_path)
all_bg_img_files = os.listdir(bg_img_folder_path)
good_data = (len(all_snap_img_files) > 0) and (len(all_bg_img_files) > 0)
if good_data:
    history_dict["locations_folder_path"] = all_loc_folder_path
    save_history_file(__file__, history_dict)

# Load example snap/bg for display purposes
ref_bg_path = osp.join(bg_img_folder_path, os.listdir(bg_img_folder_path)[0])
ref_snap_path = osp.join(snap_img_folder_path, all_snap_img_files[0])


# ---------------------------------------------------------------------------------------------------------------------
#%% Load data

# Load object trail data
t1 = perf_counter()
print("", "Loading object trail data...", sep = "\n", flush=True)
all_trails_by_class_and_id, example_obj_data_dict = load_trail_data(obj_md_folder_path)
t2 = perf_counter()

# Get trail/class counts
num_trails_per_class = {class_label: len(objid_dict) for class_label, objid_dict in all_trails_by_class_and_id.items()}
num_trails = sum(num_trails_per_class.values())
print("Found {} total objects, {} classes".format(num_trails, len(num_trails_per_class)))
print_time_taken_ms(t1, t2)

# Load image of scene for plotting trails in context
ref_bg_img = cv2.imread(ref_bg_path)
ref_snap_img = cv2.imread(ref_snap_path)


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up display image & sizing

# Create base display image (from background)
base_disp_img = scale_to_max_side_length(ref_bg_img, 800)
disp_h, disp_w = base_disp_img.shape[0:2]
disp_xy_scaling = np.float32((disp_w - 1, disp_h - 1))
disp_1ch_shape = (disp_h, disp_w)
cmap = make_inferno_colormap()


# ---------------------------------------------------------------------------------------------------------------------
#%% Remove rarely-seen classes

# Loop to keep only classes which have enough trails
trails_by_class_and_id = {}
class_labels_removed = []
for each_class_label, each_trail_count in num_trails_per_class.items():
    if each_trail_count < MIN_NUM_TRAILS_FOR_HEATMAP:
        class_labels_removed.append(each_class_label)
        continue
    trails_by_class_and_id[each_class_label] = all_trails_by_class_and_id[each_class_label]

# Provide feedback about class removal, if needed
did_remove_classes = len(class_labels_removed) > 0
if did_remove_classes:
    print("", "Removed low-count (< {}) classes:".format(MIN_NUM_TRAILS_FOR_HEATMAP), sep = "\n", flush = True)
    for each_class_label in class_labels_removed:
        print("  {} ({} trails)".format(each_class_label, num_trails_per_class[each_class_label]))
    pass


# ---------------------------------------------------------------------------------------------------------------------
#%% Draw region masks
    
# Add mouse interaction callback to display window
mask_winname = "Region Mask"
click_cb = MouseClickCB([0,0], [disp_w, disp_h])
cv2.namedWindow(mask_winname)
cv2.moveWindow(mask_winname, 50, 50)
cv2.setMouseCallback(mask_winname, click_cb)

# Set up trackbars for brush sizing and paint vs erasing
brushsize_trackbar_name, default_brush_size = "Brush Size", 30
paint_trackbar_name, default_paint_mode = "Erase/Paint", 1
cv2.createTrackbar(brushsize_trackbar_name, mask_winname, default_brush_size, 100, nullfunc)
cv2.createTrackbar(paint_trackbar_name, mask_winname, default_paint_mode, 1, nullfunc)

# Make mask data, if it doesn't already exist 
# -> this try/except is an awkward hack to avoid reseting an existing mask in an interactive session
# -> in normal (non-interactive) use, the 'except' block should always be run
try:
    user_mask.shape # Raises NameError if the variable hasn't been declared yet
    # ^^^ this will raise an error the first time the script is run, but not on repeat interactive runs
    
except NameError:
    user_mask = np.zeros(disp_1ch_shape, dtype = np.uint8)
    cv2.rectangle(user_mask, (int(disp_w*0.3), int(disp_h*0.3)), (int(disp_w*0.7), int(disp_h*0.7)), 255, -1)
    draw_centered_text(user_mask, "paint region mask", 0.5, 255, use_bg=True)

while True:
    
    # Decide whether we're painting or erasing
    is_painting_white = cv2.getTrackbarPos(paint_trackbar_name, mask_winname)
    paint_color = 255 if is_painting_white else 0
    indicator_color = (0, 255, 0) if is_painting_white else (0, 0, 255)
    
    # Get the brush size
    draw_radius = cv2.getTrackbarPos(brushsize_trackbar_name, mask_winname)
    draw_radius = max(1, draw_radius)
    
    # Update masking on mouse-click
    mouse_is_down, mouse_xy = click_cb.is_down()
    if mouse_is_down:
        draw_line = click_cb.get_line_points()
        cv2.polylines(user_mask, [draw_line], False, paint_color, int(draw_radius*2), cv2.LINE_AA)
    
    # Combine the drawn mask with the image & a side-by-side copy for display
    mask_3ch = cv2.cvtColor(user_mask, cv2.COLOR_GRAY2BGR)
    disp_img = cv2.addWeighted(base_disp_img, 0.75, mask_3ch, 0.5, 0)
    cv2.circle(disp_img, mouse_xy, draw_radius, indicator_color, 1, cv2.LINE_AA)
    cv2.imshow(mask_winname, np.hstack((disp_img, mask_3ch)))
    keypress = cv2.waitKey(5) & 0xFF
    if keypress in WINDOW_CLOSE_KEYCODES:
        break
    pass

# Clean up
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------------
#%% Remove trails outside of the region mask

t1 = perf_counter()
print("", "Removing trails outside of region mask...", sep = "\n", flush=True)
inv_mask = np.bitwise_not(user_mask)
masked_trails_by_class_and_id = {}
for each_class_label, trails_by_id_dict in trails_by_class_and_id.items():
    
    # Check masking for each object of each class, using heavy-handed image-based check
    masked_trails_by_class_and_id[each_class_label] = {}
    for each_obj_id, each_obj_dict in trails_by_id_dict.items():
        single_trail_as_mask = np.zeros(disp_1ch_shape, dtype=np.uint8)
        trail_xy_px = np.int32(np.round(each_obj_dict["xy_center"] * disp_xy_scaling))
        cv2.polylines(single_trail_as_mask, [trail_xy_px], False, 255, 1, cv2.LINE_AA)
        
        # Find where trail overlaps mask vs. overlaps area outside of mask
        masked_trail = np.bitwise_and(single_trail_as_mask, user_mask)
        outmasked_trail = np.bitwise_and(single_trail_as_mask, inv_mask)
        
        # Keep objects whose trail overlaps the masked region more often than not
        in_sum = np.sum(masked_trail)
        out_sum = np.sum(outmasked_trail)
        spends_more_time_in_masked_region = (in_sum > out_sum)
        if spends_more_time_in_masked_region:
            masked_trails_by_class_and_id[each_class_label][each_obj_id] = each_obj_dict
        pass
    pass

t2 = perf_counter()
print_time_taken_ms(t1, t2)

print("", "Trail counts before & after:", sep = "\n")
num_trails_per_class = {class_label: len(objid_dict) for class_label, objid_dict in masked_trails_by_class_and_id.items()}
num_trails = sum(num_trails_per_class.values())
for each_class_label, trails_by_id_dict in trails_by_class_and_id.items():
    print("  {}: {} -> {}".format(each_class_label, len(trails_by_id_dict), num_trails_per_class[each_class_label]))


# ---------------------------------------------------------------------------------------------------------------------
#%% Build base heatmaps

# Set up base display image & sizing
heatmap_base_img = scale_to_max_side_length(ref_bg_img, HEATMAP_SIDE_LENGTH_PX)
heat_h, heat_w = heatmap_base_img.shape[0:2]
heatmap_xy_scaling = np.float32((heat_w - 1, heat_h - 1))
heatmap_1ch_shape = (heat_h, heat_w)
heat_dtype = np.float32
blur_tuple = (HEATMAP_BLUR_SIZE, HEATMAP_BLUR_SIZE)

# Initialize blank heatmap for each class 
# -> uses float32 instead of uint8, this avoids issues with 255+ heat values
heatmaps_per_class = {}
for each_class_label in masked_trails_by_class_and_id.keys():
    heatmaps_per_class[each_class_label] = np.zeros(heatmap_1ch_shape, dtype=heat_dtype)

# Loop which builds the base heatmaps for each class
t1 = perf_counter()
print("", "Building heatmaps using line thickness of {}...".format(HEATMAP_TRAIL_THICKNESS), sep = "\n", flush=True)
for each_class_label, trails_by_id_dict in masked_trails_by_class_and_id.items():
    for each_obj_id, each_obj_dict in trails_by_id_dict.items():
        
        # Get trail data in pixel coords, so we can draw it
        trail_xy_px = np.int32(np.round(each_obj_dict["xy_center"] * heatmap_xy_scaling))
        
        # Create a 'blank' image, with just the single object trail, blurred
        # -> each trail is drawn independently to get separate blurring
        # -> blurring is done with simple box blur, this is fast but adds artifacts. Probably fine with lots of trails
        # -> each trail is drawn with weight (or 'color') of 1, so heatmap 'counts' how many objects in different areas
        single_trail_heat = np.zeros(heatmap_1ch_shape, dtype=heat_dtype)
        cv2.polylines(single_trail_heat, [trail_xy_px], False, 1, HEATMAP_TRAIL_THICKNESS, cv2.LINE_AA)
        cv2.blur(single_trail_heat, blur_tuple, borderType = cv2.BORDER_CONSTANT)
    
        # Build up each heatmap by summing the contribution for each individual trail
        heatmaps_per_class[each_class_label] = heatmaps_per_class[each_class_label] + single_trail_heat

t2 = perf_counter()
print_time_taken_ms(t1, t2)


# ---------------------------------------------------------------------------------------------------------------------
#%% Create scaled heatmaps

# Set up windows with trackbar controls for every class label
ctrls_dict = {}
winnames_dict = {}
min_trackbar_name = "Min Trails"
max_trackbar_name = "Max Trails"
for idx, each_class_label in enumerate(heatmaps_per_class.keys()):
    
    new_win_name = "Heatmap - {}".format(each_class_label)
    winnames_dict[each_class_label] = new_win_name
    cv2.namedWindow(new_win_name)
    cv2.moveWindow(new_win_name, x = 50 + 25*idx, y = 50 + 25*idx)
    cv2.createTrackbar(min_trackbar_name, new_win_name, 3, 1000, nullfunc)
    cv2.createTrackbar(max_trackbar_name, new_win_name, 50, 1000, nullfunc)

# Initialize storage for holding the min/max trail counts for heatmaps of each class
# -> this try/except is an awkward hack to avoid reseting the thresholds in an interactive session
# -> in normal (non-interactive) use, the 'except' block should always be run
try:
    heatmap_trail_count_thresholds.keys() # Raises NameError if the variable hasn't been declared yet
    for each_class_label, each_winname in winnames_dict.items():
        min_thresh, max_thresh = heatmap_trail_count_thresholds[each_class_label]
        cv2.setTrackbarPos(min_trackbar_name, each_winname, min_thresh)
        cv2.setTrackbarPos(max_trackbar_name, each_winname, max_thresh)
    
except NameError:
    heatmap_trail_count_thresholds = {}


# Interactive loop, where user sets the min/max thresholding values for the each heatmap (one for each class)
log_heatmaps_per_class = {}
while True:
    
    for each_class_label, each_heatmap in heatmaps_per_class.items():
        
        # Read trackbar controls to update the min/max thresholding values for each class
        winname = winnames_dict[each_class_label]
        min_thresh = cv2.getTrackbarPos(min_trackbar_name, winname)
        max_thresh = cv2.getTrackbarPos(max_trackbar_name, winname)
        min_thresh, max_thresh = sorted([min_thresh, max_thresh])
        max_thresh = max(1 + min_thresh, max_thresh)
        heatmap_trail_count_thresholds[each_class_label] = (min_thresh, max_thresh)
        
        # Re-compute log-scaled + normalized heatmap (using newest thresholds)
        new_clipped_heatmap = np.clip(each_heatmap, min_thresh, max_thresh)
        new_log_heatmap = np.log(new_clipped_heatmap - min_thresh + 1)
        log_heatmap = new_log_heatmap / max(1, np.max(new_log_heatmap))
        
        # Convert heatmap to a colormapped image for display
        color_heat = apply_cmap(log_heatmap, cmap)
        scaled_colored_heat = cv2.resize(color_heat, dsize = (disp_w, disp_h))
        cv2.imshow(winname, scaled_colored_heat)
        
        # Store results
        log_heatmaps_per_class[each_class_label] = log_heatmap
    
    # Leave on keypress
    keypress = cv2.waitKey(50) & 0xFF
    if keypress in WINDOW_CLOSE_KEYCODES:
        break

# Clean up
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------------
#%% Calculate anomaly scores

t1 = perf_counter()
print("", "Calculating anomaly scores...", sep = "\n", flush = True)
molly_scores_per_class = defaultdict(dict)
for each_class_label, trails_by_id_dict in masked_trails_by_class_and_id.items():
    for each_obj_id, each_obj_dict in trails_by_id_dict.items():
        
        # Get trail data in pixel coords, so we can use it to index into the heatmap
        trail_xy_px = np.int32(np.round(each_obj_dict["xy_center"] * heatmap_xy_scaling))
        trail_xy_px = np.clip(trail_xy_px, [0,0], [heat_w - 1, heat_h - 1])
        
        # Compute score by summing up 'heat' at each pixel that the trail visits
        # -> This approach ignores the travel between consecutive samples. Simpler but maybe problematic in some cases?
        total_score = 0
        for each_xpx, each_ypx in trail_xy_px:
            total_score += log_heatmaps_per_class[each_class_label][each_ypx, each_xpx]
        
        # Normalize score by sample count, so long-lasting trails don't inherently score higher
        num_trail_samples = len(trail_xy_px)
        molly_scores_per_class[each_class_label][each_obj_id] = total_score / num_trail_samples

t2 = perf_counter()
print_time_taken_ms(t1, t2)


# ---------------------------------------------------------------------------------------------------------------------
#%% Visualize low scores

# Get object IDs in order of lowest (most anomalous) to highest (most 'normal') scores
ordered_objids_by_score_per_class = {}
for each_class_label, scores_by_id_dict in molly_scores_per_class.items():
    ids_array = np.uint64([*scores_by_id_dict.keys()])
    scores_array = np.float32([*scores_by_id_dict.values()])
    ordered_objids_by_score_per_class[each_class_label] = ids_array[np.argsort(scores_array)]

# Get 'top' anomalies for plotting
num_anomalies_to_plot = 9
top_anomaly_ids_per_class = {}
for each_class_label, ordered_objids_by_anomaly_score in ordered_objids_by_score_per_class.items():
    top_anomaly_ids_per_class[each_class_label] = ordered_objids_by_anomaly_score[:num_anomalies_to_plot]

# Loop to create image with heatmap + example anomaly trails, for each class (results will be combined into one image)
trail_counts = []
final_molly_frames = []
for each_class_label, top_anomaly_ids_array in top_anomaly_ids_per_class.items():
    
    # Draw colored heatmap
    each_disp_img = base_disp_img.copy()
    color_heat = apply_cmap(log_heatmaps_per_class[each_class_label], cmap)
    scaled_colored_heat = cv2.resize(color_heat, dsize = (disp_w, disp_h))
    overlayed_heat = cv2.addWeighted(each_disp_img, 0.5, scaled_colored_heat, 0.9, 0.0)
    
    # Draw anomaly trails
    trail_color = (60,80,255)
    for each_obj_id in top_anomaly_ids_array:
        
        # Draw object trails with background for better contrast
        trail_xy_norm = masked_trails_by_class_and_id[each_class_label][each_obj_id]["xy_center"]
        trail_xy_px = np.int32(np.round(trail_xy_norm * disp_xy_scaling))
        overlayed_heat = cv2.polylines(overlayed_heat, [trail_xy_px], False, (0,0,0), 3, cv2.LINE_AA)
        overlayed_heat = cv2.polylines(overlayed_heat, [trail_xy_px], False, trail_color, 1, cv2.LINE_AA)
    
        # Draw score text
        text_xy_px = np.int32(np.mean(trail_xy_px, axis=0)).tolist()
        score_text = "{:.0f}%".format(100 * (1 - molly_scores_per_class[each_class_label][each_obj_id]))
        cv2.putText(overlayed_heat, score_text, text_xy_px, 0, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlayed_heat, score_text, text_xy_px, 0, 0.6, trail_color, 1, cv2.LINE_AA)
    
    # Add headers describing data for each class
    header_text = "{} ({} trails)".format(each_class_label, num_trails_per_class[each_class_label])
    overlayed_heat = add_header_image(overlayed_heat, header_text)
    
    # Store images for tiling later (and order byu trail counts, so store that too)
    final_molly_frames.append(overlayed_heat)
    trail_counts.append(num_trails_per_class[each_class_label])


# Make our number of display frames into a 'nice' number for tiling (by adding base image if needed)
num_molly_frames = len(final_molly_frames)
is_odd_frame_count = (num_molly_frames % 2 != 0)
is_square = (round(np.sqrt(num_molly_frames))**2 == num_molly_frames)
is_one_class = len(final_molly_frames) == 1
need_add_ref_image = (is_odd_frame_count and not is_square) or (is_one_class)
if need_add_ref_image:
    trail_counts.insert(0, 1E9)
    final_molly_frames.insert(0, add_header_image(base_disp_img, "Reference Scene"))
    num_molly_frames = len(final_molly_frames)

# Bundle all anomaly heatmap/trail frames together
disp_sorting_idx_list = np.flip(np.argsort(trail_counts))
final_disp_img = build_tiled_display(final_molly_frames, disp_sorting_idx_list) 

# Show all class heatmaps together
cv2.imshow("Anomaly Maps", final_disp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------------
#%% Show example anomalies

# Get pixel scaling for drawing trails
snap_h, snap_w = ref_snap_img.shape[0:2]
snap_xy_scaling = np.float32((snap_w - 1, snap_h - 1))

# Grab all snapshot times, so we can search for snaps closest to anomaly timing
all_snaps_ems_array = np.sort(np.uint64([int(osp.splitext(file)[0]) for file in all_snap_img_files]))

# Loop which draws a tiled set of anomaly examples for each class label
trail_color = (60,80,255)
example_images_per_class = {}
for each_class_label, top_anomaly_ids_array in top_anomaly_ids_per_class.items():
    
    # Loop which generates each of the tile images (i.e. single anomaly examples) that go into final combined image
    frame_timings_list = []
    example_molly_frames = []
    for each_obj_id in top_anomaly_ids_array:
        
        # Grab relevant object data for drawing
        obj_data_dict = load_one_metadata(obj_md_folder_path, each_obj_id)
        first_ems = obj_data_dict["first_epoch_ms"]
        final_ems = obj_data_dict["final_epoch_ms"]
        final_dt = obj_data_dict["final_datetime_isoformat"]
        final_timestamp = final_dt[11:19]
        trail_xy_norm = obj_data_dict["tracking"]["xy_center"]
        final_hull = obj_data_dict["tracking"]["hull"][-1]
        
        # Get list of snapshots to load for display
        close_snap_ems = get_final_snap(all_snaps_ems_array, final_ems - 500)
        snap_img = load_snapshot_image(snap_img_folder_path, close_snap_ems)
        
        # Draw trail & final hull, with dark background for contrast
        hull_px = np.int32(np.round(final_hull * snap_xy_scaling))
        trail_xy_px = np.int32(np.round(trail_xy_norm * snap_xy_scaling))
        cv2.polylines(snap_img, [hull_px], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(snap_img, [trail_xy_px], False, (0,0,0), 3, cv2.LINE_AA)
        cv2.polylines(snap_img, [hull_px], True, trail_color, 1, cv2.LINE_AA)
        cv2.polylines(snap_img, [trail_xy_px], False, trail_color, 1, cv2.LINE_AA)
        
        # Add a header bar with class label, time stamp and score
        score_text = "{:.0f}%".format(100 * (1 - molly_scores_per_class[each_class_label][each_obj_id]))
        header_text = "{} @ {} ({})".format(each_class_label, final_timestamp, score_text)
        snap_img = add_header_image(snap_img, header_text)
        
        # Store image frames so we can tile them later (and order by frame timing, so store that too)
        frame_timings_list.append(first_ems)
        example_molly_frames.append(snap_img)
    
    # Bundle each example into a single image (per class)
    tile_sorting_idx_list = np.argsort(frame_timings_list)
    combined_examples_image = build_tiled_display(example_molly_frames, tile_sorting_idx_list)

    # Display combined examples
    example_images_per_class[each_class_label] = combined_examples_image
    cv2.imshow("Anomaly Object - {}".format(each_class_label), combined_examples_image)
    cv2.waitKey(125)

# Clean up
cv2.waitKey(0)
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------------
#%% Demo heatmap thresholding

# Set up windows with trackbar controls for every class label
ctrls_dict = {}
winnames_dict = {}
thresh_trackbar_name = "Threshold"
thresh_trackbar_max = 1000
thresh_trackbar_default = int(thresh_trackbar_max * 0.1)
for idx, each_class_label in enumerate(heatmaps_per_class.keys()):
    new_win_name = "Thresholded Heatmap - {}".format(each_class_label)
    winnames_dict[each_class_label] = new_win_name
    cv2.namedWindow(new_win_name)
    cv2.moveWindow(new_win_name, x = 50 + 25*idx, y = 50 + 25*idx)
    cv2.createTrackbar(thresh_trackbar_name, new_win_name, thresh_trackbar_default, thresh_trackbar_max, nullfunc)

# Interactive loop, where user sets the heatmap thresholding values for each class
while True:
    
    # Draw separate window/controls for each class
    for each_class_label, each_log_heatmap in log_heatmaps_per_class.items():
        
        # Read trackbar controls to update the min/max thresholding values for each class
        winname = winnames_dict[each_class_label]
        thresh_val = cv2.getTrackbarPos(thresh_trackbar_name, winname)
        thresh_norm = thresh_val / thresh_trackbar_max
        
        # Create thresholded copy of heatmap, to generate regions
        blurred_heatmap_map = cv2.blur(each_log_heatmap, (7,7))
        _, thresholded_map_float = cv2.threshold(blurred_heatmap_map, thresh_norm, 1.0, cv2.THRESH_BINARY)
        thresholded_map_uint8 = np.uint8(np.round(255 * thresholded_map_float))
        
        # Convert map to display size & find contours
        thresholded_map_uint8 = cv2.resize(thresholded_map_uint8, dsize = (disp_w, disp_h))
        thresh_contours, _ = cv2.findContours(thresholded_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours onto original image for comparison
        disp_img = base_disp_img.copy()
        cv2.polylines(disp_img, thresh_contours, True, (0,255,255), 2, cv2.LINE_AA)
        threshold_map_bgr = cv2.cvtColor(thresholded_map_uint8, cv2.COLOR_GRAY2BGR)        
        cv2.imshow(winname, np.hstack((disp_img, threshold_map_bgr)))
    
    # Leave on keypress
    keypress = cv2.waitKey(50) & 0xFF
    if keypress in WINDOW_CLOSE_KEYCODES:
        break

# Clean up
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------------------------------
#%% Saving

save_image_results = False

if save_image_results:
    print("", flush = True)
    save_dir = os.path.join(osp.dirname(__file__), "anomaly_outputs", camera_select)
    os.makedirs(save_dir, exist_ok = True)
    save_path = osp.join(save_dir, "{}--regionmask.jpg".format(camera_select))
    final_mask_img = np.hstack((disp_img, mask_3ch))
    cv2.imwrite(save_path, final_mask_img)
    print("", "Saved region mask image:", "@ {}".format(save_path), sep = "\n")


# Save heatmaps
if save_image_results:
    print("", flush = True)
    save_dir = os.path.join(osp.dirname(__file__), "anomaly_outputs", camera_select)
    os.makedirs(save_dir, exist_ok = True)
    save_path = osp.join(save_dir, "{}--heatmap.jpg".format(camera_select))
    cv2.imwrite(save_path, final_disp_img)
    print("", "Saved anomaly map image:", "@ {}".format(save_path), sep = "\n")


# Save trail examples
if save_image_results:
    print("", flush = True)
    save_dir = os.path.join(osp.dirname(__file__), "anomaly_outputs", camera_select, "example_frames")
    os.makedirs(save_dir, exist_ok = True)
    for each_class_label, each_frame in example_images_per_class.items():
        save_path = osp.join(save_dir, "{}--{}.jpg".format(camera_select, each_class_label))
        cv2.imwrite(save_path, each_frame)
    print("", "Saved anomaly examples:", "@ {}".format(save_dir), sep = "\n")
