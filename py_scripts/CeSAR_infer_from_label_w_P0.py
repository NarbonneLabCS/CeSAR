from audioop import cross
from cgitb import small
from datetime import datetime
from dis import dis
from email.mime import multipart
from enum import unique
from ftplib import parse150
import glob
from importlib.resources import path
import math
from math import pow
import os
import pathlib
from pprint import pprint
import sys
from glob import glob
import time
import csv
import re
from tracemalloc import start
from PIL import Image, ImageFile, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2 as cv
import imagesize
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from colour import Color
from skimage.morphology import skeletonize
from fil_finder import FilFinder2D
import astropy.units as u
import scipy.misc
import imageio


np.set_printoptions(threshold=np.inf)

try:
    fn = sys.argv[1]
except Exception as e:
    #print(str(e))
    print('NO PATH ENTERED: Please write a valid path as first argument.')
    print('Exiting now.')
    sys.exit()


try:
    prcsd_img_folder = sys.argv[2]    
except Exception as e:
    #print(str(e))
    print('NO PATH ENTERED: Please write a valid path as second argument.')
    print('Exiting now.')
    sys.exit()

def init_dirs(training_dir, rights):
    # training_split DIR
    if not os.path.isdir(training_dir):
        print('Initializing [training_split] folder...')
        try:
            os.mkdir(training_dir, rights)
        except OSError:
            print ("Creation of the directory %s failed" % training_dir)
        else:
            print ("Successfully created the directory %s" % training_dir)
        
    else:
        print("[Originals] folder already created.")

def dev_stage_from_size(size):
    if 1.0 <= size < 50.0:
        return 'L1'
    elif 50.0 <= size < 83.33:
        return 'L2'
    elif 83.33 <= size < 125.0:
        return 'L3'
    elif 125.0 <= size < 166.0:
        return 'L4'
    else:
        return 'Ad'

def angle_from_three_points(pt_1, center, pt_3):
    dist_ctr_to_med = math.hypot(actual_to_new_vector[0], actual_to_new_vector[1])
    dist_ctr_to_lmp = ori_w / 2  # By definition
    dist_med_to_lmp = math.hypot(0 - actual_to_new_vector[0], ori_h / 2 - actual_to_new_vector[1])

    theta = math.acos((pow(dist_ctr_to_med, 2) + pow(dist_ctr_to_lmp, 2) - pow(dist_med_to_lmp, 2))/(2 * dist_ctr_to_med * dist_ctr_to_lmp))

    return theta

def get_xy_from_2D_axes_rotation(ori_x, ori_y, theta):
    new_x = ori_x * math.cos(theta) + ori_y * math.sin(theta)
    new_y = -ori_x * math.sin(theta) + ori_y * math.cos(theta)

    return (new_x, new_y)

if os.path.exists(fn) and os.path.exists(prcsd_img_folder):
    csv_header = ['Experiment_start','Well', 'Time', 'Egg', 'Worm']

    # Gets CUR_DIR
    directory_in_str = pathlib.Path(__file__).parent.absolute()
    directory = os.fsencode(directory_in_str)
    modified_img_dir_path = os.path.join(os.fsdecode(directory), "CeSAR_graphs/")
    
    # define the access rights
    access_rights = 0o755

    init_dirs(modified_img_dir_path, access_rights)
    
    
    # Extract data from labels
    # Test on https://www.digitalocean.com/
    hourly_labels_files_list = glob(fn + 'exp*/labels/*s_[2-3]*px.txt')
    P0_labels_files_list = glob(fn + 'exp*/labels/*s_f*_[2-3]*px.txt')

    pprint(fn)
    pprint(hourly_labels_files_list)
    pprint(P0_labels_files_list)

    init_dirs(modified_img_dir_path + '/graphs', access_rights)
    init_dirs(modified_img_dir_path + '/graphs/pop_heatmap', access_rights)
    init_dirs(modified_img_dir_path + '/graphs/pop_size', access_rights)
    init_dirs(modified_img_dir_path + '/graphs/BS', access_rights)
    init_dirs(modified_img_dir_path + '/graphs/correl', access_rights)

    pattern = re.escape(fn) + '.*exp\d*\/labels\/PRCSD_(.*)_(\d{2}-\d{2}-\d{4}_\d{2}h\d{2}m\d{2}s)_(.*)px.'  #'CeSAR_infer/exp2/labels/PRCSD_B10_17-06-2022_06h03m49s_2876px.txt']
    #                                                                                                                                   PRCSD_B9_20-07-2022_19h38m06s_f18_2816px.txt
    frame_img_pattern = r'f(\d+)_(\d+)'

    global_labels_data_list = []

    # labels_files for hourly data (brood size)
    for labels_file_path in hourly_labels_files_list:
        print(labels_file_path)

        re_match = re.search(pattern, labels_file_path)
        print(re_match.groups())
        wellname = re_match.group(1)
        acquisiton_time = re_match.group(2)
        ori_w = ori_h = int(re_match.group(3))
        datetime_object = datetime.strptime(acquisiton_time, '%d-%m-%Y_%Hh%Mm%Ss')
        formatted_datetime_object = datetime_object.strftime("%d-%m-%Y_%Hh%Mm%Ss") # 31-07-2013_03h12m12s
        year = int(acquisiton_time[6:10])
        month = int(acquisiton_time[3:5])
        day = int(acquisiton_time[0:2])
        hours = int(acquisiton_time[11:13])
        minutes = int(acquisiton_time[14:16])
        seconds = int(acquisiton_time[17:19])

        datetime_object = datetime(year, month, day, hours, minutes, seconds)
        
        temp_det_objects = []
        with open(labels_file_path) as file:
            for line in file:
                read_line = line.rstrip().split()
                temp_det_objects.append(read_line)

        for object in temp_det_objects:
            
            cls = 'Egg' if object[0] == '0' else 'Worm'
            cur_yolo_bbox_xywh = (object[1], object[2], object[3], object[4])
            voc_bbox_w = float(cur_yolo_bbox_xywh[2]) * ori_w
            voc_bbox_h = float(cur_yolo_bbox_xywh[3]) * ori_h
            center_x = float(cur_yolo_bbox_xywh[0]) * ori_w
            center_y = float(cur_yolo_bbox_xywh[1]) * ori_h
            voc_x1 = center_x - (voc_bbox_w / 2)
            voc_y1 = center_y - (voc_bbox_h / 2)
            voc_x2 = center_x + (voc_bbox_w / 2)
            voc_y2 = center_y + (voc_bbox_h / 2)
            bbox_xyxy = [voc_x1,voc_y1,voc_x2,voc_y2]
            pos = (round(center_x,2), round(center_y,2))
            size = round(math.hypot(voc_x2-voc_x1, voc_y2-voc_y1),2)
            if cls == 'Egg':
                dev_stage = 'Embryo'
            else:
                dev_stage = dev_stage_from_size(size)
            conf = round(float(object[5]),2)


            #det_object_list.append([cls,pos[0],pos[1],size,conf,dev_stage])
        
            global_labels_data_list.append([wellname, datetime_object, ori_w, cls, pos[0], pos[1], size, conf, dev_stage])

    # Populate the wellnames
    wellnames = []
    start_times = []
    for labels_data in global_labels_data_list:
        if labels_data[0] not in wellnames:
            wellnames.append(labels_data[0])
            start_times.append('')

    # Order the wellnames
    wellnames = sorted(wellnames)

    # Populate the acquisition times
    wellname_start_time = dict(zip(wellnames, start_times))
    for wellname in wellnames:
        for labels_data in global_labels_data_list:
            if labels_data[0] == wellname:
                if not wellname_start_time[wellname]:
                    wellname_start_time[wellname] = labels_data[1]
                if  labels_data[1] < wellname_start_time[wellname]:
                    wellname_start_time[wellname] = labels_data[1]

    for labels_data in global_labels_data_list:
        #pprint(labels_data[1])
        #pprint(wellname_start_time[labels_data[0]])
        # ['B10', datetime.datetime(2022, 6, 16, 11, 8, 28), 0]
        file_time = labels_data[1]
        ref_time =  wellname_start_time[labels_data[0]]
        #print(ref_time)
        rel_time = file_time - ref_time
        diff_hours = round((rel_time.total_seconds() / 3600)) #round(abs(( -]).seconds/3600),1)
        #pprint(diff_hours)
        #pprint(f'Difference between start date and acquisition date for {labels_data} is {diff_hours}')
        #pprint(wellname_start_time)
        labels_data.append(diff_hours)
        labels_data.append(ref_time)
        formatted_ref_time = ref_time.strftime("%d-%m-%Y_%Hh%Mm%Ss")
        year = formatted_ref_time[6:10]
        month = formatted_ref_time[3:5]
        day = formatted_ref_time[0:2]
        hours = formatted_ref_time[11:13]
        minutes = formatted_ref_time[14:16]
        seconds = formatted_ref_time[17:19]

        well_key = f'{labels_data[0]}{day}{month}{year}{hours}{minutes}{seconds}'
        labels_data.append(well_key)

    P0_labels_data_list = []
    # labels_files for P0
    for P0_labels_file_path in P0_labels_files_list:
        print(P0_labels_file_path)

        re_match = re.search(pattern, P0_labels_file_path)
        frame_img_re_match = re.search(frame_img_pattern, re_match.group(3))
        print(re_match.groups())
        wellname = re_match.group(1)
        acquisiton_time = re_match.group(2)
        frame_number = int(frame_img_re_match.group(1))
        ori_w = ori_h = int(frame_img_re_match.group(2))
        datetime_object = datetime.strptime(acquisiton_time, '%d-%m-%Y_%Hh%Mm%Ss')
        formatted_datetime_object = datetime_object.strftime("%d-%m-%Y_%Hh%Mm%Ss") # 31-07-2013_03h12m12s
        year = int(acquisiton_time[6:10])
        month = int(acquisiton_time[3:5])
        day = int(acquisiton_time[0:2])
        hours = int(acquisiton_time[11:13])
        minutes = int(acquisiton_time[14:16])
        seconds = int(acquisiton_time[17:19])

        datetime_object = datetime(year, month, day, hours, minutes, seconds)
        
        temp_det_objects = []
        with open(P0_labels_file_path) as file:
            for line in file:
                read_line = line.rstrip().split()
                temp_det_objects.append(read_line)

        for object in temp_det_objects:
            
            cls = 'Egg' if object[0] == '0' else 'Worm'
            cur_yolo_bbox_xywh = (object[1], object[2], object[3], object[4])
            voc_bbox_w = float(cur_yolo_bbox_xywh[2]) * ori_w
            voc_bbox_h = float(cur_yolo_bbox_xywh[3]) * ori_h
            center_x = float(cur_yolo_bbox_xywh[0]) * ori_w
            center_y = float(cur_yolo_bbox_xywh[1]) * ori_h
            voc_x1 = center_x - (voc_bbox_w / 2)
            voc_y1 = center_y - (voc_bbox_h / 2)
            voc_x2 = center_x + (voc_bbox_w / 2)
            voc_y2 = center_y + (voc_bbox_h / 2)
            bbox_xyxy = [voc_x1,voc_y1,voc_x2,voc_y2]
            pos = (round(center_x,2), round(center_y,2))
            size = round(math.hypot(voc_x2-voc_x1, voc_y2-voc_y1),2)
            if cls == 'Egg':
                dev_stage = 'Embryo'
            else:
                dev_stage = dev_stage_from_size(size)
            conf = round(float(object[5]),2)


            #det_object_list.append([cls,pos[0],pos[1],size,conf,dev_stage])
            if size > 100:
                P0_labels_data_list.append([wellname, datetime_object, int(frame_number), ori_w, cls, pos[0], pos[1], size, conf, dev_stage])
    
    P0_df = pd.DataFrame(P0_labels_data_list, columns=['wellname', 'acq_time', 'frame', 'img_w', 'class', 'x', 'y', 'size', 'conf', 'dev_stage'])
    P0_df = P0_df.sort_values(by=['wellname', 'frame'])
    P0_df = P0_df.reset_index()
    print('P0_df:')
    pprint(P0_df)

    for wellname in wellnames:
        
        cur_P0_df = P0_df.loc[P0_df['wellname'] == wellname]
        if cur_P0_df.empty:
            continue
        total_dist = 0

        pprint(cur_P0_df)

        

        vel_avg = 0
        vel_min = 0
        vel_max = 0
        dist_max = 0
        dist_avg = 0
        dist_straight = 0
        dist_total = 0
        disp_x = 0
        disp_y = 0

        x_start = cur_P0_df.at[0, 'x']
        y_start = cur_P0_df.at[0, 'y']

        center = (ori_w / 2, ori_h / 2)

        path_distances = []
        path_points = [(cur_P0_df.at[0, 'x'], cur_P0_df.at[0, 'y'])]
        path_to_start_frame = glob(prcsd_img_folder + f'*{wellname}*f0*.jpeg')[0] 
        #start_img = Image.open(path_to_start_frame)
        #end_img = Image.open(path_to_end_frame)


        #multiplied_canvas = ImageChops.multiply(start_img, end_img)
        #multiplied_canvas.show()

        #cv_multiplied_canvas = np.asarray(multiplied_canvas)

        #subtract_canvas = ImageChops.subtract(start_img, end_img, scale=1.0, offset=0)
        #subtract_canvas.show()

        cv_start_img = cv.imread(path_to_start_frame, cv.IMREAD_GRAYSCALE)

        for frame in range(len(cur_P0_df)):
            cur_dist_xy = 0
            dist_from_start = 0

            if frame != 0:
                

                frame_x1 = cur_P0_df.at[frame - 1, 'x']
                frame_y1 = cur_P0_df.at[frame - 1, 'y']
                frame_x2 = cur_P0_df.at[frame, 'x']
                frame_y2 = cur_P0_df.at[frame, 'y']
                cur_dist_xy = round(math.hypot(frame_x2 - frame_x1, frame_y2 - frame_y1), 2)
                dist_from_start = round(math.hypot(frame_x2 - x_start, frame_y2 - y_start), 2)

                path_distances.append(cur_dist_xy)
                path_points.append((frame_x2, frame_y2))
                
                # print(f'distance: {cur_dist_xy} at frame {frame}')

            # Minimum velocity as px/s
            if vel_min == 0:
                vel_min = cur_dist_xy
            else:
                if cur_dist_xy < vel_min:
                    vel_min = cur_dist_xy
                
            # Maximum velocity as px/s
            if cur_dist_xy > vel_max:
                vel_max = cur_dist_xy

            # Maximum distance
            if dist_from_start > dist_max:
                dist_max = dist_from_start

            
        x_end = cur_P0_df.at[len(cur_P0_df) - 1, 'x']
        y_end = cur_P0_df.at[len(cur_P0_df) - 1, 'y']
        
            
        dist_straight = round(math.hypot(x_end - x_start, y_end - y_start), 2)

        # Total distance
        dist_total = round(sum(path_distances), 2)

        # Average velocity as px/s
        vel_avg = round(dist_total / len(path_distances), 2)

        # Distance from start to average point 
        point_avg = [sum(x) / len(x) for x in zip(*path_points)]
        dist_avg = round(math.hypot(point_avg[0] - x_start, point_avg[1] - y_start), 2)

        # Displacement in x & y from median point (as origin with start-end line as abscissa, the start being -inf and the end being +inf)
        med_point = ((x_start + x_end)/2, (y_start + y_end)/2)
        
        left_mid_point = (0, ori_h / 2)
        actual_to_new_vector = (center[0] - med_point[0], center[1] - med_point[1]) 
        new_start = (x_start + actual_to_new_vector[0], y_start + actual_to_new_vector[1])
        

        theta = angle_from_three_points(new_start, center, left_mid_point)

        new_avg_point = get_xy_from_2D_axes_rotation(point_avg[0] + actual_to_new_vector[0], point_avg[1] + actual_to_new_vector[1], theta)
        new_avg_point = (round(new_avg_point[0], 2), round(new_avg_point[1], 2))

        disp_x = round(math.hypot(new_avg_point[0] - center[0], 0), 2) 
        disp_y = round(math.hypot(0, new_avg_point[1] - center[1]), 2)

        print(f'Average velocity: {vel_avg}')
        print(f'Minimum velocity: {vel_min}')
        print(f'Maximum velocity: {vel_max}')
        print(f'Maximum distance: {dist_max}')
        print(f'Normalized average point: {new_avg_point}')
        print(f'Distance to average point: {dist_avg}')
        print(f'Straight distance: {dist_straight}')
        print(f'Total distance: {dist_total}')
        print(f'X displacement: {disp_x}')
        print(f'Y displacement: {disp_y}')

        
        

        #print(path_to_start_frame)
        #print(path_to_end_frame)

        
        path_to_end_frame = glob(prcsd_img_folder + f'*{wellname}*f{len(cur_P0_df) - 1}*.jpeg')[0]
        cv_end_img = cv.imread(path_to_end_frame, cv.IMREAD_GRAYSCALE)
       

        roi_size = 800

        roi_x = int(center[1]) - int(roi_size / 2)
        roi_y = int(center[0]) - int(roi_size / 2)
        roi_top_left = (roi_x, roi_y)
        roi_bottom_right = (roi_x + roi_size, roi_y + roi_size)
        roi = cv_start_img[roi_x:roi_x + roi_size, roi_y:roi_y + roi_size]

        w, h = roi.shape[::-1]

        res = cv.matchTemplate(cv_end_img, roi, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        #cv.rectangle(cv_multiplied_canvas, top_left, bottom_right, 255, 2)
        #cv.rectangle(cv_multiplied_canvas, roi_top_left, roi_bottom_right, 200, 2)
    

        disp_vector_end_start_frames = (top_left[0] - roi_top_left[0], top_left[1] - roi_top_left[1])
    

        #pprint(f'disp: {disp_vector_end_start_frames}')
        s_top = 0
        s_bottom = 0
        s_left = 0
        s_right = 0

        e_top = 0
        e_bottom = 0
        e_left = 0
        e_right = 0

        if disp_vector_end_start_frames[0] < 0:
            s_right = abs(disp_vector_end_start_frames[0])
            s_left = 0
            e_right = 0
            e_left = abs(disp_vector_end_start_frames[0])
        else:
            s_right = 0
            s_left = abs(disp_vector_end_start_frames[0])
            e_right = abs(disp_vector_end_start_frames[0])
            e_left = 0
        
        if disp_vector_end_start_frames[1] < 0:
            s_top = 0
            s_bottom = abs(disp_vector_end_start_frames[1])
            e_top = abs(disp_vector_end_start_frames[1])
            e_bottom = 0
        else:
            s_top = abs(disp_vector_end_start_frames[1])
            s_bottom = 0
            e_top = 0
            e_bottom = abs(disp_vector_end_start_frames[1])
            
        cv_padded_start_image = cv.copyMakeBorder(cv_start_img, s_top, s_bottom, s_left, s_right, cv.BORDER_CONSTANT, value = [0, 0, 0])
        cv_padded_end_image = cv.copyMakeBorder(cv_end_img, e_top, e_bottom, e_left, e_right, cv.BORDER_CONSTANT, value = [0, 0, 0])
        pil_padded_start_image = Image.fromarray(cv_padded_start_image)
        #pil_padded_start_image.show()
        pil_padded_end_image = Image.fromarray(cv_padded_end_image)
        #pil_padded_end_image.show()
        padded_multiplied_canvas = ImageChops.multiply(pil_padded_start_image, pil_padded_end_image)
        #padded_multiplied_canvas.show()
        cv_padded_multiplied_canvas = np.asarray(padded_multiplied_canvas)

        cnt = 0
        color = 0
        starting_color = Color(rgb=(15/255, 253/255, 250/255))  # Crayol Sun Yellow
        color_gradient = list(starting_color.range_to(Color(rgb=(255/255, 255/255, 255/255)),len(path_points)))
        
        cv_color_padded_multiplied_canvas = cv.cvtColor(cv_padded_multiplied_canvas,cv.COLOR_GRAY2RGB)
        font = cv.FONT_HERSHEY_PLAIN
        font_scale = 1
        font_color = (0, 0, 0)
        font_thickness = 2

        h, w, _ = cv_color_padded_multiplied_canvas.shape
        cv_color_padded_multiplied_canvas = cv.line(cv_color_padded_multiplied_canvas, (0, int(h / 2)), (w, int(h / 2)), (0, 0, 0), 4)  # X line
        cv_color_padded_multiplied_canvas = cv.line(cv_color_padded_multiplied_canvas, (int(w / 2), 0), (int(w / 2), h), (0, 0, 0), 4)  # Y line
        cv_color_padded_multiplied_canvas = cv.line(cv_color_padded_multiplied_canvas, (int(path_points[0][0]), int(path_points[0][1])), (int(path_points[len(path_points) - 1][0]), int(path_points[len(path_points) - 1][1])), (0, 0, 0), 4)  # Efficient path line


        for point in path_points:
            #pprint(point)
            cur_x = int(point[0])
            cur_y = int(point[1])
            
            cur_color = (color_gradient[cnt].rgb[0] * 255, color_gradient[cnt].rgb[1] * 255, color_gradient[cnt].rgb[2] * 255)
            cv_color_padded_multiplied_canvas = cv.circle(cv_color_padded_multiplied_canvas, (cur_x, cur_y), radius = 1, color = cur_color, thickness = 2)
            if cnt >= 1:
                #print('cnt is >= 1')
                cv_color_padded_multiplied_canvas = cv.line(cv_color_padded_multiplied_canvas, (int(path_points[cnt - 1][0]), int(path_points[cnt - 1][1])), (cur_x, cur_y), cur_color, 2)
            cnt += 1

        
        cnt = 0
        for point in path_points:
            cur_x = int(point[0])
            cur_y = int(point[1])
            if cnt % 10 == 0:
                cv_color_padded_multiplied_canvas = cv.putText(cv_color_padded_multiplied_canvas, str(cnt), (cur_x, cur_y), font, font_scale, font_color, font_thickness)
            cnt += 1

        

        
        cv_color_padded_multiplied_canvas = cv.circle(cv_color_padded_multiplied_canvas, (int(point_avg[0]), int(point_avg[1])), radius = 1, color = (0, 30, 150), thickness = 5)
        cv_color_padded_multiplied_canvas = cv.putText(cv_color_padded_multiplied_canvas, 'AVG', (int(point_avg[0] + 10), int(point_avg[1]) - 10), font, 1.5, (0, 30, 150), 3)
        cv_color_padded_multiplied_canvas = cv.circle(cv_color_padded_multiplied_canvas, (int(med_point[0]), int(med_point[1])), radius = 1, color = (0, 150, 30), thickness = 5)
        cv_color_padded_multiplied_canvas = cv.putText(cv_color_padded_multiplied_canvas, 'MED', (int(med_point[0] + 10), int(med_point[1]) - 10), font, 1.5, (0, 150, 30), 3)



        cv.imshow('Summary_image', cv_color_padded_multiplied_canvas)
        cv.waitKey(0)
        cv.destroyAllWindows()


        print(wellname)
        print(len(cur_P0_df))


    # crop files for P0
    all_bbox_list = glob(fn + 'exp*/crops/Worm/*s_f*_[2-3]*px*.jpg')
    
    P0_bbox_list = []
    for bbox in all_bbox_list:
        h, w = imagesize.get(bbox)
        size = round(math.hypot(h, w),2)
        print(size)
        if size > 100:
            P0_bbox_list.append(bbox)

    pprint(P0_bbox_list)
    print(len(P0_bbox_list))

    pattern = re.escape(fn) + '.*exp\d*\/crops\/Worm\/PRCSD_(.*)_(\d{2}-\d{2}-\d{4}_\d{2}h\d{2}m\d{2}s)_(.*)px.'  #'CeSAR_infer/exp2/labels/PRCSD_B10_17-06-2022_06h03m49s_2876px.txt']
    #                                                                                                                                   PRCSD_B9_20-07-2022_19h38m06s_f18_2816px.txt
    frame_img_pattern = r'f(\d+)_(\d+)'

    P0_bbox_data_list = []

    # labels_files for P0
    for P0_bbox_path in P0_bbox_list:
        print(P0_bbox_path)

        bbox = cv.imread(P0_bbox_path, 0)
        re_match = re.search(pattern, P0_bbox_path)
        frame_img_re_match = re.search(frame_img_pattern, re_match.group(3))
        print(re_match.groups())
        wellname = re_match.group(1)
        acquisiton_time = re_match.group(2)
        frame_number = int(frame_img_re_match.group(1))
        ori_w = ori_h = int(frame_img_re_match.group(2))
        datetime_object = datetime.strptime(acquisiton_time, '%d-%m-%Y_%Hh%Mm%Ss')
        formatted_datetime_object = datetime_object.strftime("%d-%m-%Y_%Hh%Mm%Ss") # 31-07-2013_03h12m12s
        year = int(acquisiton_time[6:10])
        month = int(acquisiton_time[3:5])
        day = int(acquisiton_time[0:2])
        hours = int(acquisiton_time[11:13])
        minutes = int(acquisiton_time[14:16])
        seconds = int(acquisiton_time[17:19])

        datetime_object = datetime(year, month, day, hours, minutes, seconds)
        
        temp_det_objects = []
        with open(P0_labels_file_path) as file:
            for line in file:
                read_line = line.rstrip().split()
                temp_det_objects.append(read_line)

        for object in temp_det_objects:
            
            cls = 'Egg' if object[0] == '0' else 'Worm'
            cur_yolo_bbox_xywh = (object[1], object[2], object[3], object[4])
            voc_bbox_w = float(cur_yolo_bbox_xywh[2]) * ori_w
            voc_bbox_h = float(cur_yolo_bbox_xywh[3]) * ori_h
            center_x = float(cur_yolo_bbox_xywh[0]) * ori_w
            center_y = float(cur_yolo_bbox_xywh[1]) * ori_h
            voc_x1 = center_x - (voc_bbox_w / 2)
            voc_y1 = center_y - (voc_bbox_h / 2)
            voc_x2 = center_x + (voc_bbox_w / 2)
            voc_y2 = center_y + (voc_bbox_h / 2)
            bbox_xyxy = [voc_x1,voc_y1,voc_x2,voc_y2]
            pos = (round(center_x,2), round(center_y,2))
            size = round(math.hypot(voc_x2-voc_x1, voc_y2-voc_y1),2)
            if cls == 'Egg':
                dev_stage = 'Embryo'
            else:
                dev_stage = dev_stage_from_size(size)
            conf = round(float(object[5]),2)


            #det_object_list.append([cls,pos[0],pos[1],size,conf,dev_stage])
            if size > 100:
                P0_labels_data_list.append([wellname, datetime_object, int(frame_number), ori_w, cls, pos[0], pos[1], size, conf, dev_stage])

        bbox_cp = bbox.copy()
        frame_to_save = 0
        if frame_number == frame_to_save: cv.imwrite('bbox_cp.jpeg', bbox_cp)
        #cv.waitKey(0)
        # Otsu's thresholding after Gaussian filtering
        #blur_bilat_filt = cv.bilateralFilter(bbox,15,15,15)
        blur = cv.GaussianBlur(bbox,(7,7),150)
        blur = cv.bilateralFilter(blur,15,15,15)
        blur = cv.bilateralFilter(blur,5,5,50)
        ret3,th3 = cv.threshold(blur,127,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        if frame_number == frame_to_save: cv.imwrite('blur.jpeg', blur)
        #cv.waitKey(0)

        if frame_number == frame_to_save: cv.imwrite('thresh.jpeg', th3)
        #cv.waitKey(0)
        
        #inv = cv.bitwise_not(th3)
        erosion_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        #opening_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
        #closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

        #dilation = cv.dilate(th3, dilation_kernel, iterations = 3)
        #cv.imshow('dilation1', dilation)
        #cv.waitKey(0)

        erosion = cv.erode(th3, erosion_kernel, iterations = 2)

        if frame_number == frame_to_save: cv.imwrite('erosion1.jpeg', erosion)
        #cv.waitKey(0)

        dilation = cv.dilate(erosion, dilation_kernel, iterations = 2)

        if frame_number == frame_to_save: cv.imwrite('dilation2.jpeg', dilation)
        #cv.waitKey(0)

        #erosion = cv.erode(dilation, erosion_kernel, iterations = 1)

        #cv.imshow('erosion2', erosion)
        #cv.waitKey(0)

        contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_TC89_KCOS)#)
        cv.drawContours(bbox_cp, contours, -1, color=(0,0,0), thickness=2)
        #pprint("Contours ARE:")
        #pprint(contours)
        #filename = os.path.splitext(str(name))[0]

        #black_canvas = np.zeros([100,100,3], dtype=np.uint8)
        ske = (skeletonize(dilation//255) * 255).astype(np.uint8)

        if frame_number == frame_to_save: cv.imwrite('contours.jpeg', bbox_cp)
        #cv.waitKey(0)

        if frame_number == frame_to_save: cv.imwrite('P0_midline.jpeg', ske)
        #cv.waitKey(0)

        fil = FilFinder2D(ske, distance=250 * u.pc, mask=ske)
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

        true_fil = fil.filaments[0]
        #print(f'EXEC_RHT {fil.exec_rht()}')
        #print('FIL')
        #pprint(fil.filaments[0])

        #pprint(dir(fil))
        #pprint(dir(true_fil))
        #pprint(dir(fil.skeleton))
        #print()
        #pprint(f'filament properties: {true_fil.branch_properties}')
        #pprint(f'filament pts: {true_fil.longpath_pixel_coords}')
        #pprint(f'filament table: {true_fil.branch_table()}')
        #pprint(f'filament length: {true_fil.length()}')
        pprint(f'filament end points: {true_fil.end_pts}')
        x_coords = []
        y_coords = []

        #for xy in true_fil.longpath_pixel_coords:
        for x in true_fil.longpath_pixel_coords[1]:  # x and y seem to be inverted here. Simple fix is used
            x_coords.append(x)

        for y in true_fil.longpath_pixel_coords[0]:
            y_coords.append(y)

        coords = list(zip(x_coords, y_coords))

        P0_img_w = 200
        P0_img_h = 200


        P0_mid_point = (0,0)
        if len(coords) % 2:
            mid_x = (coords[int(len(coords) / 2) - 1][0] + coords[int(len(coords) / 2) - 1][0]) / 2
            mid_y = (coords[int(len(coords) / 2) - 1][1] + coords[int(len(coords) / 2) - 1][1]) / 2
            P0_mid_point = (mid_x, mid_y)
        else:
            mid_x = coords[int(len(coords) / 2) - 1][0]
            mid_y = coords[int(len(coords) / 2) - 1][1]
            P0_mid_point = (mid_x, mid_y)

        disp_vector = ((P0_img_w / 2) - P0_mid_point[0], (P0_img_h / 2) - P0_mid_point[1])

        P0_img = Image.new('RGB', (P0_img_w, P0_img_h))
        
        

        endpoints = true_fil.end_pts

        for xy in coords:
            
            px_x = int(xy[0] + disp_vector[0])
            px_y = int(xy[1] + disp_vector[1])
            #print(f'step-{int(step)}   .   cnt{cnt}.............{(px_x,px_y)}')
            #if cnt % 10 == 0:
            #    for i in range(-2,3):
            #        P0_img.putpixel((px_x+i, px_y), (255,245,25))
            #        P0_img.putpixel((px_x, px_y+i), (255,245,25))
            P0_img.putpixel((px_x, px_y), (255,255,255))
        
        line = []
        for x in range(P0_img_w):
            for y in range(P0_img_h):
                if P0_img.getpixel((x,y)) == (255,255,255):
                    line.append((x,y))
        pprint(line)
        pprint(f'line: {line}')
        true_end_pts = []
        nb_neighbors = 0

        for pt in line:
            cur_neighbors = []

            cur_x = pt[0]
            cur_y = pt[1]

            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i != 0 or j != 0:
                        print(P0_img.getpixel((cur_x+i, cur_y+j)))
                        if P0_img.getpixel((cur_x+i, cur_y+j)) == (255, 255, 255):
                            nb_neighbors += 1
                            cur_neighbors.append((cur_x+i, cur_y+j))
                            
            dist = 0
            if len(cur_neighbors) == 2:
                dist = math.dist(cur_neighbors[0], cur_neighbors[1])
            if nb_neighbors <= 2 and dist <= 1.1:
                true_end_pts.append((cur_x, cur_y))
            else:
                nb_neighbors = 0
                cur_neighbors = []

            print()

        pprint(f'true_end_pts: {true_end_pts}')
        step = int(len(line) / 10)
        traversed_ske = False
        line_len = len(line)
        cur_point = true_end_pts[0]
        last_point = (-1, -1)
        next_point = (-1, -1)
        cnt = 0
        nb_neighbors = 0

        keypoints = []
        pprint(P0_img)
        while not traversed_ske:
            nb_neighbors = 0
            last_x = last_point[0]
            last_y = last_point[1]
            cur_x = cur_point[0]
            cur_y = cur_point[1]
            next_x = next_point[0]
            next_y = next_point[1]

            print(f'Current (X, Y): ({cur_x}, {cur_y})')
            print(f'get px value: {P0_img.getpixel(cur_point)}')
            if cnt % step == 0 and cnt < line_len - int(step * 0.75):
                print('KEYPOINT -> YELLOW CROSS')
                keypoints.append(cur_point)
            smallest_dist = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i != 0 or j != 0:
                        print(P0_img.getpixel((cur_x+i, cur_y+j)))
                        if P0_img.getpixel((cur_x+i, cur_y+j)) == (255,255,255):
                            nb_neighbors += 1
                            cur_dist = math.dist(cur_point, (cur_x + i, cur_y + j))
                            if smallest_dist == 0 or cur_dist <= smallest_dist:
                                smallest_dist = cur_dist
                            if last_x != cur_x+i or last_y != cur_y+j and cur_dist <= smallest_dist:  # Not the last
                                next_point = (cur_x + i, cur_y + j)
                                

            last_point = (cur_point[0], cur_point[1])
            cur_point = (next_point[0], next_point[1])
            P0_img.putpixel(last_point, (254,254,254))
            print(f'#neighbors: {nb_neighbors}')
            if nb_neighbors <= 0 and cnt != 0:
                traversed_ske = True

            cnt += 1
            
        keypoints.append(true_end_pts[1])

            
        pprint(keypoints)   
        
        for index, key_pt in enumerate(keypoints):
            color = (255, 245, 25)
            cross_range = 2
            if index == 0 or index == len(keypoints) - 1:
                color = (255, 25, 15)
                cross_range = 4
            for i in range(-cross_range,cross_range + 1):
                P0_img.putpixel((key_pt[0]+i, key_pt[1]), color)
                P0_img.putpixel((key_pt[0], key_pt[1]+i), color)

            

            
        P0_img.putpixel((keypoints[0]), (255,25,15))  # RGB

        cv_P0_img = np.asarray(P0_img)
        

        #for step in range(11):
        #    ske_idx = step * 
        print(len(coords))  

        #ske_idxs = (x * step for x in range(1, 10))

        #cv_P0_img = cv.circle(cv_P0_img, (int(coords[0][0] + disp_vector[0]), int(coords[0][1] + disp_vector[1])), radius = 0, color = (245, 250, 25), thickness = 5)

        #for ske_idx in ske_idxs:
        #    int_ske_idx = int(ske_idx)
        #    if int_ske_idx > (len(coords) - 1):
        #        int_ske_idx = len(coords) - 1
        #    
        #    ske_pt_x = coords[int_ske_idx][0] #+ disp_vector[0]
        #    ske_pt_y = coords[int_ske_idx][1] #+ disp_vector[1]
        #    cv_P0_img = cv.circle(cv_P0_img, (int(ske_pt_x), int(ske_pt_y)), radius = 0, color = (245, 250, 25), thickness = 5)

        #cv_P0_img = cv.circle(cv_P0_img, (int(coords[int(len(coords)) - 1][0] + disp_vector[0]), int(coords[int(len(coords)) - 1][1] + disp_vector[1])), radius = 0, color = (245, 250, 25), thickness = 5)
        

        #cv_P0_img = cv.putText(cv_P0_img, 'AVG', (int(point_avg[0] + 10), int(point_avg[1]) - 10), font, 1.5, (0, 30, 150), 3)

        P0_bbox_data_list.append([wellname, acquisiton_time, frame_number, cv_P0_img])
        
        #P0_img_channels = 1
        #P0_img = np.zeros((P0_img_w, P0_img_w, P0_img_channels), dtype=np.uint8)
        
        
        #for y in range(P0_img.shape[0]):  # seems to be inverted again
        #    for x in range(P0_img.shape[1]):
        #        if 
        #P0_arr = np.array(coords, dtype=np.uint8)
        #P0_img = Image.fromarray(P0_arr)
        #P0_img.show()
        #plt.rcParams["figure.figsize"] = [7.50, 3.50]
        #plt.rcParams["figure.autolayout"] = True

        #plt.plot(coords, 'r*')
        #plt.axis([0,300,0,300])

        #plt.show()
        #fil.exec_rht()
        #fil.plot_rht_distrib()
        #true_fil.length()
        #pprint(true_fil.orientation, true_fil.curvature)
        # Show the longest path
        #pprint(fil.skeleton.ravel())
        #plt.imshow(fil.skeleton, cmap='gray')
        #plt.contour(fil.skeleton_longpath, colors='r')
        #plt.axis('off')
        #plt.show()

        #break
    

    P0_bbox_df = pd.DataFrame(P0_bbox_data_list, columns=['wellname', 'acq_time', 'frame', 'P0_img'])
    P0_bbox_df = P0_bbox_df.sort_values(by=['wellname', 'frame'])
    P0_bbox_df = P0_bbox_df.reset_index()

    for wellname in wellnames:
        if P0_bbox_df.empty:
            continue   
        images = []
        cur_P0_df = P0_bbox_df.loc[P0_df['wellname'] == wellname]
        pprint(cur_P0_df)
        for idx in range(len(cur_P0_df)):
            img = cur_P0_df.at[idx, 'P0_img']
            #img.show()
            images.append(img)
        imageio.mimsave('gif.gif', images, fps=2)
        


       
    pprint(P0_df)

    #print(P0_bbox_list)

    global_labels_data_list = sorted(global_labels_data_list)
    pprint(global_labels_data_list)
    df = pd.DataFrame(global_labels_data_list, columns=['wellname', 'acq_time', 'img_w', 'class', 'x', 'y', 'size', 'conf', 'dev_stage', 'hour', 'ref_time', 'well_key'])
    df = df.reindex(columns=['well_key', 'wellname', 'ref_time', 'acq_time', 'hour', 'img_w', 'class', 'dev_stage', 'x', 'y', 'size', 'conf'])
    pprint(df)

    for wellname in wellnames:
        wellname_data = {'wellname':[], 'hour':[], 'dev_stage':[]}

        wellname_list = []
        hour_list = []
        dev_stage_list = []
        pos_list = []
        
        cur_df = df.loc[df['wellname'] == wellname]
        cur_df.reset_index()
        hours = []
        # Populate the wellnames
        cur_df_hours = cur_df.loc[:,'hour']
        for hour in cur_df_hours:
            if hour not in hours:
                hours.append(hour)
        cur_df_hours.reset_index()
        # Order the hours
        hours = sorted(hours)

        for hour in hours:
            hr_cur_df = cur_df.loc[cur_df['hour'] == hour]
            hr_cur_df.reset_index()
            print('HOUR DF')
            pprint(hr_cur_df)
            for idx in range(len(hr_cur_df)):
                #print(idx)
                #print(hr_cur_df.iloc[idx,7])
                dev_stg = hr_cur_df.iloc[idx,7]
                if dev_stg == 'Embryo':
                    dev_stage_list.append('Embryo')
                elif dev_stg == 'L1':
                    dev_stage_list.append('L1')
                elif dev_stg == 'L2':
                    dev_stage_list.append('L2')
                elif dev_stg == 'L3':
                    dev_stage_list.append('L3')
                elif dev_stg == 'L4':
                    dev_stage_list.append('L4')
                elif dev_stg == 'Ad':
                    dev_stage_list.append('Ad')
                else:
                    dev_stage_list.append('Embryo')
                """match dev_stg:
                    case 'Embryo':
                        dev_stage_list.append('Embryo')
                    case 'L1':
                        dev_stage_list.append('L1')
                    case 'L2':
                        dev_stage_list.append('L2')
                    case 'L3':
                        dev_stage_list.append('L3')
                    case 'L4':
                        dev_stage_list.append('L4')
                    case 'Ad':
                        dev_stage_list.append('Ad')
                    case _:
                        print('CRITICAL ERROR. Erroneous dev_stage in df.')
                """
                
                hour_list.append(hour)
                wellname_list.append(wellname)

       
        

        """pprint(labels_data[3])
        for labels_data in global_labels_data_list:
            if wellname == labels_data[0]:
                for i in range(labels_data[4]): # Embryo
                    wellname_list.append(labels_data[0])
                    hour_list.append(int(labels_data[3]))
                    dev_stage_list.append('Embryo')
                for i in range(labels_data[5]): # L1
                    wellname_list.append(labels_data[0])
                    hour_list.append(labels_data[3])
                    dev_stage_list.append('L1')
                for i in range(labels_data[6]): # L2
                    wellname_list.append(labels_data[0])
                    hour_list.append(labels_data[3])
                    dev_stage_list.append('L2')
                for i in range(labels_data[7]): # L3
                    wellname_list.append(labels_data[0])
                    hour_list.append(labels_data[3])
                    dev_stage_list.append('L3')
                for i in range(labels_data[8]): # L4
                    wellname_list.append(labels_data[0])
                    hour_list.append(labels_data[3])
                    dev_stage_list.append('L4')
                for i in range(labels_data[9]): # Ad
                    wellname_list.append(labels_data[0])
                    hour_list.append(labels_data[3])
                    dev_stage_list.append('Ad')
        """
        print(len(wellname_list))
        print(len(hour_list))
        print(len(dev_stage_list))
        wellname_data['wellname'] = wellname_list
        wellname_data['hour'] = hour_list
        wellname_data['dev_stage'] = dev_stage_list

        wellname_df = pd.DataFrame(wellname_data)

        f, ax = plt.subplots(figsize=(7,5))
        sns.despine(f)

        graph = sns.histplot(
            wellname_df,
            x='hour', hue='dev_stage',
            multiple='stack',
            palette='Reds',
            #edgecolor='.3',
            #linewidth=.5
            binwidth=1,
            discrete=True
        )

        pprint(wellname_df)

        #hourly_broodsize = []
        #for hour in range(max(hour_list)):
        #    bs = wellname_df[wellname_df.hour == hour].count()['wellname']
        #    pprint(f'{hour}, {bs}')

        ##pprint(hourly_broodsize)

        
        
        plt.show()
        #plt.legend(labels=['Embryo','L1','L2','L3','L4','Ad'])
        #plt.legend(labels=['Ad','L4','L3','L2','L1','Embryo'])
        graph.figure.savefig(f'{modified_img_dir_path}graphs/BS/{wellname}_Broodsize.jpeg')
        #pprint(wellname_df)

    
else:
    print('INVALID PATH: Please enter a valid path as first argument.')
    print('Exiting now.')
    sys.exit()





























    

























    

























    

























    

























    
    #print(pattern)

    """for labels_file in labels_files_list:
        # record every line of the current labels file
        
        #print(labels_file)
        re_match = re.search(pattern, labels_file)

        wellname = re_match.groups()[0]
        acquisiton_time = re_match.groups()[1]
        ori_w = ori_h = int(re_match.groups()[2])
        datetime_object = datetime.strptime(acquisiton_time, '%d-%m-%Y_%Hh%Mm%Ss')
        formatted_datetime_object = datetime_object.strftime("%d-%m-%Y_%Hh%Mm%Ss") # 31-07-2013_03h12m12s
        year = int(acquisiton_time[6:10])
        month = int(acquisiton_time[3:5])
        day = int(acquisiton_time[0:2])
        hours = int(acquisiton_time[11:13])
        minutes = int(acquisiton_time[14:16])
        seconds = int(acquisiton_time[17:19])

        datetime_object = datetime(year, month, day, hours, minutes, seconds)

        # FORMAT: Cls-X-Y-W-H-Conf
        temp_det_objects = []
        with open(labels_file) as file:
            for line in file:
                read_line = line.rstrip().split()
                temp_det_objects.append(read_line)

        # FORMAT: {'class':[], 'x':[], 'y':[], 'size':[], 'conf':[], 'dev_stage':[]}
        det_object_list = []
        for object in temp_det_objects:
            
            cls = 'Egg' if object[0] == '0' else 'Worm'
            cur_yolo_bbox_xywh = (object[1], object[2], object[3], object[4])
            voc_bbox_w = float(cur_yolo_bbox_xywh[2]) * ori_w
            voc_bbox_h = float(cur_yolo_bbox_xywh[3]) * ori_h
            center_x = float(cur_yolo_bbox_xywh[0]) * ori_w
            center_y = float(cur_yolo_bbox_xywh[1]) * ori_h
            voc_x1 = center_x - (voc_bbox_w / 2)
            voc_y1 = center_y - (voc_bbox_h / 2)
            voc_x2 = center_x + (voc_bbox_w / 2)
            voc_y2 = center_y + (voc_bbox_h / 2)
            bbox_xyxy = [voc_x1,voc_y1,voc_x2,voc_y2]
            pos = (round(center_x,2), round(center_y,2))
            size = round(math.hypot(voc_x2-voc_x1, voc_y2-voc_y1),2)
            if cls == 'Egg':
                dev_stage = 'Embryo'
            else:
                dev_stage = dev_stage_from_size(size)
            conf = round(float(object[5]),2)


            det_object_list.append([cls,pos[0],pos[1],size,conf,dev_stage])
    """
    """
        data = {'class':[], 'x':[], 'y':[], 'size':[], 'conf':[], 'dev_stage':[]}
        #pprint(det_object_list)
        cls_list = []
        x_list = []
        y_list = []
        size_list = []
        conf_list = []
        dev_stage_list = []

        for det_object in det_object_list:
            cls_list.append(det_object[0])
            x_list.append(det_object[1])
            y_list.append(det_object[2])
            size_list.append(det_object[3])
            conf_list.append(det_object[4])
            dev_stage_list.append(det_object[5])


        data['class'] = cls_list
        data['x'] = x_list
        data['y'] = y_list
        data['size'] = size_list
        data['conf'] = conf_list
        data['dev_stage'] = dev_stage_list

        #pprint(data)    

        df = pd.DataFrame(data)

        #pprint(df)

        eggs = df[df['class'] == 'Egg']
        worm = df[df['class'] == 'Worm']
        
        worm_incre_size = worm.sort_values(by='size').reset_index()
        #groupby('size').size().sort_values()
        #pprint(worm_incre_size)


        # SPECIFIC HOUR FOR A WELL

        # Positions of eggs and worms for a specific hour for a well
        #graph = sns.jointplot(data=df, x='x', y='y', hue='class', hue_order = ['Egg', 'Worm'], alpha=0.5, space=0)
        #graph.plot(alpha=0.1)
        #graph.ax_marg_x.set_xlim(0,ori_w)
        #graph.ax_marg_y.set_ylim(0,ori_h)
        #plt.show()
        #graph.figure.savefig(f'{modified_img_dir_path}graphs/pop_heatmap/{wellname}_{formatted_datetime_object}.jpeg')


        # Sizes for a specific hour for a well
        #graph = sns.jointplot(data=worm_incre_size, x=worm_incre_size.index, y='size', alpha=0.5, space=0)
        #graph.ax_marg_x.set_xlim(10,25)
        #graph.ax_marg_y.set_ylim(0,250)
        #plt.show()
        #graph.figure.savefig(f'{modified_img_dir_path}graphs/pop_size/{wellname}_{formatted_datetime_object}.jpeg')


        # Correlations of a specific hour for a well
        #graph = sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})
        #plt.show()
        #graph.figure.savefig(f'{modified_img_dir_path}graphs/correl/{wellname}_{formatted_datetime_object}.jpeg')

        
        # STATS for each WELL
        for labels_data in global_labels_data_list:  #['B10', datetime.datetime(2022, 6, 16, 11, 8, 28), 0]
            #'CeSAR_infer/exp2/labels/PRCSD_B10_17-06-2022_06h03m49s_2876px.txt']
            if labels_data[0] == wellname and labels_data[1] == datetime_object:
                num_embryo = len(df[(df['dev_stage']=='Embryo')])
                num_L1 = len(df[(df['dev_stage']=='L1')])
                num_L2 = len(df[(df['dev_stage']=='L2')])
                num_L3 = len(df[(df['dev_stage']=='L3')])
                num_L4 = len(df[(df['dev_stage']=='L4')])
                num_Ad = len(df[(df['dev_stage']=='Ad')])
                labels_data.extend([num_embryo, num_L1, num_L2, num_L3, num_L4, num_Ad])
                break

        

    pprint(global_labels_data_list)

    """
        

    
    #with open(fn + 'CeSAR_inference_data.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(header)

        # BS/16-06-2022_155929/wells/A12/16-06-2022_15:59:29_A12_h7.jpeg
            #writer.writerow([exp, wellname, acquisiton_time, 0, 0])
    
    