from datetime import datetime
from enum import unique
import glob
from importlib.resources import path
import math
import os
import pathlib
from pprint import pprint
import sys
from glob import glob
import time
import csv
import re
from PIL import Image, ImageFile, ImageEnhance, ImageFilter, ImageOps, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2 as cv
import imagesize
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.set_printoptions(threshold=np.inf)

try:
    fn = sys.argv[1]
except Exception as e:
    #print(str(e))
    print('NO PATH ENTERED: Please write a valid path as first argument.')
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

if os.path.exists(fn):
    csv_header = ['Experiment_start','Well', 'Time', 'Egg', 'Worm']

    # Gets CUR_DIR
    directory_in_str = pathlib.Path(__file__).parent.absolute()
    directory = os.fsencode(directory_in_str)
    modified_img_dir_path = os.path.join(os.fsdecode(directory), "CeSAR_graphs/")
    
    # define the access rights
    access_rights = 0o755

    init_dirs(modified_img_dir_path, access_rights)
    
    
    # Extract data from labels
    labels_files_list = glob(fn + 'exp*/labels/*.txt')
    init_dirs(modified_img_dir_path + '/graphs', access_rights)
    init_dirs(modified_img_dir_path + '/graphs/pop_heatmap', access_rights)
    init_dirs(modified_img_dir_path + '/graphs/pop_size', access_rights)
    init_dirs(modified_img_dir_path + '/graphs/BS', access_rights)
    init_dirs(modified_img_dir_path + '/graphs/correl', access_rights)

    pattern = re.escape(fn) + 'exp\d*\/labels\/PRCSD_(.*)_(.*_.*)_(.*)px.txt'  #'CeSAR_infer/exp2/labels/PRCSD_B10_17-06-2022_06h03m49s_2876px.txt']

    labels_data_list = []

    for labels_file_path in labels_files_list:
        re_match = re.search(pattern, labels_file_path)
        
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
        
            labels_data_list.append([wellname, datetime_object, ori_w, cls, pos[0], pos[1], size, conf, dev_stage])

    # Populate the wellnames
    wellnames = []
    start_times = []
    for labels_data in labels_data_list:
        if labels_data[0] not in wellnames:
            wellnames.append(labels_data[0])
            start_times.append('')

    wellnames = sorted(wellnames)

    # Populate the acquisition times
    wellname_start_time = dict(zip(wellnames, start_times))
    for wellname in wellnames:
        for labels_data in labels_data_list:
            if labels_data[0] == wellname:
                if not wellname_start_time[wellname]:
                    wellname_start_time[wellname] = labels_data[1]
                if  labels_data[1] < wellname_start_time[wellname]:
                    wellname_start_time[wellname] = labels_data[1]

    for labels_data in labels_data_list:
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

    labels_data_list = sorted(labels_data_list)
    df = pd.DataFrame(labels_data_list, columns=['wellname', 'acq_time', 'img_w', 'class', 'x', 'y', 'size', 'conf', 'dev_stage', 'hour', 'ref_time', 'well_key'])
    df = df.reindex(columns=['well_key', 'wellname', 'ref_time', 'acq_time', 'hour', 'img_w', 'class', 'dev_stage', 'x', 'y', 'size', 'conf'])
    pprint(df)
    
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
        for labels_data in labels_data_list:  #['B10', datetime.datetime(2022, 6, 16, 11, 8, 28), 0]
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

        

    pprint(labels_data_list)

    for wellname in wellnames:
        wellname_data = {'wellname':[], 'hour':[], 'dev_stage':[], 'pos':[]}

        wellname_list = []
        hour_list = []
        dev_stage_list = []
        pos_list = []

        for labels_data in labels_data_list:
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

        
        
        #plt.show()
        #plt.legend(labels=['Embryo','L1','L2','L3','L4','Ad'])
        #plt.legend(labels=['Ad','L4','L3','L2','L1','Embryo'])
        #graph.figure.savefig(f'{modified_img_dir_path}graphs/BS/{wellname}_Broodsize.jpeg')
        #pprint(wellname_df)

    
    #with open(fn + 'CeSAR_inference_data.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(header)

        # BS/16-06-2022_155929/wells/A12/16-06-2022_15:59:29_A12_h7.jpeg
            #writer.writerow([exp, wellname, acquisiton_time, 0, 0])
    """
    