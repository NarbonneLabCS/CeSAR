from datetime import datetime
import glob
import math
import os
import pathlib
from pickletools import read_decimalnl_short
from pprint import pprint
import sys
from glob import glob
import shutil
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

if os.path.exists(fn):

     # Gets CUR_DIR
    directory_in_str = pathlib.Path(__file__).parent.absolute()
    directory = os.fsencode(directory_in_str)
    modified_img_dir_path = os.path.join(os.fsdecode(directory), 'CeSAR_infer/')
    
    # define the access rights
    access_rights = 0o755

    init_dirs(modified_img_dir_path, access_rights)

    pattern = re.escape(fn) + r'PRCSD_(.*)_(\d{2}-\d{2}-\d{4}_\d{2}h\d{2}m\d{2}s)_(.*)px.jpeg'  # PRCSD_B9_20-07-2022_19h39m06s_f75_2896px.jpeg
    #                                                                    -- -------------------- ----------
    #                                                              PRCSD_B9_15-06-2022_19h06m34s_2856px.jpeg
    #                                                                    -- -------------------- ------
    frame_pattern = r'(f\d+)'
    print(pattern)

    img_list = glob(fn + '*.jpeg')
    pprint(str(img_list))
    img_size_list = []

    for img_path in img_list:
        re_match = re.search(pattern, img_path)
        re_frame_match = re.search(frame_pattern, img_path)

        img_size = ''
        if re_frame_match is not None:
            img_size = re_match.group(3)[-4:]
        else:
            img_size = re_match.group(3)

        if img_size not in img_size_list:
            img_size_list.append(img_size)

    pprint(img_size_list)
    for img_size in img_size_list:
        init_dirs(modified_img_dir_path + img_size + 'px/', access_rights)
        for img_path in img_list:
            re_match = re.search(pattern, img_path)
            cur_img_size = ''
            if re_frame_match is not None:
                cur_img_size = re_match.group(3)[-4:]
            else:
                cur_img_size = re_match.group(3)
            if cur_img_size == img_size:
                shutil.copyfile(img_path, modified_img_dir_path + img_size + 'px/' + os.path.basename(img_path))
        
        
         # Call inference on all processed images
        os.system(f'python3 yolov5/detect.py --project {modified_img_dir_path} --weights 21June2022_best.pt --img {img_size} --conf 0.5 --source {modified_img_dir_path}{img_size}px/ --line-thickness 2 --save-crop --save-conf --save-txt')
        #shutil.rmtree(modified_img_dir_path + img_size + 'px/')
        
else:
    print('INVALID PATH: Please enter a valid path as first argument.')
    print('Exiting now.')
    sys.exit()    