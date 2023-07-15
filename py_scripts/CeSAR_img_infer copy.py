from datetime import datetime
import glob
import os
import pathlib
from pprint import pprint
import sys
from glob import glob
import time
import csv
import re
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
import cv2 as cv




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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def modify_img (img_path, img_name, output_dir):
    image = Image.open(img_path)

    enhancer = ImageEnhance.Sharpness(image)
    gray_enhanced_im = ImageOps.grayscale(enhancer.enhance(5.0))
    gray_enhanced_im.save(output_dir + '/ENHANCED_' + img_name, quality=80)
    
    cv_image = np.array(gray_enhanced_im) 
    # Convert RGB to BGR 
    #cv_image = cv_image[:, :, ::-1].copy()

    scale_percent = 10/100 #20 # percent of original size to get 1264 px wide image to match the rest (non necessary)
    resized_w = int(4056 * scale_percent)
    resized_h = int(3040 * scale_percent)
    dim = (resized_w, resized_h)    
    gray = cv.resize(cv_image, dim, interpolation = cv.INTER_AREA)
    #gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    minR = int(1325 * scale_percent)#280
    maxR = int(1600 * scale_percent)#300
    print(f"Looking for circle with radius {minR} < r < {maxR}")
    
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,3000,param1=150,param2=20,minRadius=minR,maxRadius=maxR)# param1=150 #param2=0.9

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:

            # draw the outer circle
            print('\ni[0] (x): ', i[0])
            print('\ni[1] (y): ', i[1])
            print('\ni[2] (r): ', i[2])

            if i[1] < i[2]:
                continue

            
            offset = 0#-10
            
            npImage=np.array(gray_enhanced_im)
            h,w=Image.fromarray(npImage).size    
            x0 = (i[0] - i[2]) * (1/scale_percent) - offset
            y0 = (i[1] - i[2]) * (1/scale_percent) - offset
            x1 = (i[0] + i[2]) * (1/scale_percent) + offset
            y1 = (i[1] + i[2]) * (1/scale_percent) + offset
            print(f'x0:{x0},\ny0:{y0},\nx1:{x1},\ny1:{y1}\n')

            # Create same size alpha layer with circle
            alpha = Image.new('L', (h,w) ,0)
            draw = ImageDraw.Draw(alpha)
            draw.pieslice([x0,y0,x1,y1],0,360,fill=255)
            # Convert alpha Image to numpy array
            npAlpha=np.array(alpha)

            # Add alpha layer to RGB
            npImage = np.dstack((npImage,npAlpha))
            output = Image.fromarray(npImage).crop((x0, y0, x1, y1))
            datas = output.getdata()

            pprint(f'Printing datas: {datas}\n')
            pprint(f'Printing datas[0]: {datas[0]}\n')
            pprint(f'Printing datas[1]: {datas[1]}\n')
            pprint(f'Printing datas[2]: {datas[2]}\n')
            pprint(f'Printing datas[3]: {datas[3]}\n')

            newData = []
            for item in datas:
                pprint(f'Item from datas: {item} from {datas}')
                if item[3] == 0:
                    newData.append((0, 0, 0, 0))
                else:
                    newData.append(item)

            
            output.putdata(newData)

            print('\n\npath_to save modded images: ' + output_dir + '\n\n')
            rgb_output = np.uint8(output.convert('RGB'))
            gray_output = rgb2gray(rgb_output)
            print(output_dir + '/' + img_name)
            cv.imwrite(output_dir + '/' + img_name, gray_output)

if os.path.exists(fn):
    print(f'working query: {fn} + **/wells/*/*.jpeg')
    img_list = glob(fn + '**/wells/*/*.jpeg')
    pprint(str(img_list))

    header = ['Experiment_start','Well', 'Time', 'Egg', 'Worm']

     # Gets CUR_DIR
    directory_in_str = pathlib.Path(__file__).parent.absolute()
    directory = os.fsencode(directory_in_str)
    modified_img_dir_path = os.path.join(os.fsdecode(directory), "Modded_IMGs")
    
    # define the access rights
    access_rights = 0o755

    init_dirs(modified_img_dir_path, access_rights)

    with open(fn + 'CeSAR_inference_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # BS/16-06-2022_155929/wells/A12/16-06-2022_15:59:29_A12_h7.jpeg
        pattern = r'.*\/(.*)\/wells\/(.*)\/(.*)'
        for img_path in img_list:
            re_match = re.search(pattern, img_path)

            exp = re_match.groups()[0]
            wellname = re_match.groups()[1]
            img_name = re_match.groups()[2]
            acquisiton_time = time.ctime(os.path.getmtime(img_path)) 
            datetime_object = datetime.strptime(acquisiton_time, '%a %b %d %H:%M:%S %Y')
            new_img_name = f'{wellname}_' + datetime_object.strftime("%d-%m-%Y_%Hh%Mm%Ss") + f'.jpeg' # 31-07-2013_03-12-12
            print(new_img_name)


            print(f'{exp}, {wellname}, {acquisiton_time}')

            modify_img(img_path, new_img_name, modified_img_dir_path)

            writer.writerow([exp, wellname, acquisiton_time, 0, 0])

        
else:
    print('INVALID PATH: Please enter a valid path as first argument.')
    print('Exiting now.')
    sys.exit()


    