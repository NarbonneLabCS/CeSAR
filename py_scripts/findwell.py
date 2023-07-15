#!/usr/local/bin/python3
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
import cv2 as cv
import argparse
import pathlib
import os
import matplotlib.pyplot as plt



def main():
    # Gets the name of the directory this file is in
    directory_in_str = pathlib.Path(__file__).parent.absolute()
    directory = os.fsencode(directory_in_str)
    
    nb_files = len([filename for filename in os.listdir(directory) if os.path.isfile(filename)])
    print("Nb of files :" + str(nb_files))

    # Iterate through the files of the folder
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(f"FILENAME: {filename}")
            img = cv.imread(filename)
            #img = cv.medianBlur(img,5)

            #im = Image.open(filename)

            #enhancer = ImageEnhance.Sharpness(im)

            #enhanced_im = enhancer.enhance(5.0)
            #enhanced_im.save("Enhanced_"+filename)   
            
            # dark = im.filter(ImageFilter.MinFilter(3))

            # im_autoC = ImageOps.autocontrast(enhanced_im, cutoff = 0.5)
            
            # im.save("ORI"+filename, quality=95)
            # enhanced_im.save("Enhanced_"+filename, quality=95)
            # im_autoC.save("E_AutoC_"+filename, quality=95)
            # dark.save("DARK"+filename, quality=95)

            
            

            # #pil_image = PIL.Image.open('image.jpg')
            # #opencvImage = cv.cvtColor(numpy.array(pil_image), cv.COLOR_RGB2BGR)
            
            #img = cv.cvtColor(np.array(enhanced_im), cv.COLOR_RGB2BGR)

            scale_percent = 20 # percent of original size to get 1264 px wide image to match the rest (non necessary)
            resized_w = int(4056 * scale_percent / 100)
            resized_h = int(3040 * scale_percent / 100)
            dim = (resized_w, resized_h)    
            resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
            gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

            minR = 275
            maxR = 300
            print(f"Looking for circle with radius {minR} < r < {maxR}")
            #circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,3000,param1=150,param2=0.9,minRadius=minR,maxRadius=maxR)
            circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,3000,param1=150,param2=0.9,minRadius=minR,maxRadius=maxR)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:

                    # draw the outer circle
                    print('\ni[0] (x): ', i[0])
                    print('\ni[1] (y): ', i[1])
                    print('\ni[2] (r): ', i[2])

                    if i[1] < i[2]:
                        continue

                    #cv.circle(resized,(i[0],i[1]),i[2],(0,255,0),3)
                    # draw the center of the circle
                    #cv.circle(resized,(i[0],i[1]),2,(0,0,255),3)
                    
                    offset = 0#-10
                    
                    npImage=np.array(img)
                    h,w=Image.fromarray(npImage).size    
                    x0 = (i[0] - i[2]) * 5 - offset
                    y0 = (i[1] - i[2]) * 5 - offset
                    x1 = (i[0] + i[2]) * 5 + offset
                    y1 = (i[1] + i[2]) * 5 + offset
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

                    newData = []
                    for item in datas:
                        if item[3] == 0:
                            newData.append((255, 0, 255, 0))
                        else:
                            newData.append(item)

                     
                    output.putdata(newData)

                    #dst = cv.cvtColor(np.array(output), cv.COLOR_RGB2BGR)
                    # denoising of image saving it into dst image 
                    #dst = cv.fastNlMeansDenoisingColored(np.uint8(output), None, 2, 2, 7, 21) 
                    #cv.imwrite('CnD_' + filename, dst)
                    rgb_output = np.uint8(output.convert('RGB'))
                    cv.imwrite('C_' + filename, rgb_output)
                    #rgb_output.save('C_' + filename, quality=90)

# If this script is called directly, executes the main function
if __name__ == '__main__':
    main()