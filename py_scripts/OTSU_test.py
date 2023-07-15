#!/usr/bin/python3
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import sys, traceback
import cv2 as cv
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage import data
import sknw
from fil_finder import FilFinder2D
import astropy.units as u
from pprint import pprint


def manipulation(img):
    # global thresholding
    ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    # Otsu's thresholding
    ret2,th2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.bilateralFilter(img,21,75,75)#cv.GaussianBlur(img,(3,3),150)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    #inv = cv.bitwise_not(th3)
    erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    #opening_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
    #closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

    #dilation = cv.dilate(inv, dilation_kernel, iterations = 0)
    erosion = cv.erode(th3, erosion_kernel, iterations = 4)
    dilation = cv.dilate(erosion, dilation_kernel, iterations = 4)

    #opening = cv.morphologyEx(inv, cv.MORPH_OPEN, opening_kernel)
    #closing = cv.morphologyEx(inv, cv.MORPH_CLOSE, closing_kernel)
    
    
    
    contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)#CHAIN_APPROX_SIMPLE)
    #skeleton = skeletonize(~dilation).astype(np.uint16)
    #print("Skeleton matrix x len: " + str(len(skeleton[[0]])))
    #print("Skeleton matrix y len: " + str(len(skeleton[0])))
    #print("type of (Skeleton x) is : " + str(type(str(skeleton[[0]]))))
    #toSplit = str(skeleton[[0]]).split()
    #print("toSplit: " + str(toSplit))
    #print("Length of toSplit: " + str(len(toSplit)))
    #print("Skeleton matrix y len: " + str(len(skeleton[0])))
    #for i in range(skeleton[[0]].length()):
        #DONOTHING
    ske_arr = (skeletonize(dilation//255) * 255).astype(np.uint8)
    pprint("skeleton is type: ")
    pprint(type(ske_arr))

    x_c = 0
    y_c = 0
    for x in ske_arr:
        y_c = 0
        for y in x:
            if y != 0: 
                print(f"(x = {x_c}, y = {y_c}) _ non-zero value: ", y)

                neighb_cnt = 0
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        #print("ske_arr[y+j,x+i] :", ske_arr[y+j,x+i])
                        onCenter = False
                        if i == 0 and j == 0:
                            onCenter = True
                        if np.any(ske_arr[y+j,x+i] != 0) and onCenter == False:
                            neighb_cnt += 1

                if neighb_cnt == 1:
                    print("THIS IS AND ENDPOINT!")

            y_c += 1
        x_c += 1
    #ske_0 = ske_arr[401]

    #print("SKE [0]: ") 
    #print(ske_0)
    ske = (skeletonize(dilation//255) * 255).astype(np.uint8)
    cv.drawContours(ske, contours, -1, color=(255,255,255), thickness=cv.FILLED)
    cv.drawContours(img, contours, -1, color=(255,255,255), thickness=cv.FILLED)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    ske = (skeletonize(th3//255) * 255).astype(np.uint8)
    cv.drawContours(ske, contours, -1, color=(255,255,255), thickness=1)
    cv.drawContours(img, contours, -1, color=(255,255,255), thickness=1)
    #fil = FilFinder2D(ske, distance=250 * u.pc, mask=ske)
    #fil.preprocess_image(flatten_percent=85)
    #fil.create_mask(border_masking=True, verbose=False,
    #use_existing_mask=True)
    #fil.medskel(verbose=False)
    #fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

    # Show the longest path
    #plt.imshow(fil.skeleton, cmap='gray')
    #plt.contour(fil.skeleton_longpath, colors='r')
    #plt.axis('off')
    #plt.show()

    #blur = cv.GaussianBlur(img,(3,3),0)
    #ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    return ske

### Main workflow
def main():

    # Read image
    filename = 'C4__04-06-21_08h55_0.jpg'
   # img = cv.imread(filename,0) #opencv
    PIimg = Image.open(filename) #PIL
    
    dark = PIimg.filter(ImageFilter.MinFilter(11))
    dark.save("temp.jpg")

    OPimg = cv.imread("temp.jpg",0)#cv.cvtColor(np.asarray(dark), cv.COLOR_RGB2BGR)

    blur = cv.bilateralFilter(OPimg, 3, 75, 75)#cv.GaussianBlur(img,(3,3),150)
    ret3,th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


    erosion_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(41,41))
    dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(41,41))


    opening_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(200,200))
    closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(50,50))

    closing = cv.morphologyEx(th, cv.MORPH_CLOSE, closing_kernel)
    #opening = cv.morphologyEx(closing, cv.MORPH_OPEN, opening_kernel)
    
    src = closing.copy()   #First create a copy
    mask = np.zeros([src.shape[0]+2, src.shape[1]+2, 1], np.uint8)   #Create a mask based on the shape of the copy. Note that the length and width must be +2, and the type can only be uint8
    cv.floodFill(src, mask, (2028, 1520), (255, 255, 255), (50,50,50), (50,50,50), cv.FLOODFILL_FIXED_RANGE)
    cv.floodFill(src, mask, (1, 1), (0, 0, 0), (50,50,50), (50,50,50), cv.FLOODFILL_FIXED_RANGE)
    cv.floodFill(src, mask, (4055, 1), (0, 0, 0), (50,50,50), (50,50,50), cv.FLOODFILL_FIXED_RANGE)
    cv.floodFill(src, mask, (1, 3039), (0, 0, 0), (50,50,50), (50,50,50), cv.FLOODFILL_FIXED_RANGE)
    cv.floodFill(src, mask, (4055, 3039), (0, 0, 0), (50,50,50), (50,50,50), cv.FLOODFILL_FIXED_RANGE)
    #(60,60) represents the starting point; (0,0,255) represents the fill color; loDiff=(50,50,50) represents only the points that are smaller than the fill color, and upDiff is the same
    cv.imshow('flood_fill', src)
    cv.imshow('mask', mask)

    src2 = cv.erode(src, erosion_kernel, iterations = 10)
    src2 = cv.dilate(src2, dilation_kernel, iterations = 10)
    #src2 = cv.morphologyEx(src, cv.MORPH_OPEN, opening_kernel)

    contours, hierarchy = cv.findContours(src2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = mask.copy()
    cv.drawContours(cnt, contours, -1, (255, 255, 255), 3)

    cv.imwrite('OTSU_mask' + filename, mask)
    cv.imwrite('OTSU_src' + filename, src)
    cv.imwrite('OTSU_src2' + filename, src2)
    cv.imwrite('OTSU_cnt' + filename, cnt)
    cv.imwrite('OTSU_pp' + filename, closing)
    cv.imwrite('OTSU_ori' + filename, th)


    # t = 200
    # ret,thresh1 = cv.threshold(img,t,255,cv.THRESH_BINARY)


# If this script is called directly, executes the main function
if __name__ == '__main__':
    main()