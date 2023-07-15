#!/usr/bin/python3
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
import os
import math


def manipulation(src, name):
    # global thresholding
    ret1,th1 = cv.threshold(src,127,255,cv.THRESH_BINARY_INV)
    # Otsu's thresholding
    ret2,th2 = cv.threshold(src,127,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.bilateralFilter(src,3,3,3)#cv.GaussianBlur(src,(3,3),150)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    #inv = cv.bitwise_not(th3)
    erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    #opening_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
    #closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

    #dilation = cv.dilate(inv, dilation_kernel, iterations = 0)
    dilation = cv.dilate(th3, dilation_kernel, iterations = 2)
    erosion = cv.erode(dilation, erosion_kernel, iterations = 3)
    dilation = cv.dilate(erosion, dilation_kernel, iterations = 1)
    

    #opening = cv.morphologyEx(inv, cv.MORPH_OPEN, opening_kernel)
    #closing = cv.morphologyEx(inv, cv.MORPH_CLOSE, closing_kernel)
    
    filename = os.path.splitext(str(name))[0]
    
    cv.imshow('dilation', dilation)
    cv.waitKey(0)
    
    
    contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)#CHAIN_APPROX_SIMPLE)
    #cnt = contours[1]
    mask = np.zeros_like(dilation)
    # Draw skeleton of banana on the mask
    img = dilation.copy()
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    ret,img = cv.threshold(img,5,255,0)
    element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    done = False
    while(not done):
        eroded = cv.erode(img,element)
        temp = cv.dilate(eroded,element)
        temp = cv.subtract(img,temp)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy() 
        zeros = size - cv.countNonZero(img)
        if zeros==size: done = True
    kernel = np.ones((2,2), np.uint8)
    skel = cv.dilate(skel, kernel, iterations=1)
    skeleton_contours, _ = cv.findContours(skel, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_skeleton_contour = max(skeleton_contours, key=cv.contourArea)

    cv.imshow('img',img)
    cv.waitKey(0)   

    # Extend the skeleton past the edges of the banana
    points = []
    for point in largest_skeleton_contour: points.append(tuple(point[0]))
    x,y = zip(*points)
    z = np.polyfit(x,y,75)
    f = np.poly1d(z)
    x_new = np.linspace(0, img.shape[1],300)
    y_new = f(x_new)
    extension = list(zip(x_new, y_new))
    img = src.copy()
    for point in range(len(extension)-1):
        a = tuple(np.array(extension[point], int))
        b = tuple(np.array(extension[point+1], int))
        cv.line(img, a, b, (0,0,255), 1)
        cv.line(mask, a, b, 255, 1)   
    mask_px = np.count_nonzero(mask)

    cv.imshow('img',img)
    cv.waitKey(0)  

    print("Filename is: " + filename)

    #cv.imwrite('new_SKE_' + filename + '.jpg', largest_skeleton_contour)
    
    #return ske

### Main workflow
def main():

    # Read image
    imgAdultPre = cv.imread('CeSAR_infer/exp2/crops/Worm/PRCSD_B9_20-07-2022_19h39m14s_f81_2896px3.jpg',0)
    #imgL3Pre = cv.imread('Fin.png',0)
    #imgL1Pre = cv.imread('Macro.png',0)
    
    imgAdultPost = manipulation(imgAdultPre, 'CeSAR_infer/exp2/crops/Worm/PRCSD_B9_20-07-2022_19h39m14s_f81_2896px3.jpg')
    #imgL3Post = manipulation(imgL3Pre, 'Fin.png')
    #imgL1Post = manipulation(imgL1Pre, 'Macro.png')
    
    
    # plot all the images and their histograms
    #images = [imgAdultPre, 0, imgAdultPost,
    #          imgL3Pre, 0, imgL3Post,
    #          imgL1Pre, 0, imgL1Post]
    #titles = ['Original Image','Histogram','Otsu Thresholding',
    #          'Original Noisy Image','Histogram',"Otsu's Thresholding",
    #          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    # for i in range(3):
    #     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    #     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    #     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    #     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    # plt.subplot(3,3,1),plt.imshow(images[0],'gray')
    # plt.title(titles[0]), plt.xticks([]), plt.yticks([])
    # plt.subplot(3,3,2),plt.hist(images[0].ravel(),256)
    # plt.title(titles[1]), plt.xticks([]), plt.yticks([])
    # plt.subplot(3,3,3),plt.imshow(images[2],'gray')
    # plt.title(titles[2]), plt.xticks([]), plt.yticks([])
    # plt.show()
    # t = 200
    # ret,thresh1 = cv.threshold(img,t,255,cv.THRESH_BINARY)
    # ret,thresh2 = cv.threshold(img,t,255,cv.THRESH_BINARY_INV)
    # ret,thresh3 = cv.threshold(img,t,255,cv.THRESH_TRUNC)
    # ret,thresh4 = cv.threshold(img,t,255,cv.THRESH_TOZERO)
    # ret,thresh5 = cv.threshold(img,t,255,cv.THRESH_TOZERO_INV)
    # titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    # images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    # for i in range(6):
    #     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])
    # plt.show()

# If this script is called directly, executes the main function
if __name__ == '__main__':
    main()