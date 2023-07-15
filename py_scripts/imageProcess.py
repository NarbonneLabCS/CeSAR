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

def getSkeletonPts(src_list, extremities):
    ske_tuple_pts = []

    if (len(extremities)>2):
        print("MORE THAN ONE EXTREMITIES!")
        return ske_tuple_pts

    x_i, y_i = extremities[0]
    x_tp, y_tp = x_i, y_i
    x_tc, y_tc = x_i, y_i
    x_tn, y_tn = x_i, y_i
    x_f, y_f = extremities[1]

    ske_tuple_pts.append((x_i,y_i))
    #print(src_list)
    while x_tc != x_f or y_tc != y_f:
        print(f"TP: {x_tp}, {y_tp} - --- - TC: {x_tc}, {y_tc} - --- - TN: {x_tn}, {y_tn}")
        nextIsFound = False
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if nextIsFound == True:
                    break
                
                x_tn = x_tc + i
                y_tn = y_tc + j

                print(f"Tentative next pt is for (i{i}, j{j}):")
                print(src_list[y_tn][x_tn])
                print()
                isCenter = False
                if(i == j == 0):
                    print("This is the center")
                    isCenter = True

                isPrev = False
                if((y_tn,x_tn) == (y_tp,x_tp)):
                    print("This is the previous pt")
                    isPrev = True


                if (src_list[y_tn][x_tn] > 0) and isCenter == False and isPrev == False:
                    print("This pt should be the next pt")
                    ske_tuple_pts.append((x_tn,y_tn))
                    nextIsFound = True

                    
        if nextIsFound == True:
            x_tp = x_tc
            y_tp = y_tc

            x_tc = x_tn
            y_tc = y_tn
        else:
            print("ERROR. NEXT pt HAS NOT BEEN FOUND.")
            return ske_tuple_pts

        

    return ske_tuple_pts

def extendSke(ske_list,cnt_list):
    extremities = [ske_list[0],ske_list[len(ske_list)]]
    pts_to_add_front = []
    pts_to_add_back = []

    new_list = ske_list

    for i in range(0,5):
        pts_to_add_front.append(ske_list[i])
        pts_to_add_back.append(ske_list[len(ske_list) - i])

    hasTouchedCnt = False

def manipulation(img, name):
    # global thresholding
    ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    # Otsu's thresholding
    ret2,th2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.bilateralFilter(img,3,3,3)#cv.GaussianBlur(src,(3,3),150)
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
    
    
    
    

    #skeleton = skeletonize(~dilation).astype(np.uint16)
    

    #toSplit = str(skeleton[[0]]).split()
    #print("toSplit: " + str(toSplit))
    #print("Length of toSplit: " + str(len(toSplit)))


    #print("Skeleton matrix y len: " + str(len(skeleton[0])))
    #for i in range(skeleton[[0]].length()):
        #DONOTHING

    #ske_arr = (skeletonize(dilation//255) * 255).astype(np.uint8)
    #pprint("skeleton is type: ")
    #pprint(type(ske_arr))
    #ero=cv.dilate(dilation, dilation_kernel, iterations = 3)
    contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_TC89_KCOS)#)
    pprint("Contours ARE:")
    pprint(contours)
    filename = os.path.splitext(str(name))[0]

    #black_canvas = np.zeros([100,100,3], dtype=np.uint8)
    ske = (skeletonize(dilation//255) * 255).astype(np.uint8)
    #cv.drawContours(ske, contours, -1, color=(255,255,255), thickness=2)

    #ske_0 = ske_arr[401]

    #print("SKE [0]: ") 
    #print(ske_0)

    
    # print("ske for " + filename)
    # pprint(ske)
    # print("Skeleton matrix x len: " + str(len(ske[[0]])))
    # print("Skeleton matrix y len: " + str(len(ske[0])))
    # print("type of (Skeleton x) is : " + str(type(str(ske[[0]]))))
    # toSplit = str(ske[[0]]).split()
    # print("toSplit: " + str(toSplit))
    # print("Length of toSplit: " + str(len(toSplit)))
    # print("Skeleton matrix y len: " + str(len(ske[0])))
    pprint("skeleton is type: ")
    pprint(type(ske)) #numpy.nd.array
    ske_ravel = ske.ravel()
    pprint("ske_ravel is type: ")
    pprint(type(ske_ravel)) #numpy.nd.array
    list(ske)
    ske_list = ske.tolist()
    #print(ske_list)
    #pprint(type(ske_list))
    

    endpts = []

    for y in range(1,len(ske_list)-1):
        for x in range(1,len(ske_list[0])-1):      
            if ske_list[y][x] != 0: 
                #print(f"(x = {x}, y = {y}) _ non-zero value: ")
                #print(ske_list[y][x])
                neighb_cnt = 0
                for i in range(-1, 2, 1):
                    for j in range(-1, 2, 1):
                        #print("X length" + str(len(ske_list[0])))
                        #print("Y length" + str(len(ske_list)))
                        #print(f"ske_list[{y+j}][{x+i}]:")
                        #print(ske_list[int(y+j)][int(x+i)])

                        
                        if ske_list[int(y+j)][int(x+i)] != 0: #and onCenter == False:
                            neighb_cnt += 1

                if neighb_cnt == 2:
                    print("---------------------")
                    print()
                    print("THIS IS AND ENDPOINT!")
                    print()
                    print("---------------------")
                    endpts.append((x,y))

                    for i in range(-1, 2, 1):
                        for j in range(-1, 2, 1):
                            print(f"ske_list[{y+j}][{x+i}]:")
                            print(ske_list[int(y+j)][int(x+i)])

    print(f"My endpoints for {filename} are:")
    pprint(endpts)
    # print("-+-+--+-+-+-+-+-+-+-+-+-+-+-+-+-")
    # print("ske as LIST for " + filename)
    # print(ske)
    # print("Skeleton matrix x len: " + str(len(ske[0])))
    # print("Skeleton matrix y len: " + str(len(ske)))
    #skeLIST = getSkeletonPts(ske_list, endpts)
    #print("skeLIST!")
    #print(skeLIST)


    #cv.drawContours(ske, contours, -1, color=(255,255,255), thickness=1)
    #contours_ori = img.copy()
    
    #ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    #ske = (skeletonize(dilation//255) * 255).astype(np.uint8)
    #dil_np = np.asarray(dilation)
    #formatted_dilation = ((dilation//255) * 255).astype(np.uint8)
    #ske = (skeletonize(dil_np))

    
    #black_canvas[:,0.(0, 0, 0)
    #cv.drawContours(ske, black_canvas, -1, color=(255,255,255), thickness=1)
    #cv.drawContours(contours_ori, contours, -1, color=(255,255,255), thickness=1)


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


    print("Filename is: " + filename)

    #cv.imwrite('ORI_CNT_' + filename + '.jpg', contours_ori)
    #cv.imwrite('SKE_' + filename + '.jpg', ske)
    #cv.imshow('ORI_CNT', contours_ori)
    cv.imshow('SKE_', ske)
    cv.waitKey(0)
    return ske

### Main workflow
def main():

    # Read image
    imgAdultPre = cv.imread('CeSAR_infer/exp2/crops/Worm/PRCSD_B9_20-07-2022_19h39m14s_f81_2896px3.jpg', 0)
    imgAdultPost = manipulation(imgAdultPre, 'CeSAR_infer/exp2/crops/Worm/PRCSD_B9_20-07-2022_19h39m14s_f81_2896px3.jpg')
    
    
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