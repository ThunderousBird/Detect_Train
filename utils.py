# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:37:10 2017

@author: yang
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#get all image in the given directory persume that this directory only contain image files
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [dirname+'/'+img_name for img_name in img_names]
    imgs = [cv2.imread(path) for path in img_paths]
    return imgs

#function take the chess board image and return the object points and image points
def calibrate(images,grid=(9,6)):
    object_points=[]
    img_points = []
    for img in images:
        object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
        object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        if ret:
            object_points.append(object_point)
            img_points.append(corners)
    return object_points,img_points

# def get_M_Minv():
#     src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
#     dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
#     M = cv2.getPerspectiveTransform(src, dst)
#     Minv = cv2.getPerspectiveTransform(dst,src)
#     return M,Minv

def get_M_Minv(img_shape):
    # img_height, img_width = img_shape[:2] - 1
    img_height, img_width, _ = img_shape.shape
    # print(img_height, img_width)
    # print(img_height, img_width)
    #src = np.float32([[(720, 3000), (2200, 1200), (2200, 3000), (2400, 1200)]])
    #dst = np.float32([[(1000, 3000), (1200, 0), (2800, 3000), (3400, 0)]])
    #src = np.float32([[(550, 1080), (780, 140), (1400, 1080), (900, 140)]])
    #dst = np.float32([[(200, 1080), (100, 0), (1500, 1080), (1600, 0)]])
    # src = np.float32([[(600, 1080), (850, 300), (1600, 1080), (1000, 300)]])
    # dst = np.float32([[(500, 1080), (0, 0), (1500, 1080), (1300, 0)]])
    src = np.float32([[(0, img_height), (0, 0), (img_width, img_height), (img_width, 0)]])
    dst = np.float32([[(0, img_height), (0, 0), (img_width, img_height), (img_width, 0)]])
    # src = np.float32([(0, img_height),(0, 0),(img_width, img_height),(img_width, 0)])
    # dst = np.float32([
    #     (0, img_height),        # 左下角
    #     (0, 0),                 # 左上角
    #     (img_width, img_height),# 右下角
    #     (img_width, 0)         # 右上角
    # ])
    #src = np.float32([[(0, 1920), (510, 500), (1000, 1920), (600, 500)]])
    #dst = np.float32([[(300, 1920), (150, 0), (850, 1920), (900, 0)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv
    
#function takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img,channel='s',thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output

def find_line(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds

def find_line_by_previous(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**3) + left_fit[1]*(nonzeroy**2) +
    left_fit[2]*nonzeroy +left_fit[3] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**3) +
    left_fit[1]*(nonzeroy**2) + left_fit[2]*nonzeroy + left_fit[3] + margin)))
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**3) + right_fit[1]*(nonzeroy**2) +
    right_fit[2]*nonzeroy + right_fit[3] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**3) +
    right_fit[1]*(nonzeroy**2) + right_fit[2]*nonzeroy + right_fit[3] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)
    return left_fit, right_fit, left_lane_inds, right_lane_inds

def expand(img):
    image = img.copy()  # 避免直接修改输入图像
    img_height, img_width, _ = img.shape
    _, green, _ = cv2.split(image)
    s = np.sum(green, axis=1)
    a = range(img_height)  # 假设图像高度为 1080

    for i in reversed(a):
        if s[i] < 10:  # 如果当前行的绿色像素总和小于阈值，停止处理
            break

        # 找到最左侧和最右侧绿色像素的位置
        j = next((x for x in range(img_width) if green[i][x] == 255), None)  # 最小 x 值
        k = next((x for x in reversed(range(img_width)) if green[i][x] == 255), None)  # 最大 x 值

        if j is None or k is None:  # 如果没有找到绿色像素，跳过该行
            continue

        # 根据绿色像素总和计算扩展范围
        expand_range = int(s[i] / 255)

        # 向左扩展
        for l in range(expand_range):
            if 0 <= j - l < img_width:  # 确保索引在合法范围内
                image[i, j - l, 2] = 255

        # 向右扩展
        for l in range(expand_range):
            if 0 <= k + l < img_width:  # 确保索引在合法范围内
                image[i, k + l, 2] = 255

    return image


# def expand(img):
#     image = img
#     _,green,_ = cv2.split(image)
#     s = np.sum(green,axis=1)
#     a = range(1080)
#     for i in reversed(a):
#         if s[i] <200:
#             break
#         for j in range(1920):  # min x
#             if green[i][j] == 255:
#                 break
#         for k in reversed(range(1920)): #max x
#             if green[i][k] == 255:
#                 break
#         for l in range(int(s[i]/255)): # s[i]/255  the number
#             image[i,j-l,2] = 255
#         for l in range(int(s[i]/255)):
#             image[i,k+l,2] = 255

#     return image

# def expand(img):
#     image = img
#     _,green,_ = cv2.split(image)
#     s = np.sum(green,axis=1)
#     a = range(img.shape[0])
#     for i in reversed(a):
#         if s[i] <200:
#             break
#         for j in range(img.shape[1]):  # min x
#             if green[i][j] == 255:
#                 break
#         for k in reversed(range(img.shape[1])): #max x
#             if green[i][k] == 255:
#                 break
#         for l in range(int(s[i]/255)): # s[i]/255  the number
#             if j>l:
#                 image[i,j-l,2] = 255
#         for l in range(int(s[i]/255)):
#             if (k+l)<1080:
#                 image[i,k+l,2] = 255
#
#     return image

def draw_area(undist,binary_warped,Minv,left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fitx = left_fit[0]*ploty**3 + left_fit[1]*ploty**2 + left_fit[2]*ploty+left_fit[3]
    right_fitx = right_fit[0]*ploty**3 + right_fit[1]*ploty**2 + right_fit[2]*ploty+right_fit[3]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    undist = np.array(undist)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    newwarp = expand(newwarp)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result,newwarp

def calculate_curv_and_pos(binary_warped,left_fit, right_fit):
    # Define y-value where we want radius of curvature
    img_height, img_width = binary_warped.shape
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    curvature = ((left_curverad + right_curverad) / 2)
    #print(curvature)
    lane_width = np.absolute(leftx[img_height - 1] - rightx[img_height - 1])
    lane_xm_per_pix = 3.7 / lane_width
    veh_pos = (((leftx[img_height - 1] + rightx[img_height - 1]) * lane_xm_per_pix) / 2.)
    cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = cen_pos - veh_pos
    return curvature,distance_from_center

def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60])
    upper = np.array([38,174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    
    return mask

def select_white(image):
    lower = np.array([170,170,170])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    
    return mask

def draw_values(img,curvature,distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm"%(round(curvature))
    
    if distance_from_center>0:
        pos_flag = 'right'
    else:
        pos_flag= 'left'
        
    cv2.putText(img,radius_text,(100,100), font, 1,(255,255,255),2)
    #center_text = "Vehicle is %.3fm %s of center"%(abs(distance_from_center),pos_flag)
    #cv2.putText(img,center_text,(100,150), font, 1,(255,255,255),2)
    return img