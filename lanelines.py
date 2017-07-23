#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This program takes an image or video and finds the lane line on the road """


import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from moviepy.editor import VideoFileClip

import random


##################
### Parameters ###
##################

gaussian_kernel_size = 5

canny_low_threshold = 200
canny_high_threshold = 250

hough_rho = 1
hough_theta = np.pi/360
hough_threshold = 20
min_line_len = 20
max_line_gap = 5

w_alpha = 1
w_beta = 1


########################
### HELPER FUNCTIONS ###
########################

def preprocess_image(image):
    """ Extracts only white and yellow elements in the image """
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow_lower_bound = np.array([ 20,   50, 50])
    yellow_upper_bound = np.array([ 40, 255, 255])
    yellow_mask = cv2.inRange(img, yellow_lower_bound, yellow_upper_bound)
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_lower_bound = np.array([  0, 230,   0])
    white_upper_bound = np.array([179, 255, 255])
    white_mask = cv2.inRange(img, white_lower_bound, white_upper_bound)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    img = cv2.bitwise_and(image, image, mask = mask)
    return img


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Initialize Global Variables
right_m_past = 0.5
left_m_past = -0.5
right_b_past = 100
left_b_past = -100
image_counter = 0
def draw_lines(img, lines, color=[255, 0, 0], thickness=20):

    global right_m_past
    global left_m_past
    global right_b_past
    global left_b_past
    global image_counter

    min_slope = 0.5
    episilon = 0.0000000000001
    y_size = img.shape[0]
    x_size = img.shape[1]
    center_y = int(y_size/2)
    center_x = int(x_size/2)
    y_min = int(y_size*1)
    y_max = int(y_size*0.6)

    right_x = []
    left_x = []
    right_y = []
    left_y = []

    if lines is None or len(lines) == 0:
        print("No lines found in frame {}".format(image_counter))
        x_right_2 = int((y_max-right_b_past)/right_m_past)
        x_right_1 = int((y_min-right_b_past)/right_m_past)
        x_left_2 = int((y_max-left_b_past)/left_m_past)
        x_left_1 = int((y_min-left_b_past)/left_m_past)   
        cv2.line(img, (x_right_1, y_min), (x_right_2, y_max), color, thickness)
        cv2.line(img, (x_left_1, y_min), (x_left_2, y_max), color, thickness)
        return

    for line in lines:
        for x1,y1,x2,y2 in line:

            # slope (m) and intercept (b)
            m = (float(y2)-float(y1))/(float(x2)-float(x1) + episilon)

            if np.absolute(m) > min_slope:
                if m > 0:
                    right_x.append(x1)
                    right_x.append(x2)
                    right_y.append(y1)
                    right_y.append(y2)
                elif m < 0:
                    left_x.append(x1)
                    left_x.append(x2)
                    left_y.append(y1)
                    left_y.append(y2)

    # Polynomial fit if possible
    if len(right_x) > 0:
        right_m, right_b = np.polyfit(right_x, right_y, 1)
    else:
        right_m = right_m_past
        right_b = right_b_past
    if len(left_x) > 0:
        left_m, left_b = np.polyfit(left_x, left_y, 1)
    else:
        left_m = left_m_past
        left_b = left_b_past

    if image_counter > 0:
        right_m = right_m*0.55 + right_m_past*0.45
        right_b = right_b*0.55 + right_b_past*0.45
        left_m = left_m*0.55 + left_m_past*0.45
        left_b = left_b*0.55 + left_b_past*0.45


    # Update State
    right_m_past = right_m
    right_b_past = right_b
    left_m_past = left_m
    left_b_past = left_b
    image_counter += 1


    x_right_2 = int((y_max-right_b)/right_m)
    x_right_1 = int((y_min-right_b)/right_m)
    x_left_2 = int((y_max-left_b)/left_m)
    x_left_1 = int((y_min-left_b)/left_m)   

    cv2.line(img, (x_right_1, y_min), (x_right_2, y_max), color, thickness)
    cv2.line(img, (x_left_1, y_min), (x_left_2, y_max), color, thickness)


def draw_segments(img, lines, color=[0, 255, 0], thickness=10):
    if lines is None or len(lines) == 0:
        return
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    draw_segments(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)



########################
### Custom Functions ###
########################

# Core Functions that use helper function and more

def process_image(img):
    """This function takes an image, finds the lane lines, and draws them.
    
    Arguments:
        img = RGB image
    Output:
        result = RGB image where lines are drawn on lanes
    """

    #cv2.imwrite("process_images/before.jpg",cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))

    pre = preprocess_image(img)
    #cv2.imwrite("process_images/preprocess.jpg",cv2.cvtColor(pre.copy(), cv2.COLOR_RGB2BGR))

    gray = grayscale(pre)
    #cv2.imwrite("process_images/gray.jpg",gray)

    # we further blur the image
    blurred = gaussian_blur(gray, kernel_size = gaussian_kernel_size)

    #cv2.imwrite("process_images/blur.jpg",blurred)
    # we detect edges with canny method
    edges = canny(blurred, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold)

    #cv2.imwrite("process_images/canny.jpg",edges)
    # Vertices are defined relative to the img shape
    y_size = img.shape[0]
    x_size = img.shape[1]
    vertices = np.array([[(x_size*0.5,y_size*0.6),(x_size*0.5, y_size*0.6), (x_size*0.1, y_size*0.9), (x_size*0.9,y_size*0.9)]], dtype=np.int32)

    # we only consider edges in region of interest (roi)
    masked_edges = region_of_interest(edges, vertices)
    #cv2.imwrite("process_images/mask.jpg",masked_edges)

    # we use hough method to calculate lines from roi
    lines = hough_lines(masked_edges, hough_rho, hough_theta, hough_threshold, min_line_len, max_line_gap)
    result = weighted_img(lines, img, w_alpha, w_beta, 0)

    #cv2.imwrite("process_images/result.jpg",cv2.cvtColor(result.copy(), cv2.COLOR_RGB2BGR))

    return result

###########################################################
### Higer Abstraction Layers for command line interface ###
###########################################################

def find_lanes_video(filepath, save_result = True):
    """This method finds laneslines on a video file and displays it on the screen.
    
    Arguments:
        filepath = Path to video file
        save_result = Switch to indicate whether you want to save result
    """
    clip = VideoFileClip(filepath)
    

    result_clip = clip.fl_image(process_image)

    # Save Result 
    if save_result:
        pos = filepath.rfind('/')
        savepath = filepath[:pos] + '_output' + filepath[pos:]
        print("\n Saving video at: {}".format(savepath))
        result_clip.write_videofile(savepath, audio=False)




def find_lanes_image(filepath, save_result = True):
    """This method finds laneslines on a single image file and displays it on the screen.
    
    Note: Similar to 
    Arguments:
        filepath = Path to image file
        save_result = Switch to indicate whether you want to save result
    """

    # Reading in an image. Note: mpimg.imread outputs in RGB format.
    image = mpimg.imread(filepath)

    # Printing image info, and displaying it
    print('Image at {} is now: {} with dimensions: {}'.format(filepath,type(image),image.shape))
    plt.imshow(image)
    plt.show()

    result = process_image(image)
    plt.imshow(result)
    plt.show()

    # Save Result in cv2 format (BGR)
    if save_result:
        pos = filepath.rfind('/')
        savepath = filepath[:pos] + '_output' + filepath[pos:]
        result_BGR = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        print("\n Saving image at: {}".format(savepath))
        cv2.imwrite(savepath,result_BGR)


def parse_args():
    """Supplies arguments to program"""

    parser = argparse.ArgumentParser()
    # Set video Mode
    parser.add_argument('--video-mode', dest='video_mode', action='store_true',default=False)
    # Set image Mode
    parser.add_argument('--image-mode', dest='image_mode', action='store_true',default=False)
    # Set video path if video Mode
    parser.add_argument('--video-path', dest='video_path', type=str, default="test_videos/challenge.mp4")
    # Set image path if image mode
    parser.add_argument('--image-path', dest='image_path', type=str, default="test_images/whiteCarLaneSwitch.jpg")
    # Save result mode
    parser.add_argument('--save-result', dest='save_result', action='store_true',default=True)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    if args.video_mode:
        find_lanes_video(args.video_path,save_result=args.save_result)
    elif args.image_mode:
        find_lanes_image(args.image_path,save_result=args.save_result)
    else:
        print('Enter: python lanelines.py --video-mode --video-path [PATH_TO_VIDEO] , or python lanelines.py --image-mode --image-path [PATH_TO_IMAGE]')