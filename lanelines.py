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


####################################
### HELPER FUNCTIONS FROM LESSON ###
####################################

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold=50, high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size=5):
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho=1, theta=np.pi/180, threshold=2, min_line_len=5, max_line_gap=1):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha=1, beta=1., gamma=0.):
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

    gray = grayscale(img)
    # we further blur the image
    blur_gray = gaussian_blur(gray, kernel_size = 3)
    # we detect edges with canny method
    edges = canny(blur_gray, low_threshold=50, high_threshold=150)

    # Vertices are defined relative to the img shape
    y_size = img.shape[0]
    x_size = img.shape[1]
    vertices = np.array([[(x_size*0.5,y_size*0.55),(x_size*0.5, y_size*0.55), (0, y_size), (x_size,y_size)]], dtype=np.int32)

    # we only consider edges in region of interest (roi)
    masked_edges = region_of_interest(edges, vertices)
    # we use hough method to calculate lines from roi
    lines = hough_lines(masked_edges, rho=1, theta=np.pi/180, threshold=15, min_line_len=60, max_line_gap=30)
    result = weighted_img(lines, img, alpha=1, beta=1., gamma=0.)

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
    parser.add_argument('--image-path', dest='image_path', type=str, default="test_images/solidWhiteCurve.jpg")
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