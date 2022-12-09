#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[2]:


#reading in an image
# image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
# print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:


import math

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

# def region_of_interest(img, vertices):
#     """
#     Applies an image mask.
    
#     Only keeps the region of the image defined by the polygon
#     formed from `vertices`. The rest of the image is set to black.
#     `vertices` should be a numpy array of integer points.
#     """
#     #defining a blank mask to start with
#     mask = np.zeros_like(img)   
    
#     #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
#     if len(img.shape) > 2:
#         channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
#         ignore_mask_color = (255,) * channel_count
#     else:
#         ignore_mask_color = 255
        
#     #filling pixels inside the polygon defined by "vertices" with the fill color    
#     cv2.fillPoly(mask, vertices, ignore_mask_color)
    
#     #returning the image only where mask pixels are nonzero
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image
#MODIFIED
def region_of_interest(img, vertices,vertices2):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
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
    cv2.fillPoly(mask, vertices2, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
#MODIFIED
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
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
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)      
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             #compute the line slope
#             #equation of line is y = m * x + b
#             if(x2!=x1):
#                 m = (y2-y1)/(x2-x1)
#                 if(abs(m)>0.2):
#                     cv2.line(img, (x1, y1), (x2, y2), color, thickness) 
#             else:
#                 m=0
    
    #get image shape, this is used for the extrapolation up to the region of interest
    imshape = img.shape
    #initialize empty lists
    left_lines = []
    right_lines = []
    left_lines_aligned = []
    right_lines_aligned = []
    left_m = []
    left_b = []
    right_m = []
    right_b = []

    #loop over the lines and sort them in left and right lists based on the slope
#     print(lines)
    if lines is not None and len(lines) !=0:
        for line in lines:
            if line is not None and len(line) !=0:
                for x1,y1,x2,y2 in line:
                    #compute the line slope
                    #equation of line is y = m * x + b
                    if(abs(x2-x1)>0.01):
                        m = (y2-y1)/(x2-x1)
                    else:
                        m=0
                    b = y1 - (m * x1)
                    #left line
                    if(np.isnan(b)):
                        print(m,b)
                    if(abs(m)>0.1): 
                        
#                         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], thickness) 
                        
#                         print(x1,y1,m)
                        if m < 0 and (x1<1000):
#                             cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], thickness)
                            left_lines.append((m,b)) 
                        #right line    
                        if m > 0 and x1>=1000:
#                           cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], thickness)                       
                            right_lines.append((m,b))            
    #calculate the average and standard deviation for the left lines' slope            
    left_m = [line[0] for line in left_lines]
#     print(left_m,len(left_m)!=0 )
    
    left_m_avg = np.nanmean(left_m) if len(left_m)!=0 else 0;
    left_m_std = np.std(left_m) if len(left_m)!=0 else 0;
#     print(left_m_avg)
    #only keep lines that are close to the average slope
    for line in left_lines:
        if abs(line[0] - left_m_avg) < left_m_std:
            if len(line)!=0 and line is not None:
#                 for m1,b1 in line:
                    m1=line[0]
                    b1=line[1]
#                     cv2.line(img, (int((b1 - imshape[0]) / (-1.00 * m1)), int(imshape[0])), (int((b1 - 6*imshape[0]/10) / (-1.00 * m1)), int(6*imshape[0]/10)), [255, 255, 0], thickness) 
            left_lines_aligned.append(line)
    #compute the average slope and intercept of the aligned left lines
    if len(left_lines_aligned) > 0:
        left_m = [line[0] for line in left_lines_aligned]
        ml = np.nanmean(left_m) if len(left_m)!=0 else 0;
        left_b = [line[1] for line in left_lines_aligned]
        bl = np.nanmean(left_b) if len(left_b)!=0 else imshape[0];
    else:
        ml = left_m_avg
        left_b = [line[1] for line in left_lines]
        bl = np.nanmean(left_b) if len(left_b)!=0 else imshape[0];
    
    #similar logic for the right lines as well
    right_m = [line[0] for line in right_lines]
    right_m_avg = np.nanmean(right_m) if len(right_m)!=0 else 0;
    right_m_std = np.std(right_m) if len(right_m)!=0 else 0;
   
    for line in right_lines:
        if abs(line[0] - right_m_avg) < right_m_std:
            if len(line)!=0 and line  is not None:
#                 for m1,b1 in line:
                    m1=line[0]
                    b1=line[1]
#                     cv2.line(img, (int((b1 - imshape[0]) / (-1.00 * m1)), int(imshape[0])), (int((b1 - 6*imshape[0]/10) / (-1.00 * m1)), int(6*imshape[0]/10)), [255, 255, 0], thickness) 
            right_lines_aligned.append(line)            
    if len(right_lines_aligned) > 0:
        right_m = [line[0] for line in right_lines_aligned]
        mr = np.nanmean(right_m) if len(right_m)!=0 else 0;
        right_b = [line[1] for line in right_lines_aligned]
        br = np.nanmean(right_b) if len(right_b)!=0 else imshape[0];
    else:
        mr = right_m_avg
        right_b = [line[1] for line in right_lines]
        br = np.nanmean(right_b) if len(right_b)!=0 else imshape[0];

    #use the previous cycle lines coeficients and smoothen lines over time
    smooth_fact = 0.8
    #only consider computed slope if angled enough
    if (abs(ml) < 1000):
        if (previous_lines[0] != 0):
            ml = previous_lines[0]*smooth_fact + ml*(1-smooth_fact)
            bl = previous_lines[1]*smooth_fact + bl*(1-smooth_fact)
    elif (previous_lines[0] != 0):
        ml = previous_lines[0]
        bl = previous_lines[1]
        
    if (abs(mr) < 1000):      
        if (previous_lines[2] != 0):
            mr = previous_lines[2]*smooth_fact + mr*(1-smooth_fact)
            br = previous_lines[3]*smooth_fact + br*(1-smooth_fact)
    elif (previous_lines[2] != 0):
        mr = previous_lines[2]
        br = previous_lines[3]
            
            
    #interpolate the resulting average line to intersect the edges of the region of interest
    #the two edges consider are y = 6*imshape[0]/10 (the middle edge)
    #                           y = imshape[0] (the bottom edge)
    if(ml==0):
        ml=0.01
    if(mr==0):
        mr=0.01
    if(math.isnan(((bl - imshape[0]) / (-1.00 * ml)))==False):
        x1l = ((bl - imshape[0]) / (-1.00 * ml))
    else:
        x1l=int(-1* imshape[0])
#     print(type(x1l),type(int(x1l.item())))
    y1l = imshape[0]   
    if(math.isnan(((bl - 6*imshape[0]/10) / (-1.00 * ml)))==False):
        x2l = int((bl - 6*imshape[0]/10) / (-1.00 * ml))
    else:
        x2l=int(bl-6*imshape[0]/10)
#     x2l = ((bl - 6*imshape[0]/10) / (-1.00 * ml)).astype(int)
    y2l = int(6*imshape[0]/10)      
    if(math.isnan(((br - 6*imshape[0]/10) / (-1.00 * mr)))==False):
        x1r = int((br - 6*imshape[0]/10) / (-1.00 * mr))
    else:
        x1r=int(br-6* imshape[0]/10)    
#     x1r = ((br - 6*imshape[0]/10) / (-1.00 * mr)).astype(int)
    y1r = int(6*imshape[0]/10) 
    if(math.isnan(((br - imshape[0]) / (-1.00 * mr)))==False):
        x2r = int((br - imshape[0]) / (-1.00 * mr))
    else:
        x2r=int(br-1* imshape[0])
#     x2r = ((br - imshape[0]) / (-1.00 * mr)).astype(int)
#     print(type(int(-1*imshape[0])),imshape[0])
    y2r = imshape[0]      
    lane_flags=[0,0,0]
    if (x2l < x1r):
        #draw the left line in green and right line in blue
        
#         print(x1l,y1l)
        if(abs(x1l-imshape[1]/2) <=0.1*imshape[1]):
            lane_flags[0]=1
            lane_flags[1]=1
        
#         print(x2r,y2r)
        if(abs(x1l-imshape[1]/2) <=0.1*imshape[1]):
            lane_flags[2]=1
            lane_flags[1]=1
        cv2.line(img, (int(x1l), int(y1l)), (int(x2l), int(y2l)), [0, 255, 0], thickness) 
        cv2.line(img, (x1r,int( y1r)), (x2r, int(y2r)), [0, 0, 255], thickness)
    
    #store lines coeficients for next cycle    
    previous_lines[0] = ml
    previous_lines[1] = bl
    previous_lines[2] = mr
    previous_lines[3] = br               
    return lane_flags

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lane_flags=draw_lines(line_img, lines)
    return line_img,lane_flags

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=0.6, γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
# os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[5]:


# img = mpimg.imread('test_images/solidYellowCurve.jpg')

# ### color selection###
# hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# mask_white = cv2.inRange(img, (200,200,200), (255, 255, 255))
# mask_yellow = cv2.inRange(hsv_img, (15,60,20), (25, 255, 255))
# color_mask = cv2.bitwise_or(mask_white, mask_yellow)
# masked_img = np.copy(img)
# masked_img[color_mask == 0] = [0,0,0]

# plt.imshow(masked_img)


# In[6]:

def convert_hsl(image):
    """
    Convert RGB images to HSL.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

# list_images(list(map(convert_hsl, test_images)))


# In[10]:


def HSL_color_selection(image):
    """
    Apply color selection to the HSL images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    #Convert the input image to HSL
    converted_image = convert_hsl(image)
    
    #White color mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    #Yellow color mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    #Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)
    
    return masked_image


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
previous_lines = [0, 0, 0, 0]
def lane_finding_pipeline(img): 
    
    ### create a color mask ###
    #convert from RGB to HSV
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     #define two color masks for yellow and white
#     #white mask is applied in RGB color space, it's easier to tune the values
#     #values from 200 to 255 on all colors are picked from trial and error
# #     mask_white = cv2.inRange(img, (200,200,200), (255, 255, 255))
#     mask_white = cv2.inRange(img, (150,150,150), (255, 255, 255))
#     #yellow mask is done in HSV color space since it's easier to identify the hue of a certain color
#     #values from 15 to 25 for hue and above 60 for saturation are picked from trial and error
#     mask_green=cv2.inRange(hsv_img, (86, 100, 100), (129, 53,48))
#     mask_grey=cv2.inRange(img,(100,100,100),(255,255,255))
    
#     mask_yellow = cv2.inRange(hsv_img, (15,60,20), (25, 255, 255))
#     #combine the two masks, both yellow and white pixels are of interest
    
#     color_mask = cv2.bitwise_or(mask_white, mask_yellow)
    
#     color_mask=mask_white
# #     mask_grey=cv2.inRange(img, (0,0,70), (180, 180, 180))
# #     color_mask=mask_grey
#     #make a copy of the original image
#     masked_img = np.copy(img)
#     #pixels that are not part of the mask(neither white or yellow) are made black
#     masked_img[color_mask == 0] = [0,0,0] 
    
#     ### smoothen image ###
#     #turn the masked image to grayscale for easier processing
#     gray_img = grayscale(masked_img)
#     #to get rid of imperfections, apply the gaussian blur
#     #kernel chosen 5, no other values are changed the implicit ones work just fine
#     kernel_size = 5
#     blurred_gray_img = gaussian_blur(gray_img, kernel_size)

#     ### detect edges ###
#     #choose values for te Canny edge detection filter
#     #for the differentioal value threshold chosen is 150 which is pretty high given that the max difference between
#     #black and white is 255
#     #low threshold of 50 which takes adjacent differential of 50 pixels as part of the edge
#     low_threshold = 50
#     high_threshold = 200
#     edges_from_img = canny(blurred_gray_img, low_threshold, high_threshold)

#     ### select region of interest###
#     #define a polygon that should frme the road given that the camera is in a fixed position
#     #polygon covers the bottom left and bottom right points of the picture
#     #with the other two top points it forms a trapezoid that points towards the center of the image
#     #the polygon is relative to the image's size
#     imshape = img.shape
#     vertices = np.array([[(0,imshape[0]),(3*imshape[1]/9, 6*imshape[0]/10), (7*imshape[1]/9, 6*imshape[0]/10), (imshape[1],imshape[0])]], dtype=np.int32)
#     print(vertices)
#     masked_edges = region_of_interest(edges_from_img, vertices)

    #MODIFIED
    hsv_img= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     mask_green=cv2.inRange(hsv_img, (50, 150,  22), ( 180, 255,  120))
    mask_green=cv2.inRange(hsv_img, (50, 150,  30), ( 100, 255,  120))
    mask_grey=cv2.inRange(hsv_img, (80,  27, 180), (255, 255, 255))
    mask_black=cv2.inRange(hsv_img, (103,  17, 100), (110,  50, 180))
    color_mask=mask_green
    # color_mask= cv2.bitwise_or(mask_green, mask_grey)
    # color_mask2=cv2.bitwise_or(color_mask, mask_black)
    masked_img = np.copy(img)
    masked_img[color_mask == 0] = [0,0,0]
    gray_img = grayscale(masked_img)
    kernel_size = 35
    blurred_gray_img = gaussian_blur(gray_img, kernel_size)

    ### detect edges ###
    #choose values for te Canny edge detection filter
    #for the differentioal value threshold chosen is 150 which is pretty high given that the max difference between
    #black and white is 255
    #low threshold of 50 which takes adjacent differential of 50 pixels as part of the edge
    low_threshold = 3
    high_threshold = 4
    edges_from_img = canny(blurred_gray_img, low_threshold, high_threshold)
    imshape = img.shape
    # vertices = np.array([[(0,imshape[0]),(3*imshape[1]/9, 4*imshape[0]/10), (7*imshape[1]/9, 4*imshape[0]/10), (imshape[1],imshape[0])]], dtype=np.int32)
    # print(vertices)
    # vertices2 = np.array([[(3*imshape[1]/9,imshape[0]),(3*imshape[1]/9, 6*imshape[0]/10), (7*imshape[1]/9, 6*imshape[0]/10), (7*imshape[1]/9,imshape[0])]], dtype=np.int32)



    vertices2 = np.array([[(0,imshape[0]),(3*imshape[1]/9,imshape[0]),(3*imshape[1]/9, 4*imshape[0]/10)]], dtype=np.int32)
    vertices3 = np.array([[(imshape[1],imshape[0]), (7*imshape[1]/9, 4*imshape[0]/10), (7*imshape[1]/9,imshape[0])]], dtype=np.int32)
#     vertices2 = np.array([[(0,imshape[0]),(0,5*imshape[0]/10),(3*imshape[1]/9,imshape[0]),(3*imshape[1]/9, 5*imshape[0]/10)]], dtype=np.int32)
#     vertices3 = np.array([[(imshape[1],imshape[0]),(imshape[1],5*imshape[0]/10), (7*imshape[1]/9, 5*imshape[0]/10), (7*imshape[1]/9,imshape[0])]], dtype=np.int32)

    masked_edges = region_of_interest(edges_from_img,vertices2, vertices3)
#     vertices2 = np.array([[(0,imshape[0]),(0,5*imshape[0]/10),(3*imshape[1]/9,imshape[0]),(3*imshape[1]/9, 5*imshape[0]/10)]], dtype=np.int32)
#     vertices3 = np.array([[(imshape[1],imshape[0]),(imshape[1],5*imshape[0]/10), (7*imshape[1]/9, 5*imshape[0]/10), (7*imshape[1]/9,imshape[0])]], dtype=np.int32)
#     masked_edges = region_of_interest(edges_from_img,vertices2, vertices3)
    # masked_edges2 = region_of_interestxor(masked_edges, vertices2)
#     masked_edges=edges_from_img
# plt.imshow(masked_edges)
    #MODIFIED
    
#     print(masked_edges)
#     masked_edges=edges_from_img #MODIFIED
    
    ### find lines from edges pixels ###
    #define parameters for the Hough transform
    #Hough grid resolution in pixels
#     rho = 2
    rho=1
    #Hough grid angular resolution in radians 
    theta = np.pi/180 
    
    #minimum number of sines intersecting in a cell, collinear points to form a line
#     threshold = 15
    threshold = 15 #modified
    #minimum length of a line in pixels
#     min_line_len = 10 
    min_line_len=30
    #maximum gap in pixels between segments to be considered part of the same line 
#     max_line_gap = 5 
    max_line_gap=10
    #apply Hough transform to color masked grayscale blurred image
    line_img,lane_flags = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    ### overlay image and lines ###
    #add lines on top of the original image
    #the lines are a bit transparent so the lane lines from the pictures still show for visual confirmation
    overlay_img = weighted_img(line_img, img)
    
    return overlay_img,lane_flags

# #get the list of images in the test folder
# img_list = os.listdir("test_images/")
# #remove items that are not .jpg pictures
# img_list.remove('.ipynb_checkpoints')

#loop over all images and put them through the lane finding pipeline
#for i in range (len(img_list)):
    #get the image name
#    img_name = img_list[i]
#    import_from = 'test_images/' + img_name 
    #read the image, matplotlib.image.imread preffered over cv2.imread() 
    #since it's easier to display through development phase and RGB is more intuitive 
#    img_in = mpimg.imread(import_from)
    #put the image thgough the lane finding pipeline
#    img_out = lane_finding_pipeline(img_in)
    #cpompose the picture destination and name
#    export_to = 'test_images_output/' + 'lanes_marked_' + img_name
    #save the image in the destination folder
#    plt.imsave(export_to, img_out)

# img = mpimg.imread('test_images/solidWhiteRight.jpg')
# img_out = lane_finding_pipeline(img)
# plt.imshow(img_out)
previous_lines = [0, 0, 0, 0]


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[7]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[8]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = lane_finding_pipeline(image)
    return result


# Let's try the one with the solid white lane on the right first ...

# In[9]:


# white_output = 'test_videos_output/solidWhiteRight.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')
previous_lines = [0, 0, 0, 0]


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[10]:


# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[11]:


# yellow_output = 'test_videos_output/solidYellowLeft.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
# clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')
# previous_lines = [0, 0, 0, 0]


# In[12]:


# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[13]:


# challenge_output = 'test_videos_output/challenge.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
# clip3 = VideoFileClip('test_videos/challenge.mp4')
# challenge_clip = clip3.fl_image(process_image)
# get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[14]:


# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(challenge_output))

