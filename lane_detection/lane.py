import cv2 
import sys
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg') 


def make_coordinates(img, line_parameters): 
    slope, intercept = line_parameters 
    y1 = img.shape[0] 
    y2 = int(y1*(3/5)) 
    x1 = int((y1 - intercept)/slope) 
    x2 = int((y2 - intercept)/slope) 
    return np.array([x1, y1, x2, y2])



def average_slope_intercept(img, lines): 
    left_fit = [] 
    right_fit = [] 
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4) 
        parameters = np.polyfit((x1, x2), (y1, y2), 1) 
        slope = parameters[0] 
        intercept = parameters[1] 
        if slope < 0: 
            left_fit.append((slope, intercept)) 
        else:
            right_fit.append((slope, intercept)) 
    left_fit_average = np.average(left_fit, axis=0) 
    right_fit_average = np.average(right_fit, axis=0) 
    left_line = make_coordinates(img, left_fit_average) 
    right_line = make_coordinates(img, right_fit_average)
    return np.array([left_line, right_line])


def canny_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #smoothening using gaussian blur 
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    canny_img = cv2.Canny(blur_img, 50, 150)
    return canny_img



def disp_lines(img, lines):
    line_img = np.zeros_like(img) 
    if lines is not None:
        for x1, y1, x2, y2 in lines:  
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10) 
    return line_img



def region_interest(img):
    height = img.shape[0] 
    polygns = np.array([
    [(200, height), (1100, height), (550, 250)]
    ]) 
    mask = np.zeros_like(img) 
    cv2.fillPoly(mask, polygns, 255) 
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img



#load images and convert to grayscale
#img = cv2.imread("test_image.jpg")
#lane_img = np.copy(img) 
#show images 
#canny_img = canny_img(lane_img)
# crop_img = region_interest(canny_img) 
# lines = cv2.HoughLinesP(crop_img,  2 , np.pi/180, 100, 
#                         np.array([]), minLineLength = 40, maxLineGap = 5) 
# average_lines = average_slope_intercept(lane_img, lines)
# line_img = disp_lines(lane_img, average_lines) 
# comb_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)                  
# cv2.imshow("canny_img", comb_img ) 
# cv2.waitKey(0) 



cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read() 
    canny = canny_img(frame)
    crop_img = region_interest(canny) 
    lines = cv2.HoughLinesP(crop_img,  2 , np.pi/180, 100, 
                        np.array([]), minLineLength = 40, maxLineGap = 5) 
    average_lines = average_slope_intercept(frame, lines)
    line_img = disp_lines(frame, average_lines) 
    comb_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)                  
    cv2.imshow("canny_img", comb_img ) 
    if cv2.waitKey(1) == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows() 