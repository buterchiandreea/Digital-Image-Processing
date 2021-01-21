import numpy as np
from cv2 import cv2
import os
import matplotlib.pyplot as plt
import sys
import os

def dilate_image(img):
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=3)
    return dilation

def apply_gaussian_blurring(img, filter_h, filter_w):
    if img is None:
        sys.exit('The image is empty.')
    elif (filter_h < 0) or (filter_w < 0) or (filter_h % 2 == 0) or (filter_w % 2 == 0):
        sys.exit('Invalid values for the dimensions of the filter.')
    else:
        blurred_img = cv2.GaussianBlur(img, (filter_w, filter_h), 2)
        return blurred_img

def get_max_contour(contours):
    max_points_contour = 0
    max_contour_index = 0
    for i in range(0, len(contours)):
        if (len(contours[i]) > max_points_contour):
            max_points_contour = len(contours[i])
            max_contour_index = i
    return max_contour_index

def compute_convex_hull(contours, max_contour_index, PATH):
    # Convex Hull
    for subdir, dirs, files in os.walk(PATH):
                i = 0
                for file in files:
                    file_path = subdir + os.sep + file
                    img = cv2.imread(file_path, 1)
                    blurred_img = apply_gaussian_blurring(img, 9, 9)
                    grayscale_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
                    _, binary_img = cv2.threshold(src=grayscale_img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
                    contours, hierachy = cv2.findContours(image=binary_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                    max_contour_index = get_max_contour(contours=contours)
                    hull = cv2.convexHull(contours[max_contour_index], returnPoints=True)
                    if (i < 5):
                        cv2.drawContours(img, [hull], 0, (255, 0, 0), 1)
                        # cv2.imshow('Convex Hull_{}'.format(i), img)
                    i += 1

    # Convexity defects
    hull_ints = cv2.convexHull(contours[max_contour_index], returnPoints=False)
    hull_points = cv2.convexHull(contours[max_contour_index], returnPoints=True)
    defects = cv2.convexityDefects(contours[max_contour_index], hull_ints)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contours[max_contour_index][s][0])
        end = tuple(contours[max_contour_index][e][0])
        far = tuple(contours[max_contour_index][f][0])
        cv2.line(img, start, end, [0, 255, 0], 2)
        cv2.circle(img, far, 2, [0, 0, 255], -1)

    cv2.imshow('Convexity defects', img)

def compute_tangent(points_first_gap, points_second_gap):
    coordinates = list()
    constants = list()
    ms = list()

    for i in range(0, len(points_first_gap)):
        for j in range(0, len(points_second_gap)):
            x1 = points_first_gap[i][0][0]
            y1 = points_first_gap[i][0][1]
            x2 = points_second_gap[j][0][0]
            y2 = points_second_gap[j][0][1]
            if ((x1 - x2) <= 0):
                continue
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                coordinates.append([points_first_gap[i], points_second_gap[j]])
                constants.append(c)
                ms.append(m)
    
    counter = 0

    for i in range(0, len(coordinates)):
        for l in range(0, len(points_first_gap)):
            for k in range(0, len(points_second_gap)):
                if ((points_first_gap[l][0][1] <= ms[i] * points_first_gap[l][0][0] + constants[i]) and 
                        points_second_gap[k][0][1] <= ms[i] * points_second_gap[k][0][0] + constants[i]):
                        counter  += 1

        if (counter == len(points_first_gap) * len(points_second_gap)):
            return coordinates[i]

    return []

def get_max_oX_coordinates(points_first_gap, points_second_gap, contour):
    max_first_gap = 0
    max_second_gap = 0
    max_contour = 0
    first_gap_coordinate = None
    second_gap_coordinate = None
    contour_max_coordinate = None

    for i in range(0, len(points_first_gap)):
        if (points_first_gap[i][0][0] > max_first_gap):
            max_first_gap = points_first_gap[i][0][0]
            first_gap_coordinate = points_first_gap[i][0]

    for i in range(0, len(points_second_gap)):
        if (points_second_gap[i][0][0] > max_second_gap):
            max_second_gap = points_second_gap[i][0][0]
            second_gap_coordinate = points_second_gap[i][0]
    
    for i in range(0, len(contour)):
        if (contour[i][0][0] > max_contour):
            max_contour = contour[i][0][0]
            contour_max_coordinate = contour[i][0]

    return first_gap_coordinate, second_gap_coordinate, contour_max_coordinate

if __name__ == "__main__":
    # BAD
    # PATH = r'E:\University\Master SAI\An II\DIP\Practical works\Bonus\Test images\PolyU_379_S_09.bmp'
    # GOOD
    PATH = r'Palmprint Detection\Test images\PolyU_381_S_06.bmp'

    # (a) Original image
    img = cv2.imread(PATH, cv2.IMREAD_COLOR)
    cv2.imshow('Original image', img)

    # (b) 
    # Blurred image
    blurred_img = apply_gaussian_blurring(img, 9, 9)
    # cv2.imshow('Blurred image', blurred_img)

    # Grayscale conversion + Thresholding
    grayscale_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(src=grayscale_img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
    dilated_img = dilate_image(binary_img)
    cv2.imshow('Binary palmprint', dilated_img)

    # (c)
    # Finding the contours of the palm
    contours, hierachy = cv2.findContours(image=dilated_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # Getting the contour with the maximum area
    max_contour_index = get_max_contour(contours=contours)

    # Contour colored with black, background white
    background = np.full(grayscale_img.shape, 255, np.uint8)
    cv2.drawContours(background, contours, max_contour_index, (0, 0, 0), 1)
    cv2.imshow('Palmprint contour', background)

    # Finding the gaps
    # Wrap the palmprint within a polygon
    epsilon = 0.009 * cv2.arcLength(contours[max_contour_index], True)
    approx = cv2.approxPolyDP(contours[max_contour_index], epsilon, True)
    cv2.drawContours(img, [approx], 0, (0, 0, 255), 1)

    start_points = list()
    start_points_indeces = list()
    end_points = list()
    end_points_indeces = list()

    # Start points of the gaps (polygon)
    for i in range(0, len(approx) - 1):
        if ((approx[i + 1][0][0]) - (approx[i][0][0]) > 20):
            start_points.append(approx[i][0])
            start_points_indeces.append(i)

    for i in range(0, len(start_points)):
        if (i == 0 or i == 2):
            cv2.circle(img, tuple(start_points[i]), 2, [0, 255, 0], -1)

    # End points of the gaps (polygon)
    for index in start_points_indeces:
        for j in range(index + 1, len(approx) - 1):
            if(abs(approx[j][0][0] - approx[j + 1][0][0]) > 20):
                end_points.append(approx[j + 1][0])
                end_points_indeces.append(j + 1)
                break

    for i in range(0, len(end_points)):
        if (i == 0 or i == 2):
            cv2.circle(img, tuple(end_points[i]), 2, [255, 0, 0], -1)

    # Finding the countour points based on the proximity 
    start_first_gap = None
    end_first_gap = None
    start_second_gap = None
    end_second_gap = None

    # First gap start
    for i in range(0, len(contours[max_contour_index])):
        if((abs(int(contours[max_contour_index][i][0][0]) - int(start_points[0][0])) <= 10) and
            (abs(int(contours[max_contour_index][i][0][1]) - int(start_points[0][1])) <= 10) and
            (int(contours[max_contour_index][i][0][1] > start_points[0][1]))):
            start_first_gap = contours[max_contour_index][i][0]
            break

    # First gap end
    for i in range(0, len(contours[max_contour_index])):
        if((abs(int(contours[max_contour_index][i][0][0]) - int(end_points[0][0])) <= 10) and
            (abs(int(contours[max_contour_index][i][0][1]) - int(end_points[0][1])) <= 10) and
            (int(contours[max_contour_index][i][0][0] < end_points[0][0]))):
            end_first_gap = contours[max_contour_index][i][0]
            break   

    # Second gap start
    for i in range(0, len(contours[max_contour_index])):
        if((abs(int(contours[max_contour_index][i][0][0]) - int(start_points[2][0])) <= 10) and
            (abs(int(contours[max_contour_index][i][0][1]) - int(start_points[2][1])) <= 10) and
            (int(contours[max_contour_index][i][0][1] > start_points[2][1]))):
            start_second_gap = contours[max_contour_index][i][0]
            break

    # Second gap end
    for i in range(0, len(contours[max_contour_index])):
        if((abs(int(contours[max_contour_index][i][0][0]) - int(end_points[2][0])) <= 10) and
            (abs(int(contours[max_contour_index][i][0][1]) - int(end_points[2][1])) <= 10) and
            (int(contours[max_contour_index][i][0][0] < end_points[2][0]))):
            end_second_gap = contours[max_contour_index][i][0]
            break   
    
    # Getting the contour points for each gap
    # First gap
    first_start_pos = None
    first_end_pos = None

    for i in range(0, len(contours[max_contour_index])):
        if (start_first_gap[0] == contours[max_contour_index][i][0][0] and start_first_gap[1] == contours[max_contour_index][i][0][1]):
            first_start_pos = i
            break

    for i in range(0, len(contours[max_contour_index])):
        if (end_first_gap[0] == contours[max_contour_index][i][0][0] and end_first_gap[1] == contours[max_contour_index][i][0][1]):
            first_end_pos = i
            break

    # Second gap
    second_start_pos = None
    second_end_pos = None

    for i in range(0, len(contours[max_contour_index])):
        if (start_second_gap[0] == contours[max_contour_index][i][0][0] and start_second_gap[1] == contours[max_contour_index][i][0][1]):
            second_start_pos = i
            break

    for i in range(0, len(contours[max_contour_index])):
        if (end_second_gap[0] == contours[max_contour_index][i][0][0] and end_second_gap[1] == contours[max_contour_index][i][0][1]):
            second_end_pos = i
            break
    
    # Color each pixel with an intensity of 190 
    # First gap
    points_first_gap = contours[max_contour_index][first_start_pos : first_end_pos]
    for point in points_first_gap:
        background[point[0][1]][point[0][0]] = 190

    # Second gap
    points_second_gap = contours[max_contour_index][second_start_pos : second_end_pos]
    for point in points_second_gap:
        background[point[0][1]][point[0][0]] = 190


    # Computations for the tangent of the gap
    max_first_gap, max_second_gap, contour_max = get_max_oX_coordinates(points_first_gap, points_second_gap, contours[max_contour_index])
    midpoint_oY = tuple([int((max_first_gap[0] + max_second_gap[0]) / 2), int((max_first_gap[1] + max_second_gap[1]) / 2)]) 
    midpoint_oX = tuple([int((midpoint_oY[0] + contour_max[0]) / 2), midpoint_oY[1]])
    third = int(int((midpoint_oY[0] + contour_max[0]) / 2) / 3)
    # Up corner
    left_up = tuple([midpoint_oX[0] - third, midpoint_oX[1] - third])
    # Down corner
    right_down = tuple([midpoint_oX[0] + third, midpoint_oX[1] + third])

    # Drawing the points/lines on the original image
    cv2.line(img, tuple(max_first_gap), tuple(max_second_gap), [255, 45, 100], 1)
    cv2.line(img, tuple(midpoint_oY), tuple([contour_max[0], midpoint_oY[1]]), [255, 45, 100], 1)
    cv2.circle(img, midpoint_oX, 2, [186, 3, 252], -1)
    cv2.circle(img, left_up, 2, [186, 3, 252], -1)
    cv2.circle(img, right_down, 2, [186, 3, 252], -1)

    cv2.imshow('Gaps', background)
    cv2.imshow('Interest zones', img)

    # Region of interest
    last_step = cv2.imread(PATH)
    roi = last_step[left_up[1]:right_down[1], left_up[0]:right_down[0]]
    cv2.imshow('ROI', roi)

    # Convex Hull section
    CONVEX_HULL_PATH = r'Palmprint Detection\Convex Hull'
    compute_convex_hull(contours, max_contour_index, CONVEX_HULL_PATH)

    cv2.waitKey(0)
    cv2.destroyAllWindows()