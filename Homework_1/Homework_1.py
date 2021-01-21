import numpy as np
from cv2 import cv2
import os
import matplotlib.pyplot as plt

def convert_from_BGR_to_HSV(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = hsv_img.astype(np.float32)
    height = hsv_img.shape[0]
    width = hsv_img.shape[1]
    max_H = 0
    max_S = 0
    max_V = 0

    for y in range(0, height):
        for x in range(0, width):
            H_channel = hsv_img.item(y, x, 0)
            S_channel = hsv_img.item(y, x, 1)
            V_channel = hsv_img.item(y, x, 2)

            if (H_channel > max_H):
                max_H = H_channel
            if (S_channel > max_S):
                max_S = S_channel
            if (V_channel > max_V):
                max_V = V_channel

    if ((max_H >= 0) and (max_H <= 255)):
        for y in range(0, height):
            for x in range(0, width):
                H = hsv_img.item(y, x, 0)
                hsv_img.itemset((y, x, 0), round(((H / 255.0) * 360), 2))

    if ((max_S >= 0) and (max_S <= 255)):
        for y in range(0, height):
            for x in range(0, width):
                S = hsv_img.item(y, x, 1)
                hsv_img.itemset((y, x, 1), round((S / 255.0), 2))

    if ((max_V >= 0) and (max_V <= 255)):
        for y in range(0, height):
            for x in range(0, width):
                V = hsv_img.item(y, x, 2)
                hsv_img.itemset((y, x, 2), round((V / 255.0), 2))

    return hsv_img


def convert_from_BGR_to_YCbCr(img):
    yCbCr_img = img.astype(np.float32)
    height = yCbCr_img.shape[0]
    width = yCbCr_img.shape[1]

    for y in range(0, height):
        for x in range(0, width):
            B_channel = yCbCr_img.item(y, x, 0)
            G_channel = yCbCr_img.item(y, x, 1)
            R_channel = yCbCr_img.item(y, x, 2)

            Y = (0.299 * R_channel) + (0.587 * G_channel) + (0.114 * B_channel)
            Cb = (-0.1687 * R_channel) + (-0.3313 * G_channel) + (0.5 * B_channel) + 128
            Cr = (0.5 * R_channel) + (-0.4187 * G_channel) + (-0.0813 * B_channel) + 128

            yCbCr_img.itemset((y, x, 0), Y)
            yCbCr_img.itemset((y, x, 1), Cb)
            yCbCr_img.itemset((y, x, 2), Cr)

    return yCbCr_img

def method_1(img):
    img_height = img.shape[0]
    img_width = img.shape[1]

    for y in range (0, img_height):
        for x in range (0, img_width):
            B = img.item(y, x, 0)
            G = img.item(y, x, 1)
            R = img.item(y, x, 2)

            if ((R > 95) and (G > 40) and (B > 20) and
                ((max(R, G, B) - min(R, G, B)) > 15) and 
                (abs(R - G) > 15) and
                (R > G) and
                (R > B)):

                img.itemset((y, x, 0), 255)
                img.itemset((y, x, 1), 255)
                img.itemset((y, x, 2), 255)
            else:
                img.itemset((y, x, 0), 0)
                img.itemset((y, x, 1), 0)
                img.itemset((y, x, 2), 0)

    return img

def method_2(img):
    img_height = img.shape[0]
    img_width = img.shape[1]

    for y in range (0, img_height):
        for x in range (0, img_width):
            B = img.item(y, x, 0)
            G = img.item(y, x, 1)
            R = img.item(y, x, 2)

            if ((R == 0) or (G == 0) or (B == 0)):
                img.itemset((y, x, 0), 0)
                img.itemset((y, x, 1), 0)
                img.itemset((y, x, 2), 0)

            elif (((R / G) > 1.185) and 
                (((R * B) / pow((R + G + B), 2)) > 0.107) and
                (((R * G) / pow((R + G + B), 2)) > 0.112)):

                img.itemset((y, x, 0), 255)
                img.itemset((y, x, 1), 255)
                img.itemset((y, x, 2), 255)
            else:
                img.itemset((y, x, 0), 0)
                img.itemset((y, x, 1), 0)
                img.itemset((y, x, 2), 0)
    
    return img

def method_3(img):
    HSV_img = convert_from_BGR_to_HSV(img)
    img_height = HSV_img.shape[0]
    img_width = HSV_img.shape[1]

    for y in range (0, img_height):
        for x in range (0, img_width):
            H = HSV_img.item(y, x, 0)
            S = HSV_img.item(y, x, 1)
            V = HSV_img.item(y, x, 2)

            if ((V >= 0.4) and 
                (0.2 < S < 0.6) and 
                ((0 < H < 25) or (335 < H <= 360))):
                HSV_img.itemset((y, x, 0), 255)
                HSV_img.itemset((y, x, 1), 255)
                HSV_img.itemset((y, x, 2), 255)
            else:
                HSV_img.itemset((y, x, 0), 0)
                HSV_img.itemset((y, x, 1), 0)
                HSV_img.itemset((y, x, 2), 0)

    return HSV_img

def method_4(img):
    HSV_img = convert_from_BGR_to_HSV(img)
    img_height = HSV_img.shape[0]
    img_width = HSV_img.shape[1]

    for y in range (0, img_height):
        for x in range (0, img_width):
            H = HSV_img.item(y, x, 0)
            S = HSV_img.item(y, x, 1)
            V = HSV_img.item(y, x, 2)

            if ((0 <= H <= 50) and
                (0.23 <= S <= 0.68) and
                (0.35 <= V <= 1)):
                HSV_img.itemset((y, x, 0), 255)
                HSV_img.itemset((y, x, 1), 255)
                HSV_img.itemset((y, x, 2), 255)
            else:
                HSV_img.itemset((y, x, 0), 0)
                HSV_img.itemset((y, x, 1), 0)
                HSV_img.itemset((y, x, 2), 0)

    return HSV_img

def method_5(img):
    HSV_img = convert_from_BGR_to_HSV(img)
    img_height = HSV_img.shape[0]
    img_width = HSV_img.shape[1]

    for y in range (0, img_height):
        for x in range (0, img_width):
            H = HSV_img.item(y, x, 0)
            S = HSV_img.item(y, x, 1)
            V = HSV_img.item(y, x, 2)

            if (((0 <= H <= 50) or (340 <= H <= 360)) and
                (S >= 0.2) and
                (V >= 0.35)):
                HSV_img.itemset((y, x, 0), 255)
                HSV_img.itemset((y, x, 1), 255)
                HSV_img.itemset((y, x, 2), 255)
            else:
                HSV_img.itemset((y, x, 0), 0)
                HSV_img.itemset((y, x, 1), 0)
                HSV_img.itemset((y, x, 2), 0)

    return HSV_img

def method_6(img):
    yCbCr_img = convert_from_BGR_to_YCbCr(img)
    img_height = yCbCr_img.shape[0]
    img_width = yCbCr_img.shape[1]

    for y in range (0, img_height):
        for x in range (0, img_width):
            Y = yCbCr_img.item(y, x, 0)
            Cb = yCbCr_img.item(y, x, 1)
            Cr = yCbCr_img.item(y, x, 2)

            if ((Y > 80) and (85 < Cb < 135) and (135 < Cr < 180)):
                yCbCr_img.itemset((y, x, 0), 255)
                yCbCr_img.itemset((y, x, 1), 255)
                yCbCr_img.itemset((y, x, 2), 255)
            else:
                yCbCr_img.itemset((y, x, 0), 0)
                yCbCr_img.itemset((y, x, 1), 0)
                yCbCr_img.itemset((y, x, 2), 0)

    return yCbCr_img

def method_7(img):
    yCbCr_img = convert_from_BGR_to_YCbCr(img)
    img_height = yCbCr_img.shape[0]
    img_width = yCbCr_img.shape[1]

    for y in range (0, img_height):
        for x in range (0, img_width):
            Y = yCbCr_img.item(y, x, 0)
            Cb = yCbCr_img.item(y, x, 1)
            Cr = yCbCr_img.item(y, x, 2)

            if ((Cr <= 1.5862 * Cb + 20) and
                (Cr >= 0.3448 * Cb + 76.2069) and
                (Cr >= -4.5652 * Cb + 234.5652) and
                (Cr <= -1.15 * Cb + 301.75) and
                (Cr <= -2.2857 * Cb + 432.85)):

                yCbCr_img.itemset((y, x, 0), 255)
                yCbCr_img.itemset((y, x, 1), 255)
                yCbCr_img.itemset((y, x, 2), 255)
            else:
                yCbCr_img.itemset((y, x, 0), 0)
                yCbCr_img.itemset((y, x, 1), 0)
                yCbCr_img.itemset((y, x, 2), 0)

    return yCbCr_img

def detect_skin(method_used=None):
    if ((method_used is None) or (method_used < 1) or (method_used > 7)):
        print('No method selected. Please choose one of the 7 existing methods for skin detection.')
    else:
        input_dir = r'Homework_1\Test images'
        output_dir = 'Homework_1\Output' + os.sep + 'Output_{}'.format(method_used)
        for subdir, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = subdir + os.sep + file
                img = cv2.imread(file_path, 1)
                if (method_used == 1):
                    new_img = method_1(img)
                    cv2.imwrite((output_dir + os.sep + file), new_img)
                elif (method_used == 2):
                    new_img = method_2(img)
                    cv2.imwrite((output_dir + os.sep + file), new_img)
                elif (method_used == 3):
                    new_img = method_3(img)
                    cv2.imwrite((output_dir + os.sep + file), new_img)
                elif (method_used == 4):
                    new_img = method_4(img)
                    cv2.imwrite((output_dir + os.sep + file), new_img)
                elif (method_used == 5):
                    new_img = method_5(img)
                    cv2.imwrite((output_dir + os.sep + file), new_img)
                elif (method_used == 6):
                    new_img = method_6(img)
                    cv2.imwrite((output_dir + os.sep + file), new_img)
                elif (method_used == 7):
                    new_img = method_7(img)
                    cv2.imwrite((output_dir + os.sep + file), new_img)

def display_grid(images, titles):
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    columns = 4
    rows = 2
    for i in range(1, columns*rows + 1):
        if ((i - 1) == 0):
            img = cv2.imread(images[i - 1], 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(images[i - 1], 0)
        fig.add_subplot(rows, columns, i, title=titles[i - 1])
        if ((i - 1) == 0):
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()

def display_skin_detection_results(file_name):
    output_directory_path = r'Homework_1\Output\Output_{}\{}'
    input_directory_path = r'Homework_1\Test images\{}'
    file_paths = list()
    titles = ['Original image', 'RGB 1', 'RGB 2', 'HSV 1', 'HSV 2', 'HSV 3', 'YCbCr 1', 'YCbCr 2']
    file_paths.append(input_directory_path.format(file_name))

    for i in range(0, 7):
        file_paths.append(output_directory_path.format(str(i + 1), file_name))

    display_grid(file_paths, titles)

def create_emoticon():
    emoticon = np.zeros([512, 512, 3], np.uint8)
    emoticon = cv2.circle(emoticon, (256, 256), 200, (0, 215, 255), -1)
    emoticon = cv2.ellipse(emoticon, (166, 200), (40, 30), 90, 0, 360, (0, 0, 0), -1)
    emoticon = cv2.ellipse(emoticon, (166, 200), (40, 30), 90, 0, 360, (255, 255, 255), 1)
    emoticon = cv2.ellipse(emoticon, (346, 200), (40, 30), 90, 0, 360, (0, 0, 0), -1)
    emoticon = cv2.ellipse(emoticon, (346, 200), (40, 30), 90, 0, 360, (255, 255, 255), 1)
    emoticon = cv2.circle(emoticon, (180, 175), 7, (255, 255, 255), -1)
    emoticon = cv2.circle(emoticon, (360, 175), 7, (255, 255, 255), -1)
    pts = [(130, 260), (150, 390), (362, 390), (382, 260), (256, 248)]
    cv2.fillPoly(emoticon, np.array([pts]), (208, 224, 64))
    cv2.polylines(emoticon, np.array([pts]), True, (255, 255, 255), 2)
    cv2.line(emoticon, (150, 290), (362, 290), (209, 206, 0), 2)
    cv2.line(emoticon, (155, 330), (357, 330), (209, 206, 0), 2)
    cv2.line(emoticon, (160, 360), (352, 360), (209, 206, 0), 2)
    cv2.line(emoticon, (130, 260), (65, 200), (255, 255, 255), 2)
    cv2.line(emoticon, (382, 260), (447, 200), (255, 255, 255), 2)
    cv2.line(emoticon, (150, 390), (125, 405), (255, 255, 255), 2)
    cv2.line(emoticon, (362, 390), (387, 405), (255, 255, 255), 2)
    cv2.imshow('emoticon', emoticon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dilate_image(img):
    cv2.imshow('Original image', img)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=2)
    cv2.imshow('Dilated image', dilation)
    
    return dilation

def detect_faces(col_img, img):
    dilated_img = dilate_image(img)
    img_gray = cv2.cvtColor(dilated_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_points_contour = 0
    max_contour_index = 0

    for i in range(0, len(contours)):
        if (len(contours[i]) > max_points_contour):
            max_points_contour = len(contours[i])
            max_contour_index = i

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        width = cv2.boundingRect(contour)[2]
        height = cv2.boundingRect(contour)[3]
        if ((cv2.contourArea(contour) < 200) or (width > height * 2)):
            continue
        cv2.rectangle(col_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Colored image', col_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # The results for skin detection   
    images_to_display = ['1.jpg', '2.jpg', 't63.jpg']
    for img in images_to_display:
        display_skin_detection_results(img)

    # The emoticon
    # create_emoticon()

    # Face detection
    # GROUP = r'E:\University\Master SAI\An II\DIP\Practical works\Homework_1\Test images\1.jpg'
    # GROUP_O = r'E:\University\Master SAI\An II\DIP\Practical works\Homework_1\Output\Output_1\1.jpg'
    # col_img = cv2.imread(GROUP)
    # img = cv2.imread(GROUP_O)
    # detect_faces(col_img, img)
    pass