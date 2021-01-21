import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import statistics


def simple_averaging(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    gray_img = np.zeros((img_height, img_width), np.float32)

    for y in range (0, img_height):
        for x in range (0, img_width):
            B = img.item(y, x, 0)
            G = img.item(y, x, 1)
            R = img.item(y, x, 2)

            gray_img[y][x] = ((R + G + B) / 3.0)

    return gray_img

def weighted_average(img, method):
    img_height = img.shape[0]
    img_width = img.shape[1]
    gray_img = np.zeros((img_height, img_width), np.float32)

    for y in range (0, img_height):
        for x in range (0, img_width):
            B = img.item(y, x, 0)
            G = img.item(y, x, 1)
            R = img.item(y, x, 2)

            if (method == 1):
                gray_img[y][x] = 0.3 * R + 0.59 * G + 0.11 * B
            elif (method == 2):
                gray_img[y][x] = 0.2126 * R + 0.7152 * G + 0.0722 * B
            elif (method == 3):
                gray_img[y][x] = 0.299 * R + 0.587 * G + 0.114 * B

    return gray_img

def desaturation(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    gray_img = np.zeros((img_height, img_width), np.float32)

    for y in range (0, img_height):
        for x in range (0, img_width):
            B = img.item(y, x, 0)
            G = img.item(y, x, 1)
            R = img.item(y, x, 2)

            gray_img[y][x] = (min(R, G, B) + max(R, G, B)) / 2

    return gray_img

def single_colour_channel(img, method):
    img_height = img.shape[0]
    img_width = img.shape[1]
    gray_img = np.zeros((img_height, img_width), np.float32)

    for y in range (0, img_height):
        for x in range (0, img_width):
            B = img.item(y, x, 0)
            G = img.item(y, x, 1)
            R = img.item(y, x, 2)

            if (method == 1):
                gray_img[y][x] = R
            elif (method == 2):
                gray_img[y][x] = G
            elif (method == 3):
                gray_img[y][x] = B

    return gray_img

def decomposition(img, method):
    img_height = img.shape[0]
    img_width = img.shape[1]
    gray_img = np.zeros((img_height, img_width), np.float32)

    for y in range (0, img_height):
        for x in range (0, img_width):
            B = img.item(y, x, 0)
            G = img.item(y, x, 1)
            R = img.item(y, x, 2)

            if (method == 1):
                gray_img[y][x] = max(R, G, B)
            elif (method == 2):
                gray_img[y][x] = min(R, G, B)

    return gray_img

def custom_gray_shades(img, p):
    gray_img = weighted_average(img, 1)
    img_height = gray_img.shape[0]
    img_width = gray_img.shape[1]

    for y in range (0, img_height):
        for x in range (0, img_width):
            interval = int(255 / p)
            offset = interval
            pixel = gray_img[y][x]

            iterations = 1
            while (pixel > offset):
                offset += offset
                iterations += 1
            
            if (offset > 255):
                offset = 255
                interval_values = [x for x in range((offset - interval - (255 % p)), offset)]
                gray_img[y][x] = statistics.median(interval_values)
            else:
                interval_values = [x for x in range((offset - interval), offset)]
                gray_img[y][x] = statistics.median(interval_values)

    return gray_img

def error_diffusion_dithering(img):
    img = weighted_average(img, 1)
    img_height = img.shape[0]
    img_width = img.shape[1]
    gray_img = np.zeros([img_height, img_width], np.float32)

    for y in range (1, img_height - 1):
        for x in range (1, img_width - 1):
            disstance_to_black = img[y][x]
            disstance_to_white = 255 - img[y][x]
            if (disstance_to_black > disstance_to_white):
                gray_img[y][x] = 255
            else:
                gray_img[y][x] = 0
            err = img[y][x] - gray_img[y][x]
            img[y][x + 1] += err * (7 / 16)
            img[y + 1][x - 1] += err * (3 / 16)
            img[y + 1][x] += err * (5 / 16)
            img[y + 1][x + 1] += err * (1 / 16)
    
    return gray_img

def convert_from_gray_scale_to_RGB(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    RGB_img = np.zeros((img_height, img_width, 3), np.uint8)

    for y in range (0, img_height):
        for x in range (0, img_width):
            RGB_img.itemset((y, x, 0), img[y][x] * (1 / (0.299 * 3)))
            RGB_img.itemset((y, x, 1), img[y][x] * (1 / (0.587 * 3)))
            RGB_img.itemset((y, x, 2), img[y][x] * (1 / (0.114 * 3)))
    return RGB_img

def display_grid(images, titles):
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    columns = 7
    rows = 2
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i, title=titles[i - 1])
        if (((i - 1) == 0) or ((i - 1) == 1)):
            plt.imshow(images[i - 1])
        else:
            plt.imshow(images[i - 1], cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == "__main__":
    img = cv2.imread(r'Homework_2\lena.png', 1)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.imread(r'Homework_2\lena.png', 0)

    img_simple_avg = simple_averaging(img)
    img_weight_avg_1 = weighted_average(img, 1)
    img_weight_avg_2 = weighted_average(img, 2)
    img_weight_avg_3 = weighted_average(img, 3)
    img_desaturation = desaturation(img)
    img_single_channel_1 = single_colour_channel(img, 1)
    img_single_channel_2 = single_colour_channel(img, 2)
    img_single_channel_3 = single_colour_channel(img, 3)
    img_decomposition_1 = decomposition(img, 1)
    img_decomposition_2 = decomposition(img, 2)
    img_custom_gray_shades = custom_gray_shades(img, 16)
    img_err_diffusion = error_diffusion_dithering(img)
    converted_from_grayscale = convert_from_gray_scale_to_RGB(img_gray)

    images = [
        RGB_img,
        converted_from_grayscale,
        img_simple_avg, 
        img_weight_avg_1, 
        img_weight_avg_2, 
        img_weight_avg_3, 
        img_desaturation, 
        img_single_channel_1, 
        img_single_channel_2, 
        img_single_channel_3,
        img_decomposition_1, 
        img_decomposition_2, 
        img_custom_gray_shades, 
        img_err_diffusion
    ]

    titles = [
        "Original image", 
        "From Grayscale to RGB", 
        "Simple AVG", 
        "Weighted AVG 1", 
        "Weighted AVG 2", 
        "Weighted AVG 3", 
        "Saturation", 
        "Single channel 1",
        "Single channel 2", 
        "Single channel 3", 
        "Decomposition 1", 
        "Decomposition 2", 
        "Custom Gray Shades", 
        "Error diffusion" 
    ]

    display_grid(images, titles)