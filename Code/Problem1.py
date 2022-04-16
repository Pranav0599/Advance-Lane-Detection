import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import isfile, join



def get_histogram_of_image(image):
    """Function to get the histogram of the image

    Args:
        image (Numpy array): Input image

    Returns:
        Numpy array: returns the array of histogram
    """
    height, width = image.shape
    histogram = [0.0] * 256
    for i in range(height):
        for j in range(width):
            intensity = image[i, j]
            histogram[intensity] += 1


    return np.array(histogram)/(height*width), np.array(histogram)




def get_cumulative_sum(hist, image):
    """Function to get the cumulative sum of the histogram(CDF)

    Args:
        hist (Numpy array): Histogram aquired of the image
        image (Numpy array): input image

    Returns:
        float, array: returns the cumulative sum of the histogram
    """
    cumsum = []
    for i in range(len(hist)):
        cumsum.append(sum(hist[:i+1]))

    # plt.plot(cumsum)
    # plt.show()
    return cumsum, np.array(cumsum)/(image.shape[0]*image.shape[1])




def get_equlized_histogram_image(image):
    """Function to equlize the histogram of an image

    Args:
        image (Numpy array): input image

    Returns:
        Numpy array: returns image with equlized histogram applied on it
    """
    histo, _ = get_histogram_of_image(image)
    cum_sum, _ = np.array(get_cumulative_sum(histo, image))
    # print(cum_sum)
    transfered_values = np.uint8(255 * cum_sum)
    h, w = image.shape
    final_image = np.zeros_like(image)
    #final equalized image
    for i in range(0, h):
        for j in range(0, w):
            final_image[i, j] = transfered_values[image[i, j]]


    return final_image



def get_AH_equalized_image(image, win_size):
    """Function to get perform adaptive historgam equalization on the image

    Args:
        image (Numpy array): input image
        win_size (int): size of the window in a grid

    Returns:
        Numpy array: returns the image with Adaptive histogram equalization applied on it
    """
    image_copy = image.copy()

    for row in range(win_size, image.shape[0]-win_size, win_size):
        for col in range(win_size, image.shape[1]-win_size, win_size):

            window = image_copy[row-win_size:row+win_size, col-win_size:col+win_size]
            _, hist = get_histogram_of_image(window)     
            _, cum_sum = get_cumulative_sum(hist, window)
            transfered_values = np.uint8(255 * cum_sum)

            final_image = np.zeros_like(window) #final equalized image
    
            for row_2 in range(window.shape[0]):
                for col_2 in range(window.shape[1]):
                    final_image[row_2, col_2] = transfered_values[window[row_2, col_2]]

            image_copy[row-win_size:row+win_size, col-win_size:col+win_size] = final_image

    
    return image_copy





def convert_frames_to_video(in_put_path,out_put_path,path_out_AHE, fps):
    """Funtion to convert the sequance of imaged in a video

    """
    frame_array = []
    files = [f for f in os.listdir(in_put_path) if isfile(join(in_put_path, f))]
    files.sort(key = lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename=in_put_path + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        # print(filename)
        b,g,r= cv2.split(img)
        if out_put_path is not None:
            #Normal histogram eqaluzation on each frame
            equalized_B_channel = get_equlized_histogram_image(b)
            equalized_G_channel = get_equlized_histogram_image(g)
            equalized_R_channel = get_equlized_histogram_image(r)
            final = cv2.merge((equalized_B_channel, equalized_G_channel, equalized_R_channel))
            frame_array.append(final)

        else:
            #Adaptive histogram equalization on each frmae
            equalized_B_channel_AHE = get_AH_equalized_image(b, 100)
            equalized_G_channel_AHE = get_AH_equalized_image(g, 100)
            equalized_R_channel_AHE = get_AH_equalized_image(r, 100)
            final_AHE = cv2.merge((equalized_B_channel_AHE, equalized_G_channel_AHE, equalized_R_channel_AHE))
            frame_array.append(final_AHE)

    if out_put_path is not None:
        out = cv2.VideoWriter(out_put_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    else:
        out = cv2.VideoWriter(path_out_AHE,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)


    
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    



if __name__=="__main__":

    pathIn= '/home/pranav/673/Project_2/adaptive_hist_data/'
    pathOut = 'Eqalized_output.avi'
    path_out_AHE = 'AHE_output.avi'
    fps = 20.0

    convert_frames_to_video(pathIn, pathOut,None,  fps)
    convert_frames_to_video(pathIn, None, path_out_AHE, fps)