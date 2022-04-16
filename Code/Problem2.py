import numpy as np
import cv2

def detect_edge(gray_image):
    """function to detect edges

    Args:
        gray_image (Numpy array): Input image converted from BGR to Grayscale

    Returns:
        list: returns edges detected
    """

    blur = cv2.GaussianBlur(gray_image, (3,3), 0)
    edges = cv2.Canny(blur, 150, 255)

    return edges


def get_line_type(frame):
    """Function to get the type of lane(dashed or solid)

    Args:
        frame (Numpy array): input frame

    Returns:
        contours: retyrns detected contours
    """


    height = frame.shape[0]
    width = frame.shape[1]
    src = np.array([[(0, height),(round(width/3.6), round(height - (height/3))), (round(width - (width/3.5)), round(height - (height/3))), ((width), height)]]) 
    destination = np.array([[0,0], [0,200], [200,200], [200,0]])

    
    h, _ = cv2.findHomography(src, destination, cv2.RANSAC, 5)
    ret,thresh2 = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
    dst = np.zeros([200,200,3],dtype=np.uint8)
    img_dst = cv2.warpPerspective(thresh2, h, (dst.shape[1], dst.shape[0]))
    final = cv2.flip(img_dst, 0)

    w = final.shape[1]

    #cropping the image in half to get only one lane line
    half = w//2

    left_part = final[:, :half] 
    right_part = final[:, half:] 

    gray_l = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_part, cv2.COLOR_BGR2GRAY) 

    #finding contours of the left and right lane lones
    cont_l, hierarchy = cv2.findContours(gray_l, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(cont_l))
    cont_r, hierarchy = cv2.findContours(gray_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(cont_r))

    return cont_l, cont_r


def process_image(image):
    """Funtion to mask the input image with a polygon

    Args:
        image (Numpy array): Input image

    Returns:
        Numpy arrya: Masked image with region of interest mask
    """

    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    ROI = np.array([[(0, height),(round(width/3.6), round(height - (height/3))), (round(width - (width/3.5)), round(height - (height/3))), ((width), height)]]) 

    match_mask_color = (255)
    cv2.fillPoly(mask, ROI, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def draw_line(img, lines, color, thickness):
    """Function to dray the lane lines

    Args:
        img (Numpy array): input image
        lines (array): sorted lines 
        color (list): color of line
        thickness (int): thicknbess of the line

    Returns:
        Numpy array: returns the image with lines drawn
    """

    if len(lines) > 1:
        lines_array = np.vstack(lines)
        x1 = np.min(lines_array[:, 0])
        x2 = np.max(lines_array[:, 2])
        y1 = lines_array[np.argmin(lines_array[:, 0]), 1]
        y2 = lines_array[np.argmax(lines_array[:, 2]), 3]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        return img



def get_line_image(img,lines):
    """Function to process the image and get the final image with lines drawn

    Args:
        img (Numpy arrya): input image
        lines (array): lines detected 

    Returns:
        Numpy array: returns the final image with drawn lines
    """

    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            p = np.polyfit((x1, x2), (y1, y2), 1)
            if p[0] < -0.5:
                left_lines.append(line)
            elif p[0] > 0.5:
                right_lines.append(line)

    # print(right_lines)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cont_l, cont_r = get_line_type(frame)


    #dotted line in blue all the time
    if len(cont_l) < len(cont_r):
        color_l = (0,255,255)
        color_r = (255,0,0) #yellow
    else:
        color_r = (0,255,255)
        color_l = (255,0,0)   #blue

    draw_line(line_img, left_lines, color_l, thickness=10)
    draw_line(line_img, right_lines, color_r, thickness=10)

    #Adding the input image and the line image
    img = cv2.addWeighted(img,0.8,line_img,1.0,0.0)
    
    return img

if __name__ == "__main__":
    cap = cv2.VideoCapture("Project_2/whiteline.mp4")
    while cap.isOpened():

        ret, frame = cap.read()
        if ret == False:
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        edges = detect_edge(gray_image)
        mask_img = process_image(edges)

        height = frame.shape[0]
        width = frame.shape[1]
        src = np.array([[(0, height),(round(width/3.6), round(height - (height/3))), (round(width - (width/3.5)), round(height - (height/3))), ((width), height)]]) 

        #Using Houhghlines to detect the lines
        lines = cv2.HoughLinesP(mask_img,rho=2,threshold=60,theta=np.pi/180,minLineLength=30 ,maxLineGap=100,lines=np.array([]))
        final_output = get_line_image(frame,lines)


        cv2.imshow('serh', final_output)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()