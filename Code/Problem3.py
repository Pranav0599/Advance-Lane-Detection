import numpy as np
import cv2


#global parameters
windows = 10
margin = 100 
min_pixels = 50


def apply_color_mask(image):
    """Function to for color segmentation

    Args:
        image (Numpy array): input image

    Returns:
        Numpy array: Masked image with segmented colors
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
    # denoise = cv2.fastNlMeansDenoisingColored(gray_image,None,10,10,7,21)
    l = np.uint8([  0, 0,   200])
    u = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(thresh_image, l, u)


    lower = np.uint8([ 15, 50, 80])
    upper = np.uint8([ 50, 200, 255])
    yellow_mask = cv2.inRange(thresh_image, lower, upper)
    

    masked = cv2.bitwise_or(white_mask, yellow_mask)

    return masked




def get_birdseye_view(frame):
    """Function to get the birds eye view of the frame

    Args:
        frame (Numpy array): input image

    Returns:
        Array, Array: returns the warped image and inverse perspective transform matrix
    """
    src = np.float32([(580, 447),(150, 730),(1250, 720),(725, 447)]) 
    img_size = frame.shape
    dst = np.float32([[0, 0],[0, img_size[1]],[img_size[0], img_size[1]],[img_size[0], 0]])

    m = cv2.getPerspectiveTransform(src, dst)
    inv_m = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(frame, m, img_size, flags=cv2.INTER_LINEAR)


    return warped, inv_m




def histogram(warped):
    """Function to get the histogram of the image and get the base points of the left and right lane lines

    Args:
        warped (Numpy array): input image

    Returns:
        index array of type int: returns the left and right lane base points along with the midpoint of the histogram 
    """
    hist = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(hist.shape[0] // 2)
    left_x_base = np.argmax(hist[:midpoint])
    right_x_base = np.argmax(hist[midpoint:]) + midpoint 

    
    return left_x_base, right_x_base, midpoint




def get_fitted_poly(img_shape, leftx, lefty, rightx, righty):
    """Function to fit the points in polynomial

    Args:
        img_shape (tuple): input image shape
        leftx (int): left nonzero x coordinate
        lefty (int): left nonzero y coordinate
        rightx (int): right nonzero x coordinate
        righty (int): right nonzero y coordinate

    Returns:
        int, int, array: returns the fitted pololygon points
    """
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


    return left_fitx, right_fitx, ploty




def get_poly_fit_points(image):
    """Function to get the poly points required for polynomial fitting

    Args:
        image (Numpy array): input image

    Returns:
        array, float, float, array: returns the optimal fit left and right points of the curved lines and images with the projected polygon fitting
    """
    margin = 100

    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])


    left_x_base,right_x_base, midpoint = histogram(image)
    out_img = np.dstack((image, image, image))

    window_ht = np.int(image.shape[0] // windows)

    nonZero = image.nonzero()
    nonZero_y = np.array(nonZero[0])
    nonZero_x = np.array(nonZero[1])

    current_left_x = left_x_base
    current_right_x = right_x_base
    left_lane_index, right_lane_index = [], []

    
    for window in range(windows):
        window_y_low = image.shape[0] - (window + 1) * window_ht
        window_y_high = image.shape[0] - window * window_ht
        left_window_x_low = current_left_x - margin
        left_window_x_high = current_left_x + margin
        right_window_x_low = current_right_x - margin
        right_window_x_high = current_right_x + margin
        cv2.rectangle(image, (left_window_x_low, window_y_low), (left_window_x_high, window_y_high), (255, 255, 255), 4) #draw the windows on left lane
        cv2.rectangle(image, (right_window_x_low, window_y_low), (right_window_x_high, window_y_high), (255, 255, 255), 4) #draw the windows on right lane
        good_left_inds = ((nonZero_y >= window_y_low) & (nonZero_y < window_y_high) & (nonZero_x >= left_window_x_low) & (nonZero_x < left_window_x_high)).nonzero()[0]
        good_right_inds = ((nonZero_y >= window_y_low) & (nonZero_y < window_y_high) & (nonZero_x >= right_window_x_low) & (nonZero_x < right_window_x_high)).nonzero()[0]
        left_lane_index.append(good_left_inds)
        right_lane_index.append(good_right_inds)
        if len(good_left_inds) > min_pixels:
            current_left_x = np.int(np.mean(nonZero_x[good_left_inds]))
        if len(good_right_inds) > min_pixels:
            current_right_x = np.int(np.mean(nonZero_x[good_right_inds]))

    left_lane_index = np.concatenate(left_lane_index)
    right_lane_index = np.concatenate(right_lane_index)


    left_x = nonZero_x[left_lane_index]
    left_y = nonZero_y[left_lane_index]
    right_x = nonZero_x[right_lane_index]
    right_y = nonZero_y[right_lane_index]

    left_index = (left_x, left_y)
    right_index = (right_x, right_y)

    # left_index, right_index, out_img = get_poly_fit_pred(image)
    leftx, lefty, rightx, righty = left_index[0], left_index[1], right_index[0], right_index[1]
    if ((len(leftx) == 0) or (len(rightx) == 0) or (len(righty) == 0) or (len(lefty) == 0)):
        out_img = np.dstack((image, image, image)) * 255
        optimal_left_fit = 0
        optimal_right_fit = 0
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_insdices = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_indices = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +right_fit[1] * nonzeroy + right_fit[2] + margin)))

        leftx = nonzerox[left_insdices]
        lefty = nonzeroy[left_insdices]
        rightx = nonzerox[right_indices]
        righty = nonzeroy[right_indices]

        # Get new fitted polynomials
        left_fitx, right_fitx, ploty = get_fitted_poly(image.shape, leftx, lefty, rightx, righty)

        ym_per_pix = 30 / 720  
        xm_per_pix = 3.7 / 650  

        # Calculate the curvature
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        y_eval = np.max(ploty)
        optimal_left_fit = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        optimal_right_fit = ((1 + ( 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        out_img = np.dstack((image, image, image)) * 255
        out_img[nonzeroy[left_insdices], nonzerox[left_insdices]] = [255, 0, 0]
        out_img[nonzeroy[right_indices], nonzerox[right_indices]] = [0, 0, 255]

        left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
        fit_points = np.hstack((left, right))
        
        out_img = cv2.fillPoly(out_img, np.int_(fit_points), (200, 255, 0))
        pts_center = np.array([np.transpose(np.vstack([((left_fitx+right_fitx)/2),ploty]))])
        cv2.polylines(out_img, np.int32([pts_center]), isClosed=False, color=(255,0,0), thickness=2)

    return out_img, optimal_left_fit, optimal_right_fit , image




def get_radius_of_curvature(optimal_left_fit, optimal_right_fit, imput_frame, res):
    """Function to get the radius of curvature

    Args:
        optimal_left_fit (float): left fit point
        optimal_right_fit (float): right fit point
        imput_frame (Numpy array): input image
        res (Numpy array): result image

    Returns:
        Numpy array: returns result image with the radius of curvature 
    """
    curvature = (optimal_left_fit + optimal_right_fit) / 2
    car_pos = imput_frame.shape[1] / 2
    center = (abs(car_pos - curvature)*(3.7/650))/100
    curvature = 'Radius of Curvature: ' + str(round(curvature)/1000) + ' km'
    center = 'vehicle is ' + str(round(center, 3)) + 'm away from center'
    cv2.putText(res, curvature, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(res, center, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return res




def predict_turn(image_center, right_lane_pos, left_lane_pos):
    """Function to predict the turn

    Args:
        image_center (float): centre of the warped image
        right_lane_pos (floaf): left lane point
        left_lane_pos (floaf): right lane point

    Returns:
        string: returns a string whichg is the predicted turn
    """
    lane_center = (left_lane_pos + right_lane_pos )/2
    # print(lane_center - image_center)
    if (lane_center - image_center < -50):
        return ("Turn left")
    elif (lane_center - image_center < 8):
        return ("Go straight")
    else:
    	return ("Turn right")





if __name__ == "__main__":
    cap = cv2.VideoCapture("Project_2/challenge.mp4")
    while cap.isOpened():

        ret, imput_frame = cap.read()
        if ret == False:
            break
            

        color_mask = apply_color_mask(imput_frame)
        warped, invM = get_birdseye_view(color_mask)
        out, optimal_left_fit, optimal_right_fit, image = get_poly_fit_points(warped)
        
        final_image = cv2.warpPerspective(out, invM, (imput_frame.shape[1], imput_frame.shape[0]), flags=cv2.INTER_LINEAR)
        

        res = cv2.addWeighted(final_image, 0.3, imput_frame, 0.7, 0)

        final = get_radius_of_curvature(optimal_left_fit, optimal_right_fit, imput_frame, res)

        image_center = int(warped.shape[1]/2)
        c = predict_turn(image_center, optimal_right_fit, optimal_left_fit)
        cv2.putText(res,c, (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # warped2 = cv2.resize(warped, (final.shape[0],final.shape[0])) #uncomment to visualize just the warped image
        final_image_out = np.concatenate((final, final_image), axis=0)
        cv2.imshow('asdfgvasd', final_image_out)


        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        
cap.release()
cv2.destroyAllWindows()