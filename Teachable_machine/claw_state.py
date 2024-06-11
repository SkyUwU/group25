import numpy as np
import cv2

def state_calculate(img_bgr):
    claw_lower = np.array([6,16,15])
    claw_upper = np.array([6,16,15])
    shadow_lower = np.array([73,32,87])
    shadow_upper = np.array([73,32,87])
    
    img = img_bgr
    img = cv2.resize(img,(640,360))
    
    p1 = np.float32([[202, 215],[433, 215],[533, 293],[103, 293]])
    p2 = np.float32([[0,0],[450,0],[450,500],[0,500]])

    q1 = np.float32([[80, 12],[558, 12],[435, 65],[194, 65]])
    q2 = np.float32([[0,500],[450,500],[450,0],[0,0]])
    m = cv2.getPerspectiveTransform(p1,p2)
    n = cv2.getPerspectiveTransform(q1,q2)
    output_p = cv2.warpPerspective(img, m, (450, 500))
    output_q = cv2.warpPerspective(img, n, (450, 500))
    img_p = output_p
    img_q = output_q
    
    output_p = cv2.inRange(img_p, shadow_lower, shadow_upper)
    output_q = cv2.inRange(img_q, claw_lower, claw_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    output_p = cv2.dilate(output_p, kernel)
    output_q = cv2.dilate(output_q, kernel)
    output_p = cv2.erode(output_p, kernel)
    output_q = cv2.erode(output_q, kernel)
    contours, hierarchy = cv2.findContours(output_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x_list = []
    y_list = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if(area > 100):
            x, y, w, h = cv2.boundingRect(contour)
            if h/w<2.4:
                x_list.append(x+w/2)
                y_list.append(y+w/2)
            elif h>80:
                x_list.append(x+w/2)
                y_list.append(y+w/2)

    contours, hierarchy = cv2.findContours(output_q, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    claw_x = 0
    claw_y = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if(area > 3000):
            x, y, w, h = cv2.boundingRect(contour)
            claw_x = x + w/2
            claw_y = y + h/2

    min_dis = 9999
    return_x = 9999
    return_y = 9999
    for k in range(len(x_list)):
        dis = ((x_list[k]-claw_x)**2 + (y_list[k]-claw_y)**2)**0.5
        min_dis = min(min_dis, dis)
        return_x = x_list[k]
        return_y = y_list[k]

    return claw_x, claw_y, return_x, return_y

    