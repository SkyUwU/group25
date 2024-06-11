import numpy as np
import cv2
import classify

def reward_calculate(claw, time, img_bgr):

    if claw == True:
        claw_lower = np.array([14,12,11])
        claw_upper = np.array([28,15,13])
        
        img = cv2.resize(img_bgr,(640,360))
        img2 = img
        img = cv2.inRange(img, claw_lower, claw_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        img = cv2.dilate(img, kernel)
        img = cv2.erode(img, kernel)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        x_m = -1
        y_m = -1
        area_max = -1
        img = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2RGB)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if(w*h > area_max and area > 100):
                area_max = w*h
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                x_m = x+w/2
                y_m = y+h/2
        #print(x_m, y_m)
        saveFile = "C:/Users/user/Downloads/track/o/" + str(time) +".png"
        cv2.imwrite(saveFile, img)

        reward = 0
        end = False

        if x_m > 79 and x_m < 235 and y_m > 107 and y_m < 263:
            end = True
            if classify.winlose(img_bgr) == 1:
                reward = 10000
            else:
                reward = -1000
        
        return reward, end
    else:
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
            color = (0,0,255)
            if(area > 3000):
                x, y, w, h = cv2.boundingRect(contour)
                claw_x = x + w/2
                claw_y = y + h/2

        end = False
        min_dis = 9999
        if len(x_list)==0:
            end = True
        else:
            for k in range(len(x_list)):
                dis = ((x_list[k]-claw_x)**2 + (y_list[k]-claw_y)**2)**0.5
                min_dis = min(min_dis, dis)
        min_dis = round(min_dis, 3)
        reward = 80 - min_dis - time
        return reward, end