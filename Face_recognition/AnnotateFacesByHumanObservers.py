import cv2
from time import time
import glob
import os
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
#Define some global variables
boxes = []
singleBox = [-1, -1, -1, -1]
img = None
flg_mouse_clicked = False
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def on_mouse(event, x, y, flags, params):
    global img
    global flg_mouse_clicked
    global singleBox
    global boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        print ('Start Mouse Position: '+str(x)+', '+str(y))
        singleBox[0] = x
        singleBox[1] = y
        flg_mouse_clicked = True

    if event == cv2.EVENT_MOUSEMOVE and flg_mouse_clicked == True:
        clone = img.copy()
        cv2.rectangle(clone, pt1=(singleBox[0], singleBox[1]), pt2=(x, y), color=(0, 255, 255), thickness=1)
        cv2.imshow("Image", clone)

    if event == cv2.EVENT_LBUTTONUP:
        print ('End Mouse Position: '+str(x)+', '+str(y))
        singleBox[2] = x
        singleBox[3] = y
        flg_mouse_clicked = False

        if cv2.waitKey(0) == ord('k') and singleBox[2] != -1:
            boxes.append(singleBox.copy())
            cv2.rectangle(img, pt1=(singleBox[0], singleBox[1]), pt2=(x, y), color=(0, 255, 255), thickness=1)
            singleBox = [-1, -1, -1, -1]
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def main():

    global img
    global boxes

    #Create the window for annotation
    cv2.namedWindow('Image')

    #Calling the mouse click event
    cv2.setMouseCallback("Image", on_mouse)

    #Read the name of the images from GT_FaceImages folder
    imgfiles = glob.glob("./GT_FaceImages/*.jpg")

    for imgf in imgfiles:
        img = cv2.imread(imgf)

        while(True):
            cv2.imshow("Image", img)

            if cv2.waitKey(0) == ord('n'):
                file_name, file_extension = os.path.splitext(imgf)
                f = open(file_name + "_GT.txt", "w")
                for box in boxes:
                    f.write(str(box[0]) + " "+str(box[1]) + " "+str(box[2]) + " "+str(box[3]) +"\n")
                f.close()
                boxes = []
                print ("Go to next image!")
                break
                
########################################################################################################################
########################################################################################################################



########################################################################################################################
if __name__ == "__main__":
    main()
########################################################################################################################
########################################################################################################################
