import cv2
import glob
import os
from mtcnn.mtcnn import MTCNN
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def faceDetectionViolaJones(img):

    #Exercise 1 - The faceDetectionViolaJones method receives as input an image and returns the bounding boxes
    #of all the detected faces

    bboxes = []

    #TODO - Exercise 1 - Convert the image to a grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    #TODO - Exercise 1 - Load the pre-trained cascade classifier model called haarcascade_frontalface_default.xml
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



    #TODO - Exercise 1 - Perform face detection using the defaul parameters (scaleFactor = 1.1, minNeighbors = 3)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)



    #TODO - Exercise 1 - ViolaJones detector returns bounding boxes as [x1, y1, w, h].
    # Convert them to a format [x1, y1, x2, y2]
    for (x, y, w, h) in faces:
        bboxes.append([x, y, x + w, y + h])


    print(bboxes)
    return bboxes
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def drawDetectedFaces(img, bboxes):

    #Exercise 2 - Draw a bounding box for each detected face on the image

    for box in bboxes:

        #TODO - Exercices 2 - Draw a rectangle over the image img
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)



    #TODO - Exercise 2 - Display the marked image
    cv2.imshow('Detected Faces', img)



    #TODO - Exercise 2 - Keep the window open until we press a key
    cv2.waitKey(0)



    #TODO - Exercise 2 - Close the window
    cv2.destroyAllWindows()



    return
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def computeIntersectionOverUnion(box, box_GT):

    #Exercise 3 - Step 3 - This function computes the intersection over union score (iou)
    iou = 0

    #TODO - Exercise 3 - Step 3 - Compute the rectangle resulted by the intersection of the two bounding boxes
    # This should be specified in the following format [x1, y1, x2, y2]

    xi1 = max(box[0], box_GT[0])
    yi1 = max(box[1], box_GT[1])
    xi2 = min(box[2], box_GT[2])
    yi2 = min(box[3], box_GT[3])
    rectInters = [xi1, yi1, xi2, yi2]




    #TODO - Exercise 3 - Step 3 - Compute the area of rectInters (rectIntersArea)
    width = max(xi2 - xi1, 0)
    height = max(yi2 - yi1, 0)
    rectIntersArea = width * height



    #TODO - Exercise 3 - Step 3 - Compute the area of the box (boxArea)
    boxArea = (box[2] - box[0]) * (box[3] - box[1])




    # TODO - Exercise 3 - Step 3 - Compute the area of the box_GT (boxGTArea)
    boxGTArea = (box_GT[2] - box_GT[0]) * (box_GT[3] - box_GT[1])



    # TODO - Exercise 3 - Step3 - Compute the union area (unionArea) of the two boxes
    unionArea = boxArea + boxGTArea - rectIntersArea




    #TODO - Exercise 3 - Step 3 - Compute the intersection over union score (iou)
    if unionArea > 0:
        iou = rectIntersArea / unionArea

    return iou
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def compareAgainstGT(imgf, bboxes):

    #Exercise 3 - This function compare the list of detected faces against the ground truth

    d_faces = 0         #the number of correctly detected faces
    md_faces = 0        #the number of missed detected faces
    fa = 0              #the number of false alarms

    bboxes_GT = []

    #TODO - Exercise 3 - Step 1 - Open the file with the ground truth for the associated image (imgf)
    # and read its content
    with open(imgf.replace('.jpg', '_GT.txt'), 'r') as f:
        lines = f.readlines()



    #TODO - Exercise 3 - Step 2 - Save the bounding boxes parsed from the GT file into the bboxes_GT list
    for line in lines:
        x1, y1, x2, y2 = line.strip().split()
        bboxes_GT.append([int(x1), int(y1), int(x2), int(y2)])



    #TODO - Exercise 3 - Step 4 - Perform the validation of the bboxes (detected automatically)
    # against the bboxes_GT (annotated manually). In order to verify if two bounding boxes overlap it is necessary
    # to define another function denoted "computeIntersectionOverUnion(box, box_GT)"
    iou_threshold = 0.2
    matched_gt_indices = set()

    for box in bboxes:
        iou_scores = [computeIntersectionOverUnion(box, box_GT) for box_GT in bboxes_GT]
        
        # Find the max IoU score for this predicted box
        max_iou = max(iou_scores) if iou_scores else 0
        max_index = iou_scores.index(max_iou) if max_iou > iou_threshold else -1

        if max_iou > iou_threshold:
            d_faces += 1
            matched_gt_indices.add(max_index)
        else:
            fa += 1

    md_faces = len(bboxes_GT) - len(matched_gt_indices)  # Missed detections are GT boxes not matched


    #Exercise 3 - Display the scores
    print("The scores for image {} are:".format(imgf))
    print("   - The number of correctly detected faces: {}".format(d_faces))
    print("   - The number of missed detected faces: {}".format(md_faces))
    print("   - The number of false alarms: {}".format(fa))


    return d_faces, md_faces, fa
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def faceDetectionMTCNN(img):

    # Exercise 4 - The faceDetectionMTCNN method receives as input an image
    # and returns the bounding boxes of all the detected faces

    bboxes = []

    #TODO - Exercise 4 - Convert the image from BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    #TODO - Exercise 4 - Create the detector, using the default weights
    detector = MTCNN()



    #TODO - Exercise 4 - Detect faces in the image
    results = detector.detect_faces(rgb_img)



    #TODO - Exercise 4 - Save all the faces bounding boxes in the bboxes list
    for result in results:
        x, y, w, h = result['box']


    #TODO - Exercise 4 - MTCNN detector returns bounding boxes as [x1, y1, w, h].
    # Convert them to a format [x1, y1, x2, y2]
        bboxes.append([x, y, x + w, y + h])



    print(bboxes)
    return bboxes
########################################################################################################################
########################################################################################################################
def faceDetectionInVideos(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    # Define the codec and create VideoWriter objects for each method
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_viola_jones = cv2.VideoWriter('videoFaces_Detected_ViolaJones.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    out_mtcnn = cv2.VideoWriter('videoFaces_Detected_MTCNN.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    # Initialize the face detectors
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detector = MTCNN()
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Viola-Jones detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_vj = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            for (x, y, w, h) in faces_vj:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            out_viola_jones.write(frame)
            
            # MTCNN detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_mtcnn = detector.detect_faces(frame_rgb)
            for result in faces_mtcnn:
                x, y, w, h = result['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            out_mtcnn.write(frame)
            
            # Display the frame
            cv2.imshow('Frame', frame)
            
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release everything if job is finished
    cap.release()
    out_viola_jones.release()
    out_mtcnn.release()
    cv2.destroyAllWindows()




########################################################################################################################
########################################################################################################################
def main():

    # Read images from GT_FaceImages directory
    imgfiles = glob.glob("./GT_FaceImages/*.jpg")
    for imgf in imgfiles:
        img = cv2.imread(imgf)


        #TODO - Exercise 1 - Call the faceDetectionViolaJones method OR Exercise 4 - Call the faceDetectionMTCNN method
        # To use Viola-Jones, uncomment the next line:
        bboxes = faceDetectionViolaJones(img)
        
        # To use MTCNN, uncomment the next line (ensure MTCNN is installed and imported):
        #bboxes = faceDetectionMTCNN(img)


        #TODO - Exercise 2 - Call the drawDetectedFaces method
        drawDetectedFaces(img, bboxes)



        #TODO - Exercise 3 - Call the compareAgainstGT method
        compareAgainstGT(imgf, bboxes)

        #detect faces in video
        #faceDetectionInVideos('videoFaces.mp4')


    return
########################################################################################################################
########################################################################################################################

########################################################################################################################
if __name__ == "__main__":
    main()