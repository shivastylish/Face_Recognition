import cv2
import glob
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
import numpy as np
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
import os



"""
IF YOU GET THIS ERROR: ModuleNotFoundError: No module named 'keras.engine.topology'
Change the import from
from keras.engine.topology import get_source_inputs
to
from keras.utils.layer_utils import get_source_inputs
in YOUR_ENVIRONMENT/Lib/site-packages/keras_vggface/models.py
"""
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def faceDetectionMTCNN(img):

    # Exercise 6 - Step 2 - The faceDetectionMTCNN method receives as input an image
    # and returns a list of image patches containing the cropped faces

    croppedFaces = []

    # TODO - Exercise 6 - Step 2 - Convert the image from BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    # TODO - Exercise 6 - Step 2 - Create the detector, using the default weights
    detector = MTCNN()



    # TODO - Exercise 6 - Step 2 - Detect faces in the image
    results = detector.detect_faces(rgb_img)



    # TODO - Exercise 6 - Step 2 - Save images with the cropped faces in the croppedFaces list
    for result in results:
        x, y, width, height = result['box']
        cropped_face = rgb_img[max(y, 0):y + height, max(x, 0):x + width]
        croppedFaces.append(cropped_face)




    return croppedFaces
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def extractCNNFeatures(croppedFaces):

    #Exercise 6 - Step 3 - Extract CNN features from cropped face images

    feats = []

    #TODO - Exercise 6 - Step 3 - Create a vggface model object
    model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')



    resizedFaces = []
    for crFace in croppedFaces:
        #TODO - Exercise 6 - Step 3 - Resize the image to (224, 224)
        resized = cv2.resize(crFace, (224, 224))
        resizedFaces.append(resized)


	#TODO - Exercise 6 - Step 3 - Convert resizedFaces to a float32 numpy array of size (n, 224, 224, 3)
    resizedFaces_np = np.asarray(resizedFaces, dtype=np.float32)


	#TODO - Exercise 6 - Step 3 - Pre-process the face images to the standard format accepted by VGG16
    preprocessed_faces = preprocess_input(resizedFaces_np, version=2)  


	# TODO - Exercise 6 - Step 3 - Extract the low level features by forwarding the images through the CNN
    feats = model.predict(preprocessed_faces)


    return feats
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def personReID(featsDict, threshold=0.7):
    #TODO - Exercise 6 - Step 5 - This function find the images that contain the same character

    # Iterate over each pair of images to compare their features
    keys = list(featsDict.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            # Calculate the cosine similarity between feature vectors of two images
            for vec_i in featsDict[keys[i]]:
                for vec_j in featsDict[keys[j]]:
                    # Ensure vec_i and vec_j are 1-D
                    vec_i = vec_i.flatten()
                    vec_j = vec_j.flatten()
                    similarity = 1 - cosine(vec_i, vec_j)

            # If similarity score is above the threshold, it indicates a potential match
                    if similarity > threshold:
                        print(f"Images {keys[i]} and {keys[j]} contain the same character with a similarity score of {similarity:.2f}")


    return
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
def main():

    featsDict = {}

    # Get the image names from FaceReID directory
    imgfiles = glob.glob("./FaceReID/*.jpg")

    for imgf in imgfiles:

        # TODO - Exercise 6 - Step 1 - Open the images one by one from the FaceReID folder
        img = cv2.imread(imgf)

        # TODO  - Exercise 6 - Step 2 - Call the faceDetectionMTCNN method
        croppedFaces = faceDetectionMTCNN(img)

        # TODO - Exercise 6 - Step 3 - Call the extractCNNFeatures method
        feats = extractCNNFeatures(croppedFaces)

        #TODO - Exercise 6 - Step 4 - Store the face features returned by the extractCNNFeatures method
        # into a dictionary (featsDict) where the key represents the image name,
        # while the values are given by the list of face features
        featsDict[os.path.basename(imgf)] = feats

    #TODO - Exercise 6 - Step 5 - Call the function personReID
    personReID(featsDict, threshold=0.5)

    return
########################################################################################################################
########################################################################################################################

########################################################################################################################
if __name__ == "__main__":
    main()