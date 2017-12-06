#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2, os
import numpy as np
from PIL import Image


class FaceRecognizer: 

    def get_images(self, path, faceCascade):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
        images = []
        labels = []      

        for image_path in image_paths:
            gray = Image.open(image_path).convert('L')
            image = np.array(gray, 'uint8')
            subject_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        
            faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(subject_number)
                cv2.imshow("", image[y: y + h, x: x + w])
                cv2.waitKey(50)
        return images, labels

    def __call__(self):
        path = './faces'

        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)

        images, labels = self.get_images(path, faceCascade)
        cv2.destroyAllWindows()

        recognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8,123)

        recognizer.train(images, np.array(labels))

        path = './recogn'
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]       

        for image_path in image_paths:
            gray = Image.open(image_path).convert('L')
            image = np.array(gray, 'uint8')
            faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])

                number_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        
                if number_actual == number_predicted:
                    print ("{} is Correctly Recognized with confidence {}".format(number_actual, conf))
                else:
                    print ("{} is Incorrect Recognized as {}".format(number_actual, number_predicted))

                cv2.imshow("Recognizing Face", image[y: y + h, x: x + w])
                cv2.waitKey(1000)
                
if __name__ == "__main__":
    faceRecognizer = FaceRecognizer()()