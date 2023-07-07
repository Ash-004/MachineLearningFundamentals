import cv2
import numpy as np
from cv2 import COLOR_BGR2GRAY

cam = cv2.VideoCapture(0)
fileName = input("Enter the name of the person: ")
dataset_path = "./data/"
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
offset = 20
cnt = 0
faceData = []
skip = 0

while True:

    success, img = cam.read()
    grayImg = cv2.cvtColor(img, COLOR_BGR2GRAY)
    if not success:
        print("Reading Camera Failed")
        break

    faces = model.detectMultiScale(img, 1.3, 5)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3])
        face = faces[-1]
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = img[y-offset:y + h + offset, x-offset:x + w + offset]
        cropped_face=cv2.resize(cropped_face,(100,100))

        cv2.imshow("Image", img)
        cv2.imshow("cropped_face", cropped_face)

        skip+=1
        if (skip%10==0):
            faceData.append(cropped_face)
            print("saved so far", str(len(faceData)))


    key = cv2.waitKey(1)
    if key == ord('q'):
        break
faceData = np.asarray(faceData)
faceData=np.reshape(faceData,(len(faceData),-1))
file = dataset_path+fileName+'.npy'
np.save(file,faceData)
print("data saved"+file)
cam.release()
cv2.destroyAllWindows()
