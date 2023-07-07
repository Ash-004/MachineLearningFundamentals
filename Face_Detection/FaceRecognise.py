import cv2
import numpy as np
import os

dataset_path = "./data/"
faceData = []
labels = []
namemap = {}

classId = 0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        namemap[classId] = f[:-4]
        dataItem = np.load(dataset_path+f)
        m = dataItem.shape[0]
        faceData.append(dataItem)

        target = classId * np.ones((m,1))
        classId += 1
        labels.append(target)

XT = np.concatenate(faceData,axis=0)
yT = np.concatenate(labels,axis=0).reshape((-1,1))



def dist(p,q):
    return np.sqrt(np.sum((p - q)**2))

def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []

    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d, y[i]))

    dlist = sorted(dlist)
    dlist = np.array([tup[1] for tup in dlist[:k]])

    labels, cnts = np.unique(dlist, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

offset = 20
while True:

    success, img = cam.read()
    if not success:
        print("Reading Camera Failed")
        break

    faces = model.detectMultiScale(img, 1.3, 5)
    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = img[y-offset:y + h + offset, x-offset:x + w + offset]
        cropped_face=cv2.resize(cropped_face,(100,100))

        cv2.imshow("Image", img)

        classPredicted = knn(XT,yT,cropped_face.flatten())
        namePredicted = namemap[classPredicted]

        print(namePredicted)

        cv2.putText(img,namePredicted,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2,cv2.LINE_4)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Image",img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()