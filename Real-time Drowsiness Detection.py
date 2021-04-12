import os
import cv2 as cv
from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

labels = []
images = []
dir_ = "MRL"
files = os.listdir(dir_)
for filename in files:
    img = dir_+"/"+filename
    img = cv.imread(img)
    img = cv.resize(img, (100,100))
    img = img/255.0
    images.append(img)
    if filename[-12] == "1":
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=1)
images = np.array(images)

model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(100,100,3)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

model.summary()

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size= 0.2, random_state= 47)

model.fit(
    X_train,
    Y_train,
    epochs = 30,
    batch_size = 64,
    validation_data = (X_test, Y_test)
)

model.save("My_model.h5")

model = load_model("my_model.h5")

cap = cv.VideoCapture(0)

face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye = cv.CascadeClassifier("haarcascade_eye.xml")

while True:

    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.2, 3)

    for x,y,w,h in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]

        eyes = eye.detectMultiScale(gray)

        for ex,ey,ew,eh in eyes:

            roi_eye = frame[ey:ey+eh, ex:ex+ew]
            roi_eye = cv.resize(roi_eye, (100,100))
            roi_eye = np.expand_dims(roi_eye, axis=0)

            pred = model.predict(roi_eye)

            if pred == 1.:
                cv.putText(frame, "Open eyes", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            elif pred == 0.:

                cv.putText(frame, "Close eyes", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2)
            else:
                continue

    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv.destroyAllWindows()