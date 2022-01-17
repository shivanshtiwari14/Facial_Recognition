import cv2
import numpy as np
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier(r"C:\Users\shiva\Desktop\haarcascade_frontalface_alt.xml")
data_path = r"C:\Users\shiva\Documents\pycharming/"
skip = 0
face_data = []
file_name = input("enter your name")
while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces , key = lambda f : f[2]*f[3])
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),5)

        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        if face_section is not None:
            face_section = cv2.resize(face_section,(100,100))

        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))
        skip += 1
    try:
        cv2.imshow("frame", frame)
        cv2.imshow("face_section",face_section)
    except:
        cv2.imshow('frame',frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(data_path+file_name+".npy",face_data)
print("Data saved successfully")
cap.release()
cv2.destroyAllWindows()
