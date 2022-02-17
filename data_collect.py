import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

count = 0

face_dataset = []
dataset_path = '.\\data\\'
file_name = input("Enter the name:")

while True:
	ret,frame = cap.read()

	if ret == False:
		continue
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces, key=lambda f:f[2]*f[3], reverse=True)

	for face in faces:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(204,102,0),2)

		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		count += 1
		if count%10==0:
			face_dataset.append(face_section)

	cv2.imshow("Frame", frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break
face_dataset = np.asarray(face_dataset)
face_dataset = face_dataset.reshape((face_dataset.shape[0],-1))

np.save(dataset_path+file_name+'.npy',face_dataset)

cap.release()
cv2.destroyAllWindows()
