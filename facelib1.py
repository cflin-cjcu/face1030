import cv2
from face_lib import face_lib
FL = face_lib()
img = cv2.imread('cflin1.png')
faces = FL.get_faces(img)
cv2.imshow('ttt', faces[0])
cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("Error in opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    cflin_exist, no_of_faces = FL.recognition_pipeline(
        frame, img, only_face_gt=True)
    if no_of_faces != 0:
        _, faces_locations = FL.faces_locations(frame)
        x, y, w, h = faces_locations[0]
        if cflin_exist == True:
            cv2.putText(frame, 'cflin', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 200, 40), 2)
        else:
            cv2.putText(frame, 'not cflin', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 200, 40), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('cflin', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
