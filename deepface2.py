from deepface import DeepFace
import cv2
# In VideoCapture object either Pass address of your Video file
# Or If the input is the camera, pass 0 instead of the video file
imgpath = "cflin.jpg"
cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
    print("Error in opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        result = DeepFace.verify(
            frame, imgpath, model_name='Facenet', enforce_detection=False)
        print(result)
        if result['verified'] == True:
            print("這二張圖片是同一個人")
        else:
            print("這二張圖片不是同一個人")
        # Press esc to exit
        if cv2.waitKey(10) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
