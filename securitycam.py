#first install opencv from terminal
#pip3 install opencv-python

import cv2
import time
import datetime

#access to webcam and view video on python, 0 because it is the only cam I have
cap = cv2.VideoCapture(0)

#define pre trained frontal_face classifier and body classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

#however, note that these classifiers requires gray scale image input, so we will turn our webcam 
#images to gray scale in the following lines

#display the captured frames on he screen (so it will be seen as a wideo)
#underscore is just a placeholder
#"Camera" is the name of the screen which will show us the video, we can name it as we wish

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5
#actually this is about for how many more second it will record after there is no more detected face,
#after when recording will stop

frame_size = (int(cap.get(3)), int(cap.get(4)))#framesize of the recording
fourcc = cv2.VideoWriter_fourcc(*"mp4v")#(type of the output file which is video)


while True:
    _, frame = cap.read()

    #this well be our image with which we will detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)
    #these will give us all the faces on image, 1.3 the speed and accuracy parameter, should be between
    #1.1 and 1.5
    #5 is about the minimum number of neighbours ot requires to detect a face, if it cant detect faces you can 
    #lower it, but it shoukd be between 3 and 6, if it detects thing that arent faces make it 6

    #draw a rectangle on faces (on frame image, not gray, gray is only used for detection)
    
    #for (x, y, width, height) in faces:
    #    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
    
    #x is top left, y is bottom right positions of the face, so here we are drawing the shape of 
    #the rectangle here (x + width, y + height)
    #(255, 0, 0) this is the colour of the rectangle, which is blue (as we are using BGR, not RGB)
    #3 is the line thickness
    #I commented this as I dont need it, I just need to start recording when it detects a face
    #and this is how we do it:

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")#this for output name
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)#20 is frame rate
            print("Started recording!")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stop recording!")
        else:
            timer_started = True
            detection_stopped_time = time.time()#refers to now

    #if there is detection, dont start timer_started, so do not end the video, 
    #else part is, if no previous detection, now there is detection so continue recording
    #so this prevents us from having very short seconds of videos, as the face may disappear,
    #or in case it could not detect the face
    #otherwise we would need a video for every frame
    
    if detection:
        out.write(frame)

    #write the recorded video
    
    #the logic is, when a body or face is detected, it will save it to the faces and/or bodies which are lists,
    #we check the len, and it will start recording if a face or body is detected

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break
    #this is for us to be able to quit the pragram loop
    #so "q" key will stop the program

out.release()
#this will save the video once we quit
cap.release()
#this is to release the camera when we quit this program

cv2.destroyAllWindows()
#this will destroy the screen

#now that we have access to our camera, we want to detect faces and bodies

#we will use haar cascade, which is pre trained by opencv to detect faces or whatever, but we will use face