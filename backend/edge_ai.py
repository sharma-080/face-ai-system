import cv2
from .recognition import recognize_face

def process_frame(frame):

 faces = []

 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

 detected = face_cascade.detectMultiScale(gray,1.3,5)

 for (x,y,w,h) in detected:

    face = frame[y:y+h, x:x+w]

    name,confidence = recognize_face(face)

    faces.append({
        "name":name,
        "confidence":confidence,
        "box":(x,y,w,h),
        "face":face
    })
 return faces