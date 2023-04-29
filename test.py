"""
Load the trained network, extract each frame from a video, perform face detection with yoloface
pass each face detected to the network and draw a rectangle around the face with the prediction of the actor,
in the end reconstruct the video.
"""
import torch
import cv2
import torch.nn.functional as F
from lib import VGGFace
import numpy as np
from yoloface import face_analysis

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VGGFace().double().to(device)
model_dict = torch.load('Weights_trained/vggface_trained.pth', map_location=lambda storage, loc: storage)
model.load_state_dict(model_dict)
names = [line.rstrip('\n') for line in open('data/names.txt')]
print(len(names))

face=face_analysis()
video_capture = cv2.VideoCapture("data/Vi presento Joe Black.mp4")
img_array = []
#i = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if ret is None or frame is None:
      break

    height, width, layers = frame.shape
    size = (width, height)
    
    img,box,conf=face.face_detection(frame_arr=frame, model='tiny')
	
    #SOVRAPPONGO LE DETECTION ALL'IMMAGINE
    for (x, y, w, h) in box:
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        actor = img[y:y+h+80, x:x+w]
        actor = cv2.resize(actor, (224, 224))
        actor = torch.Tensor(actor).permute(2, 0, 1).view(1, 3, 224, 224).double().to(device)
        predictions = F.softmax(model(actor), dim=1)
        score, index = predictions.max(-1)
        nome = '{}, prob. {}'.format(names[index], score.item())

        cv2.rectangle(frame, (x, y), (x + w, y + h + 80), (0, 255, 0), 4)
        cv2.putText(frame, nome, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    #img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_array.append(frame)
    #if i > 200:
      #break
    #i+=1
out = cv2.VideoWriter('data/joe_black_yolo_names.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20, size)
for i in range(len(img_array)):
    out.write(img_array[i])
# When everything is done, release the capture
out.release()
video_capture.release()
cv2.destroyAllWindows()

