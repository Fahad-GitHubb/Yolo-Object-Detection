import cv2
import cvzone
import os
import shutil
import math
import logging
import time
import torch
from ultralytics import YOLO
from torchvision.transforms import functional as F

# One time execution of the foloowing
logging.basicConfig(level=logging.INFO, filename='events.log', filemode='w', format='%(asctime)s - %(message)s')

# Making my custom logger
myLogger = logging.getLogger(__name__) 
handler = logging.FileHandler('events.log')
format = logging.Formatter('%(asctime)s => %(message)s')
handler.setFormatter(format)
myLogger.addHandler(handler)


class Yolo_Predictions:
    def __init__(self, obj_model, face_model):
        # Load YOLOv8n model
        self.model = YOLO(obj_model)  
        self.face_model = YOLO(face_model)
        self.classes = self.model.names
        # Declaring the assets and suspects classes
        self.assets = [39, 62] # bottle, tv (laptop= 63, call phone = 67)
        self.suspects = [0]

        # Load MiDaS for depth estimation
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.midas.to('cpu')
        self.midas.eval()
        # MiDaS transformation
        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        self.tracking_ids = []
        self.start_time = []
        self.tracking_logs = []
    
        # For Face Detection
        self.face_id = []
        self.face_conf = []
        self.cap_done =[]

        # Cleaning the Images folder in the directory 
        folder = './images'
        shutil.rmtree(folder)
        os.mkdir(folder)


    def predictions(self, frame, old_frame):
        results = self.model.track(source=frame, verbose=False, tracker='botsort.yaml', persist=True, conf=0.2)
        boxes = results[0].boxes   
        asset_list = []
        suspect_list = []

        # Looping through all detected boundary boxes
        # Check here kay bhai boxes hein bhe ya nai
        if(boxes):  
            for box in boxes:
                points = box[0].xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                conf = box.conf[0]
                cls_name = self.classes[cls]
                
                # Checking if box.id was available
                if(box.id): box_id = int(box.id[0])
                else: box_id='NA'

                # Default Color
                Color = (0,0,0)

                if(box_id != 'NA' and box_id not in self.tracking_ids):
                    # New Object Entry 
                    self.tracking_ids.append(box_id)
                    entry_time = time.time()
                    self.start_time.append(entry_time)

                    # Log event 
                    myLogger.info(f'ID: {box_id} - {cls_name} - Object ENTERED and color changed to GREEN')
                    # Capture picture
                    img = old_frame.copy()
                    self.capture_event(img, points, box_id, 'GREEN')

                    # Person or some other obj
                    if (cls==0):
                        Color = (0,255,0) # Green
                    else:
                        Color = (255,0,0) # Blue
                        
                elif (box_id != 'NA' and box_id in self.tracking_ids):
                    # Already exits
                    index = self.tracking_ids.index(box_id)
                    # We check time
                    start_time = self.start_time[index]
                    duration = time.time() - start_time
                    # Then we decide event based on Time duration
                    if cls != 0:
                        Color = (255,0,0)
                    elif duration < 5:
                        Color = (0,255,0)
                    elif (duration >=5 and duration < 15):
                        Color = (0,165,255)
                        if(f'{box_id}O' not in self.tracking_logs):
                            myLogger.info(f'ID: {box_id} - {cls_name} - Object color changed to ORANGE')
                            img = old_frame.copy()
                            self.capture_event(img,points, box_id, 'ORANGE')
                            self.tracking_logs.append(f'{box_id}O')
                    else: # >15s
                        Color = (0,0,255)
                        if (f'{box_id}R' not in self.tracking_logs):
                            myLogger.info(f'ID: {box_id} - {cls_name} - Object olor changed to RED and ALARM triggered')
                            img = old_frame.copy()
                            self.capture_event(img,points, box_id, 'RED')
                            self.tracking_logs.append(f'{box_id}R')
                        
                    
                if(cls in self.assets): 
                    # Prepare frame for MiDaS
                    input_tensor = self.transform(frame).to('cpu')
                    # Depth estimation with MiDaS
                    with torch.no_grad():
                        depth_map = self.midas(input_tensor)
                        depth_map = torch.nn.functional.interpolate(
                            depth_map.unsqueeze(1),
                            size=old_frame.shape[:2],
                            mode='bicubic',
                            align_corners=False
                        ).squeeze()
                        depth_map = depth_map.cpu().numpy()
                        
                    # Calculate centroid of bounding box
                    x1,y1,x2,y2 = points
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    # Depth value at the centroid
                    depth_value = depth_map[cy, cx]

                    asset_list.append([cls, points, depth_value])
                    self.bbox_maker(old_frame, points, conf, cls_name, depth_value, Color, box_id)

                elif(cls in self.suspects): 
                    # Prepare frame for MiDaS
                    input_tensor = self.transform(frame).to('cpu')
                    # Depth estimation with MiDaS
                    with torch.no_grad():
                        depth_map =self.midas(input_tensor)
                        depth_map = torch.nn.functional.interpolate(
                            depth_map.unsqueeze(1),
                            size=old_frame.shape[:2],
                            mode='bicubic',
                            align_corners=False
                        ).squeeze()
                        depth_map = depth_map.cpu().numpy()

                    # Calculate centroid of bounding box
                    x1,y1,x2,y2 = points
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    # Depth value at the centroid
                    depth_value = depth_map[cy, cx]

                    suspect_list.append([cls,points,depth_value])
                    self.bbox_maker(old_frame, points, conf, cls_name, depth_value, Color, box_id)

                    # Display depth map
                    # depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    # depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)  # Optional color map
                    # cv2.imshow('Depth Estimation', depth_display)
                    # cv2.waitKey(0)
                else:
                    self.bbox_maker(old_frame, points, conf, cls_name, 0, Color, box_id)


                # Face detection if a person exists:
                if(cls==0): 
                    self.face_getter(frame=frame, old_frame=old_frame, id=box_id)

                
            for asset in asset_list:
                x1, y1, x2, y2 = asset[1]
                x1 += 10
                y1 += 10
                x2 += 10
                y2 += 10
                for suspect in suspect_list:
                    X1, Y1, X2, Y2 = suspect[1]
                    # Check if there's any overlap
                    if (x1 < X2 and x2 > X1 and y1 < Y2 and y2 > Y1):
                        print(f"Overlapping: {self.classes[suspect[0]]} with {self.classes[asset[0]]}" )
                        # Overlap kr rhay => ab check kro depth if kareeb then alarm else not
                        if(abs(int(asset[2])-int(suspect[2])))<60:
                            print("------EMERGENCYYYYYYYYYYYYYYYYYYYYYYYYYYYY!-------")
                        else:
                            print('-----No Worries!----')
            

    def bbox_maker(self, image, cords, conf, cls_name, depth,color, id='NA'):
        x1, y1, x2, y2 = cords
        cv2.rectangle(image, (x1, y1), (x2, y2), color,2)
        text = f'ID: ({id}) {cls_name}({conf * 100:.2f}%) Depth: {depth:.2f} units'
        cv2.putText(image, text, (x1 + 15, y1 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

    
    def face_getter(self, frame, old_frame, id):
        face_result = self.face_model.predict(frame,conf = 0.30, verbose=False)
        for info in face_result:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h,w = y2-y1,x2-x1
                cvzone.cornerRect(old_frame,[x1,y1,w,h],l=9,rt=3)

                # Checking Face
                if(id not in self.face_id):
                    # New face
                    self.face_id.append(id)
                    conf = box.conf[0].item()
                    self.face_conf.append(conf)
                    if(conf >= 0.65):
                        self.cap_done.append(True)
                        self.capture_event(old_frame,(x1-5, y1-5, x2+5, y2+5),id, "FACE")
                    else: 
                        self.cap_done.append(False)
                        self.capture_event(old_frame,(x1-5, y1-5, x2+5, y2+5),id, "FACE")
                        
                else:
                    index = self.face_id.index(id)
                    conf = box.conf[0].item()
                    if(self.cap_done[index]==False):
                        score = self.face_conf[index]
                        if(conf >= 0.65):
                            self.face_conf[index] = conf
                            self.capture_event(old_frame,(x1, y1, x2, y2),id, "FACE")
                            self.cap_done[index]=True
                        elif (conf >= score):
                            self.face_conf[index] = conf
                            self.capture_event(old_frame,(x1, y1, x2, y2),id, "FACE")



    
    def capture_event(self, image, bbox, id, event):
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2]) 
        y2 = int(bbox[3])
        cropped_image = image[y1:y2, x1:x2]
        cv2.imwrite(f'./images/Image[{id}][{event}].jpg',cropped_image)



# Initialize Yolo Object
yolo = Yolo_Predictions('./Models/yolov8n.pt', './Models/yolov8n-face.pt')

# Initliaze Video Frame
cap = cv2.VideoCapture(0)   
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame to RGB for MiDaS and YOLO
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    yolo.predictions(rgb_frame, frame)
    

    cv2.imshow('Results', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
