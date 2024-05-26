from ultralytics import YOLO
import supervision as sv                     # this library has the tracker
import pickle
import os
import sys 
import cv2
import numpy as np
sys.path.append('../')
from utilities import get_center_of_bbox, get_width_of_bbox

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack() 


    def detect_frames(self, frames):
        batch_size=20                 # instead of predicting on whole frame, we do it for 20 frames (Goes by 20,40...). So we won't go to memory issues
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)       # instead of predict we can use model.track, but we didn't use as GK is labelled as player sometime as we don't have large dataset. Therefore, we use predict now then override GK with player and then run tracker on it. 
            detections += detections_batch
        return detections


    def get_object_tracks(self, frames,read_from_stub=False, stub_path=None):

        # if it's true and stub_path exists, we load the tracks and return it. This prevents us from running everything in this func again
        # Deserialize the byte stream back to an object
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}          # this reverse the position of key:value pair of detection names to {person:0, .....}
            # print(cls_names)

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
                         
            # Track Objects - adds tracker object to the detection
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # for each track and frame, we append a dict. This dict has track id as key and bounding box as value
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()                  # 0 index is bounding box , then comes mask, confidence, class id.........
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}          #instead of track_id we put 1 ball



            # print(detection_supervision)  # to view supervision format
            # print(detection_with_tracks)            # to show how i track across frames

        #save this track object as a pickle. ie) Serialize the object to a byte stream
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f) 

        return tracks                                  # list of dictionaries  

    # this depends on the width of the object's bbox 
    def draw_ellipse(self,frame,bbox,color,track_id=None):        
        y2 = int(bbox[3])   # y2 is the bottom of the bounding box
        x_center, _ = get_center_of_bbox(bbox)
        width = get_width_of_bbox(bbox)

        cv2.ellipse(frame, center=(x_center,y2),axes=(int(width), int(0.30*width)), angle= 0.0, startAngle=-45, endAngle=225, color= color, thickness= 2, lineType= cv2.LINE_4)
        
        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            # for three or more digit numbers
            # if track_id > 99:
            #     x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        
        return frame

    def draw_circular_ball(self,frame,bbox,color):
        x,y = get_center_of_bbox(bbox)
        cv2.circle(frame, (int(x),int(y)), 7, color, 2)
        return frame
    
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-7,y-15],
            [x+7,y-15],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_annotations(self,video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
    
            # Draw ellipse on Players 
            for track_id, player in player_dict.items():     
                frame = self.draw_ellipse(frame, player["bbox"],(0,0,255), track_id)

            # Draw ellipse on referee 
            for _, referee in referee_dict.items():     
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))

            # Draw circular marker on ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_circular_ball(frame, ball["bbox"],(255,255,255))
            # Draw triangular marker on ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            output_video_frames.append(frame)

        return output_video_frames
