Downloading the dataset from https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data

https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/
https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
https://pytorch.org/get-started/locally/

predict the input video using the model. we cans see that the ball is not predicted in every case and we can also see that it detects every other person also. Therefore, we need to custom train the model with custom dataset. 
We download the dataset from roboflow for the player detection. (https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/9)
Download for specfic yolo version.
Train test val images are similar type as the input video we have. 


shutil libarary helps to move files and folder.
We move the test train val folder into another football-player-detection-1 as this is a requirement for the ultralytics model.

In order to train we can do it on google colab for the GPU. Then train the model and then download the best, last weights.
Then we use the best.pt for the inference of our input video, it genralizes well.

But the boundingbox and labels are too big. Therefore, i change this with different annotating.

Organise the functions

Tracking function
- For each frame, the bounding box's coordinates of the same object changes and we can't bounding box belongs to one object. We can say the closest cox in the previous one and current one belongs to the same object. 
- We want to track this. But closest one approach doesn't work. Tracking is nothing but to say where was the object in the previous frame and where in the current frame. 
- We are going to use white tracker, we need bounding boxes and tracker matches it to an ID


print(detection_supervision)  # to view supervision format
    - for  last frame
    Detections(xyxy=array([[      381.6,      595.09,      420.82,      681.05],                         # detection in xyxy format for boxes
       [     360.56,      315.57,      386.26,       381.4],
       [     302.11,      430.56,      334.06,      506.83],
       [     1079.5,      336.61,      1106.4,      395.85],
       [     1528.7,      476.52,      1583.1,      546.13],
       [     1436.5,      230.24,        1460,      286.38],
       [     359.84,       448.9,      388.47,      529.99],
       [     719.03,      299.29,      744.68,      361.23],
       [     1533.8,      304.87,      1565.5,      367.02],
       [     1127.5,      266.19,      1153.4,      320.77],
       [      444.4,      196.89,       465.1,      249.82],
       [     681.34,      729.46,      726.57,      828.14],
       [     1204.7,      262.45,      1229.9,       327.7],
       [     916.01,      347.57,      939.77,      415.68],
       [     911.88,      233.58,       933.2,      285.29],
       [     1062.2,       276.7,      1096.2,      326.41],
       [     896.98,         385,      929.41,      440.16],
       [     423.02,      274.62,       445.8,       334.9],
       [     389.46,      300.04,      418.58,      364.78],
       [     850.23,      239.41,      871.66,      294.11],
       [     658.46,         253,      679.21,       307.2],
       [     855.24,       381.4,      877.67,      452.19],
       [     1520.5,      359.04,      1533.1,      370.68]], dtype=float32), mask=None, confidence=array([    0.93308,     0.92365,     0.92282,     0.92012,     0.91975,     0.91713,     0.91482,     0.91465,      0.9124,      0.9106,     0.91042,     0.90831,      0.9072,     0.90203,     0.90048,     0.89573,     0.88988,     0.88358,     0.88245,     0.87712,     0.86496,     0.85677,     0.55974],
      dtype=float32), class_id=array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0]), tracker_id=None, data={'class_name': array(['player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'referee', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'referee', 'ball'], dtype='<U7')})

      class_id=array has the class id for detected bounding boxes

    print(detection_with_tracks)            # to show how i track across frames
    output in last frame:   tracker_id=array([14,  9,  1, 15, 12, 20,  2,  8,  7,  4, 17,  3, 11, 10, 24,  6, 19, 13, 21, 18,  5, 16])
    - in first frame, track id 1 is bounding box 1             output in 1st frame: tracker_id=array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    - for the initial frames it might be in ascending order



    We don't need to track the ball, as we only have one ball and we only need one bounding box


Let's see what this func returns as tracks
- tracks is list of dicts

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        tracks={
            "players":[
                {0:{"bbox":[0,0,0,0]},1:{"bbox":[0,0,0,0]}},                    # frame 0 has track id 0, track id 1, track id has bboxes for each track id 
                {1:{"bbox":[0,0,0,0]},1:{"bbox":[0,0,0,0]}},                    # frame 1 has track id 0, track id 1, track id has bboxes for each track id 

            ],
            "referees":[],
            "ball":[]
        }




To save the results, we save this track object as a pickle. 

Now we draw circular annotation,
    take each frame and copy it
    take the objects of the tracks, ie players_dict,ball_dict and referee_dict 

Similarly drawn circular and triangular annotation on the ball.


Split the team based on color classification
    - we only need the T-shirt, therefore crop the top part only
    - We can't take the average color as the background is dominant, therefore we segment out the green background using KMeans classification
    - Then we divide and assign team index label to two teams, for this again KMeans classification is used


PROBLEM SEEN:
    - there's some frames where the ball is not detected and we get a flickering detection of the ball
    - when the bounding boxes of different player coincide and pass through, the track id differs and the model detects, assigns a new object and it's is not tracked properly


### Ball Interpolation
#### Problem task
- there's some frames where the ball is not detected and we get a flickering detection of the ball

We use interpolate function from pandas to find the missing detections of the ball. It gives us a decent result, as there are some lags noted in the output. But this generalizes well to our problem.

### Player ball assigner
- we annotate the player who has the ball by calculating the distance between the ball and the player's foot. We set a threshold and see if the distance is below the threshold to draw a red triangle on the player.

### Ball possession percentage
- this is calculated by finding the ratio of frames assigned to each team and total number of frames assigned to both teams

### Perspective Transformation
- helps us to know how much a player has moved in unit measurements in real world
- a classic example of this if to transform the page on table to only select the page and transform it so that it appears as a top view of the image
