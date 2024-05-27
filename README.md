# Football_AI_object_detection_project

This project aims to detect and track football players, referees, and balls using YOLO, classify players by team colors with Kmeans, measure ball possession, track player movement with optical flow and perspective transformation, and calculate player speed and distance covered.

## **The following tasks for completed:**
1. Object detection model using YOLOv8 for detection and tracking
2. Pixel segmentation and clustering using KMeans for team classification

# **1. Object detection model using YOLOv8 for detection and tracking**

![YOLOv8 Integrations Banner](https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png)

## Ultralytics YOLOv8 provides these following modes:
1.   Train
2.   Val
3.   Predict
4.   Track
5.   Benchmark

## Ultralytics YOLOv8 supports these following computer vision tasks:
1.   Detection
2.   Segmentation
3.   Classification
4.   Oriented object detection
5.   Keypoints detection

## **Annotation**

The YOLO format for annotating the labels is followed. This is nothing but a txt file with coordinates for rectangular bounding boxes with class index.

    0 0.49375 0.484375 0.2875 0.4078125

The first index (0) denotes the class of the sample and the rest four denotes the 4 points of bounding box. This is a horizontal bounding box and not a oriented bounding box.

![Bounding boxes](Horizontal and oriented bounding box.png)

Here i used x1,y1,x2,y2 as our horizontal bounding box.

## **YOLO Architecture**

YOLO architecture is similar to GoogleNet. As illustrated below, it has overall 24 convolutional layers, four max-pooling layers, and two fully connected layers.

![YOLO Architecture](https://images.datacamp.com/image/upload/v1664382694/YOLO_Architecture_from_the_original_paper_ff4e5383c0.png)

![YOLOv8 Architecture](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*DeULH1Huz5zkny1aO_WUGQ.png)

**( [YOLO Original Paper](https://arxiv.org/pdf/1506.02640) )**

**Refer:**

https://medium.com/@juanpedro.bc22/detailed-explanation-of-yolov8-architecture-part-1-6da9296b954e
https://www.datacamp.com/blog/yolo-object-detection-explained
        
## Training
As i don't have a GPU, i ran the training of the model in Google colab with their T4 GPU.

Then train the model and then download the best, last weights.

Then the best.pt is used for the inference of our input video.best.py is nothing but the best model with best performance during training,that is low loss and high mAP / accuracy. This genralizes well on unseen new data.

https://drive.google.com/file/d/1LZMhpWz51GE3PrzXKrNNvpb5j-TOmZbK/view?usp=sharing

## Tracking
This is implemented in the ( [tracker.py](https://github.com/Vigneshwar-KR/Football_AI_object_detection_project/tree/main/trackers) ), where the end goal is to assign and store each bounding box with the tracker id for every frame, which enables us to track each object. 

Now annotation the objects (Players, referee and ball) is done. We use different annotations for different objects.


# 2. Pixel segmentation and clustering using KMeans for team classification

This is used to split the team based on jersey color.

    - only need the T-shirt, therefore crop the top part only
    
    - can't take the average color as the background is dominant, therefore green grass background is segmented out using KMeans classification. This results in two different segemented part. This is used to find the two colors available and to find which one is the background grass and which one is the player
    
    - Then the team index label divide and assign to two teams, for this again KMeans classification is used


# Inference

The input video i used can be seen ( [here](https://drive.google.com/file/d/14FqtHMLTJVRtzL-7uA5jg7Cdq8bMFbTP/view?usp=drive_link) )

The model is then used to predict this video and the output video can be seen ( [here](https://drive.google.com/file/d/1Oq3k9UuIrjIFftvtbLKl57hdFXpTSjva/view?usp=drive_link) )


# PROBLEM SEEN:
    - there's some frames while the ball is not detected and get a flickering detection of the ball
    - when the bounding boxes of different player coincide and pass through, the track id differs and the model detects, assigns a new object and it's is not tracked properly

this problem can be seen clearly in this link ( [here](https://drive.google.com/file/d/16tOUZa9zORrCSXzhFO24p4ihA9Cjh88X/view?usp=drive_link) )

    
# **Upcoming tasks:** 
1. Ball possession
2. Perform perspective transformation
3. Speed and distance calulation in meters and not pixels
4. Calculation of player movement
