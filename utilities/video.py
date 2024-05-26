import cv2

# read the video and returns the frames in a list 
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()      # ret is a flag if video ended
        if not ret:
            break
        frames.append(frame)
    return frames\
    
# save video by writing each frame to a video
def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0])) #(path, format, FPS,  frame width and height)
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()