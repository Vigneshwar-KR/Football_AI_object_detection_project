from utilities import read_video, save_video
from trackers import Tracker


def main():
    # print("Hello World")

    # Read Video
    video_frames = read_video('input/08fd33_5.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    #draw annotation
    video_frames = tracker.draw_annotations(video_frames,tracks)
    

    # Save video
    save_video(video_frames, 'output/output_video.avi')



if __name__ == '__main__':
    main()