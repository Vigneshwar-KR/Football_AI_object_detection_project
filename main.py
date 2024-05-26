from utilities import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
import numpy as np

def main():
    # print("Hello World")

    # Read Video
    video_frames = read_video('input/08fd33_5.mp4')


    # Initialize Tracker
    # Use the best weights from trained model
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.assign_player_to_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            
            # save the team to each player
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    

    # # Crop the image of a player
    # for track_id,player in tracks['players'][0].items():
    #     bbox= player['bbox']
    #     frame = video_frames[0]

    #     # crop the bbox
    #     cropped_frame = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

    #     # save the cropped frame
    #     cv2.imwrite(f'output/cropped_image.jpg', cropped_frame)
    #     break

    #draw annotation
    video_frames = tracker.draw_annotations(video_frames,tracks)
    

    # Save video
    save_video(video_frames, 'output/output_video.avi')



if __name__ == '__main__':
    main()