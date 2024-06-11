from utilities import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
import numpy as np
import os
from ball_to_player_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from perspective_transformation import PerspectiveTransformation
from speed_distance_estimator import SpeedDistanceEstimator

def main():
    # print("Hello World")

    # Read Video
    video_frames = read_video('input/08fd33_5.mp4')


    # Initialize Tracker
    # Use the best weights from trained model
    # tracker = Tracker('models/best.pt')
    model_path = 'models/best.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file was not found: {model_path}")
    tracker = Tracker(model_path)


    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # Perspective Transformation
    perspective_transformer = PerspectiveTransformation()
    perspective_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball detection
    tracks["ball"] = tracker.interpolate_ball(tracks["ball"])

    # # Speed and distance estimator
    # speed_distance_estimator = SpeedDistanceEstimator()
    # speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
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


    # Assign ball to player  
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])              # assign to last team who had the ball
    team_ball_control= np.array(team_ball_control)

    ## draw annotation
    video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)
    
    ## Draw Camera movement
    video_frames = camera_movement_estimator.draw_camera_movement(video_frames,camera_movement_per_frame)

    ## Draw speed and distance
    # speed_distance_estimator.draw_speed_and_distance(video_frames,tracks)
    # Save video
    save_video(video_frames, 'output/output_video.avi')



if __name__ == '__main__':
    main()