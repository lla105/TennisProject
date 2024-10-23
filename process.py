import os
import time
import sys

# Add the parent directory of 'src' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from detection import DetectionModel, center_of_box
from pose import PoseExtractor
from smooth import Smooth
from src.ball_detection import BallDetector
from my_statistic import Statistics
from src.stroke_recognition import ActionRecognition
from utils import get_video_properties, get_dtype, get_stickman_line_connection
from court_detection import CourtDetector
import matplotlib.pyplot as plt


def get_stroke_predictions(video_path, stroke_recognition, strokes_frames, player_boxes):
    """
    Get the stroke prediction for all sections where we detected a stroke
    """
    predictions = {}
    cap = cv2.VideoCapture(video_path)
    fps, length, width, height = get_video_properties(cap)
    video_length = 2
    # For each stroke detected trim video part and predict stroke
    for frame_num in strokes_frames:
        # Trim the video (only relevant frames are taken)
        starting_frame = max(0, frame_num - int(video_length * fps * 2 / 3))
        cap.set(1, starting_frame)
        i = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            stroke_recognition.add_frame(frame, player_boxes[starting_frame + i])
            i += 1
            if i == int(video_length * fps):
                break
        # predict the stroke
        probs, stroke = stroke_recognition.predict_saved_seq()
        predictions[frame_num] = {'probs': probs, 'stroke': stroke}
    cap.release()
    return predictions


def find_strokes_indices(player_1_boxes, player_2_boxes, ball_positions, skeleton_df, verbose=0):
    """
    Detect strokes frames using location of the ball and players
    """
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2)
    smooth_y = signal.savgol_filter(ball_y, 3, 2)

    # Ball position interpolation
    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)
    ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(smooth_y)), smooth_y, 'o', xnew,
                 ball_f2_y(xnew), '-r')
        plt.legend(['data', 'inter'], loc='best')
        plt.show()

    # Player 2 position interpolation
    player_2_centers = np.array([center_of_box(box) for box in player_2_boxes])
    player_2_x, player_2_y = player_2_centers[:, 0], player_2_centers[:, 1]
    player_2_x = signal.savgol_filter(player_2_x, 3, 2)
    player_2_y = signal.savgol_filter(player_2_y, 3, 2)
    x = np.arange(0, len(player_2_y))
    indices = [i for i, val in enumerate(player_2_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(player_2_y, indices)
    y2 = np.delete(player_2_x, indices)
    player_2_f_y = interp1d(x, y1, fill_value="extrapolate")

    player_2_f_x = interp1d(x, y2, fill_value="extrapolate")
    xnew = np.linspace(0, len(player_2_y), num=len(player_2_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(player_2_y)), player_2_y, 'o', xnew, player_2_f_y(xnew), '--g')
        plt.legend(['data', 'inter_cubic', 'inter_lin'], loc='best')
        plt.show()

    coordinates = ball_f2_y(xnew)
    # Find all peaks of the ball y index
    peaks, _ = find_peaks(coordinates)
    if verbose:
        plt.plot(coordinates)
        plt.plot(peaks, coordinates[peaks], "x")
        plt.show()

    neg_peaks, _ = find_peaks(coordinates * -1)
    if verbose:
        plt.plot(coordinates)
        plt.plot(neg_peaks, coordinates[neg_peaks], "x")
        plt.show()

    # Get bottom player wrists positions
    left_wrist_index = 9
    right_wrist_index = 10
    skeleton_df = skeleton_df.fillna(-1)
    left_wrist_pos = skeleton_df.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    right_wrist_pos = skeleton_df.iloc[:, [right_wrist_index, right_wrist_index + 15]].values

    dists = []
    # Calculate dist between ball and bottom player
    for i, player_box in enumerate(player_1_boxes):
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
            box_dist = np.linalg.norm(player_center - ball_pos)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            dists.append(min(box_dist, right_wrist_dist, left_wrist_dist))
        else:
            dists.append(None)
    dists = np.array(dists)

    dists2 = []
    # Calculate dist between ball and top player
    for i in range(len(player_2_centers)):
        ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
        box_center = np.array([player_2_f_x(i), player_2_f_y(i)])
        box_dist = np.linalg.norm(box_center - ball_pos)
        dists2.append(box_dist)
    dists2 = np.array(dists2)

    strokes_1_indices = []
    # Find stroke for bottom player by thresholding the dists
    for peak in peaks:
        player_box_height = max(player_1_boxes[peak][3] - player_1_boxes[peak][1], 130)
        if dists[peak] < (player_box_height * 4 / 5):
            strokes_1_indices.append(peak)

    strokes_2_indices = []
    # Find stroke for top player by thresholding the dists
    for peak in neg_peaks:
        if dists2[peak] < 100:
            strokes_2_indices.append(peak)

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_1_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists[strokes_1_indices[i]], dists[strokes_1_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_1_indices = np.delete(strokes_1_indices, to_del)
        if len(to_del) == 0:
            break

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_2_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists2[strokes_2_indices[i]], dists2[strokes_2_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_2_indices = np.delete(strokes_2_indices, to_del)
        if len(to_del) == 0:
            break

    # Assume bounces frames are all the other peaks in the y index graph
    bounces_indices = [x for x in peaks if x not in strokes_1_indices]
    if verbose:
        plt.figure()
        plt.plot(coordinates)
        plt.plot(strokes_1_indices, coordinates[strokes_1_indices], "or")
        plt.plot(strokes_2_indices, coordinates[strokes_2_indices], "og")
        plt.legend(['data', 'player 1 strokes', 'player 2 strokes'], loc='best')
        plt.show()

    return strokes_1_indices, strokes_2_indices, bounces_indices, player_2_f_x, player_2_f_y


def mark_player_box(frame, boxes, frame_num):
    box = boxes[frame_num]
    if box[0] is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
    return frame


def mark_skeleton(skeleton_df, img, img_no_frame, frame_number):
    """
    Mark the skeleton of the bottom player on the frame
    """
    # landmarks colors
    circle_color, line_color = (0, 0, 255), (255, 0, 0)
    stickman_pairs = get_stickman_line_connection()

    skeleton_df = skeleton_df.fillna(-1)
    values = np.array(skeleton_df.values[frame_number], int)
    points = list(zip(values[5:17], values[22:]))
    # draw key points
    for point in points:
        if point[0] >= 0 and point[1] >= 0:
            xy = tuple(np.array([point[0], point[1]], int))
            cv2.circle(img, xy, 2, circle_color, 2)
            cv2.circle(img_no_frame, xy, 2, circle_color, 2)

    # Draw stickman
    for pair in stickman_pairs:
        partA = pair[0] - 5
        partB = pair[1] - 5
        if points[partA][0] >= 0 and points[partA][1] >= 0 and points[partB][0] >= 0 and points[partB][1] >= 0:
            cv2.line(img, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
            cv2.line(img_no_frame, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
    return img, img_no_frame

import cv2
import os
import numpy as np


def get_unique_filename(output_folder , output_file) :
    base_name = output_file
    extension = '.mp4'
    counter = 1
    while os.path.isfile(os.path.join(output_folder , base_name+extension)):
        base_name = f'{output_file}({counter})'
        counter+=1
    return base_name + extension
"""
Creates a new video with only ball tracking overlay.
:param input_video: str, path to the input video
:param ball_detector: object, detector used to mark ball positions
:param show_video: bool, display output video while processing
:param output_folder: str, path to output folder
:param output_file: str, name of the output file
:return: None
"""
def add_ball_tracking_to_video(input_video, ball_detector, show_video, output_folder, output_file):
    # test = ball_detector.printarray()
    
    # Read video file
    cap = cv2.VideoCapture(input_video)
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    unique_output_file = get_unique_filename( output_folder , output_file )
    # Video writer to save the output
    out = cv2.VideoWriter(os.path.join(output_folder, unique_output_file ),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    # Initialize frame counter
    frame_number = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Add ball location
        img = ball_detector.mark_positions1(img, frame_num=frame_number)
        # img = ball_detector.mark_positions2(img, frame_num=frame_number)

        # Display frame if needed
        if show_video:
            cv2.imshow('Output', img)
            if cv2.waitKey(1) & 0xff == 27:  # Press 'Esc' to exit
                break

        # Save output video
        out.write(img)
        frame_number += 1

        print(f'Processing frame {frame_number}/{length}', '\r', end='')

    print(f'\nFinished processing video. Output saved as {unique_output_file}.mp4')

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def create_top_view(court_detector, detection_model):
    """
    Creates top view video of the gameplay
    """
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('output/top_view.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_width, v_height))
    # players location on court
    # smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)

    # for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
    #     frame = court.copy()
    #     frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 10, (0, 0, 255), 15)
    #     if feet_pos_2[0] is not None:
    #         frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 10, (0, 0, 255), 15)
    #     out.write(frame)
    out.release()
    cv2.destroyAllWindows()




def video_process(video_path, show_video=False, include_video=True,
                  stickman=True, stickman_box=True, court=True,
                  output_file='output', output_folder='output',
                  smoothing=True, top_view=True, videoname='videoname'):
    # initialize extractors
    court_detector = CourtDetector()
    # detection_model = DetectionModel(dtype=dtype)
    # pose_extractor = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None
    # stroke_recognition = ActionRecognition('storke_classifier_weights.pth')
    ball_detector = BallDetector('saved states/tracknet_weights_2_classes.pth', out_channels=2)

    print(' video name is : ', videoname)
    has_cache = ball_detector.check_cache(videoname)
    if has_cache:
        print(' !!!!!!!!!!!!!!!! has cache')

    # Load videos from videos path
    video = cv2.VideoCapture(video_path)

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # frame counter
    frame_i = 0

    # time counter
    total_time = 0

    # Loop over all frames in the videos
    while True:
        start_time = time.time()
        ret, frame = video.read()
        frame_i += 1

        if ret:
            if frame_i == 1:
                start_time = time.time()
            if not has_cache:
                ball_detector.detect_ball(frame)

                total_time += (time.time() - start_time)
                print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
                if not frame_i % 100:
                    print('')
        else:
            break
    # print('Processing frame %d/%d  FPS %04f' % (length, length, length / total_time), '\n', end='')
    print('Processing completed')
    
    coordinate_bulb = ball_detector.get_coordinates_obj()
    print(' process.py > coordinate bulb : ' , coordinate_bulb)
    save_ball_coordinates(videoname, coordinate_bulb)

    video.release()
    cv2.destroyAllWindows()

    add_ball_tracking_to_video(input_video=video_path, ball_detector=ball_detector, show_video=show_video, output_folder=output_folder, output_file=output_file)


def save_ball_coordinates(videoname, coordinate_bulb):
    # Check if xy_coordinates is not empty before saving
    # if self.xy_coordinates.size > 0:
    output_file = f'output/{videoname}.npy'
    if len(coordinate_bulb) >0:
        np.save(output_file, coordinate_bulb)
        print(f'Saved coordinates to {output_file}.npy with shape: {coordinate_bulb.shape}')
    else:
        print('No coordinates to save.')


def main():
    for i in range(3):
        print()
    s = time.time()
    # MUST TURN ON : show_video , stickman , smoothing , 
    videoname = '34frames'
    # videoname = '19secs'
    # videoname = '16secs'
    # videoname = 'temp11'
    # videoname = 'test164k'
    # videoname = 'small'
    # videoname = ''

    video_process(video_path=f'../videos/{videoname}.mp4', show_video=True, stickman=True, stickman_box=False, smoothing=True,
                  court=False, top_view=True, videoname=videoname)
    print(f'Total computation time : {time.time() - s} seconds')


if __name__ == "__main__":
    main()
