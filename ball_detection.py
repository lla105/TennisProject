import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageColor
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks

from src.ball_tracker_net import BallTrackerNet
from src.detection import center_of_box
from src.utils import get_video_properties


def combine_three_frames(frame1, frame2, frame3, width, height):
    """
    Combine three frames into one input tensor for detecting the ball
    """

    # Resize and type converting for each frame
    img = cv2.resize(frame1, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # resize it
    img1 = cv2.resize(frame2, (width, height))
    # input must be float type
    img1 = img1.astype(np.float32)

    # resize it
    img2 = cv2.resize(frame3, (width, height))
    # input must be float type
    img2 = img2.astype(np.float32)

    # combine three imgs to  (width , height, rgb*3)
    imgs = np.concatenate((img, img1, img2), axis=2)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)


class BallDetector:
    """
    Ball Detector model responsible for receiving the frames and detecting the ball
    """
    def __init__(self, save_state, out_channels=2):
        self.search_radius = 50

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load TrackNet model weights
        self.detector = BallTrackerNet(out_channels=out_channels)
        # saved_state_dict = torch.load(save_state)
        saved_state_dict = torch.load(save_state, map_location=torch.device('cpu'))

        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.threshold_dist = 100
        self.xy_coordinates = np.array([[None, None], [None, None]])

        self.bounces_indices = []

    def detect_ball(self, frame):
        """
        After receiving 3 consecutive frames, the ball will be detected using TrackNet model
        :param frame: current frame
        """
        # Save frame dimensions
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        # detect only in 3 frames were given
        if self.last_frame is not None:
            # combine the frames into 1 input tensor
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                          self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            # Inference (forward pass)
            x, y = self.detector.inference(frames)
            if x is not None:
                # Rescale the indices to fit frame dimensions
                x = x * (self.video_width / self.model_input_width)
                y = y * (self.video_height / self.model_input_height)

                # Check distance from previous location and remove outliers
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x,y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)

    # def mark_positions(self, frame, mark_num=40, frame_num=None, ball_color='yellow'):
    #     """
    #     Mark the last 'mark_num' positions of the ball in the frame
    #     :param frame: the frame we mark the positions in
    #     :param mark_num: number of previous detection to mark
    #     :param frame_num: current frame number
    #     :param ball_color: color of the marks
    #     :return: the frame with the ball annotations
    #     """
    #     bounce_i = None
    #     # if frame number is not given, use the last positions found
    #     if frame_num is not None:
    #         q = self.xy_coordinates[frame_num-mark_num+1:frame_num+1, :]
    #         for i in range(frame_num - mark_num + 1, frame_num + 1):
    #             if i in self.bounces_indices:
    #                 bounce_i = i - frame_num + mark_num - 1
    #                 break
    #     else:
    #         q = self.xy_coordinates[-mark_num:, :]
    #     pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     pil_image = Image.fromarray(pil_image)
    #     # Mark each position by a circle
    #     for i in range(q.shape[0]):
    #         if q[i, 0] is not None:
    #             draw_x = q[i, 0]
    #             draw_y = q[i, 1]
    #             bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
    #             draw = ImageDraw.Draw(pil_image)
    #             if bounce_i is not None and i == bounce_i:
    #                 draw.ellipse(bbox, outline='red')
    #             else:
    #                 # draw.ellipse(bbox, outline=ball_color)
    #                 draw.ellipse(bbox, fill=ball_color)

    #         # Convert PIL image format back to opencv image format
    #         frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    #     return frame


    def smooth_positions(self, positions):
        smoothed_positions = []
        for i in range(len(positions)):
            # Filter out None values before computing the mean
            valid_positions = [pos for pos in positions[:i+1] if pos[0] is not None and pos[1] is not None]
            if valid_positions:  # Only calculate mean if there are valid positions
                smoothed_positions.append(np.mean(valid_positions, axis=0))
            else:
                smoothed_positions.append((None, None))  # Or append a placeholder for no valid positions
        return smoothed_positions
    def detect_ball_in_roi(self, frame, roi):
        # Crop the frame to the ROI
        x, y, w, h = roi
        cropped_frame = frame[y:y+h, x:x+w]
        
        # Perform object detection on the cropped frame
        # (Replace this with your actual detection code)
        detected_objects = self.yolo_detection(cropped_frame)
        
        # Adjust detected objects' coordinates to the original frame
        for obj in detected_objects:
            obj.x += x
            obj.y += y
        
        return detected_objects

    def mark_positions(self, frame, trail_length=5, frame_num=None, ball_color='yellow'):
        base_color = ImageColor.getrgb(ball_color)
        # If frame number is not given, use the last positions found
        if frame_num is not None:
            # Ensure there are enough positions
            if frame_num - trail_length + 1 < 0:
                positions = self.xy_coordinates[:frame_num + 1, :]
            else:
                positions = self.xy_coordinates[frame_num-trail_length+1:frame_num+1, :]
        else:
            positions = self.xy_coordinates[-trail_length:, :]

        # Apply smoothing
        # positions = self.smooth_positions(positions)
        
        # Append the current position
        if frame_num is not None and len(self.xy_coordinates) > frame_num:
            current_position = self.xy_coordinates[frame_num]
            if current_position[0] is not None and current_position[1] is not None:
                positions.append(current_position)
            else:
                positions.append((None, None))

        positions = np.array(positions)
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        thicknesses = [1,2,3,4,5]

        for i in range(1, len(positions)):
            if positions[i-1, 0] is not None and positions[i, 0] is not None:
                fade_factor = i / len(positions)
                transparency = int(255 * fade_factor)
                faded_color_with_alpha = base_color + (transparency,)
                line_thickness = thicknesses[min(i-1, len(thicknesses)-1)]

                draw.line(
                    (positions[i-1, 0], positions[i-1, 1], positions[i, 0], positions[i, 1]),
                    fill=faded_color_with_alpha,
                    width=line_thickness
                )

        # Determine ROI based on the last known position
        if positions[-1, 0] is not None and positions[-1, 1] is not None:
            last_known_position = positions[-1]
            x, y = int(last_known_position[0]), int(last_known_position[1])
            roi = (max(0, x - self.search_radius), max(0, y - self.search_radius),
                   min(frame.shape[1], x + self.search_radius) - max(0, x - self.search_radius),
                   min(frame.shape[0], y + self.search_radius) - max(0, y - self.search_radius))
            
            # Perform detection in the ROI
            detected_objects = self.detect_ball_in_roi(frame, roi)
            
            # Process detected objects as needed

        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame



    def show_y_graph(self, player_1_boxes, player_2_boxes):
        """
        Display ball y index positions and both players y index positions in all the frames in a graph
        :param player_1_boxes: bottom player boxes
        :param player_2_boxes: top player boxes
        """
        player_1_centers = np.array([center_of_box(box) for box in player_1_boxes])
        player_1_y_values = player_1_centers[:, 1]
        # get y value of top quarter of bottom player box
        player_1_y_values -= np.array([(box[3] - box[1]) // 4 for box in player_1_boxes])

        # Calculate top player boxes center
        player_2_centers = []
        for box in player_2_boxes:
            if box[0] is not None:
                player_2_centers.append(center_of_box(box))
            else:
                player_2_centers.append([None, None])
        player_2_centers = np.array(player_2_centers)
        player_2_y_values = player_2_centers[:, 1]

        y_values = self.xy_coordinates[:, 1].copy()
        x_values = self.xy_coordinates[:, 0].copy()

        plt.figure()
        plt.scatter(range(len(y_values)), y_values)
        plt.plot(range(len(player_1_y_values)), player_1_y_values, color='r')
        plt.plot(range(len(player_2_y_values)), player_2_y_values, color='g')
        plt.show()


if __name__ == "__main__":
    print('====== RUNNING ball_dectection MAIN ======')
    ball_detector = BallDetector('saved states/tracknet_weights_lr_1.0_epochs_150_last_trained.pth')
    cap = cv2.VideoCapture('../videos/vid1.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(cap)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        frame_i += 1
        if not ret:
            break

        ball_detector.detect_ball(frame)


    cap.release()
    cv2.destroyAllWindows()

    from scipy.interpolate import interp1d

    y_values = ball_detector.xy_coordinates[:,1]

    new = signal.savgol_filter(y_values, 3, 2)

    x = np.arange(0, len(new))
    indices = [i for i, val in enumerate(new) if np.isnan(val)]
    x = np.delete(x, indices)
    y = np.delete(new, indices)
    f = interp1d(x, y, fill_value="extrapolate")
    f2 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(y_values), num=len(y_values), endpoint=True)
    plt.plot(np.arange(0, len(new)), new, 'o',xnew,
             f2(xnew), '-r')
    plt.legend(['data', 'inter'], loc='best')
    plt.show()

    positions = f2(xnew)
    peaks, _ = find_peaks(positions, distance=30)
    a = np.diff(peaks)
    plt.plot(positions)
    plt.plot(peaks, positions[peaks], "x")
    plt.show()