import os
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageEnhance
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
        # saved_state_dict = torch.load(save_state, map_location=torch.device('cpu'))
        saved_state_dict = torch.load(save_state, map_location=torch.device('cpu'), weights_only=False)


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


    def get_coordinates_obj(self):
        print('GET OBJ: ', self.xy_coordinates)
        print(" xy coord type : ", type(self.xy_coordinates))
        return self.xy_coordinates

    def load_ball_coordinates(self, coordinates_file):
        try:
            xy_coordinates = np.load(coordinates_file, allow_pickle=True)
            print(f"Loaded coordinates from {coordinates_file} with shape: {xy_coordinates.shape}")
            return xy_coordinates
        except Exception as e:
            print(f"Failed to load coordinates from {coordinates_file}. Error: {e}")
            return None
        
    def check_cache(self, videoname):
        coordinates_file = f'output/{videoname}.npy'
        if os.path.exists(coordinates_file):
            print("Coordinates file found. Loading from file...")
            self.xy_coordinates = self.load_ball_coordinates(coordinates_file)
            print(' SELF XY CORRD: ' , self.xy_coordinates)
            return True
        else:
            print(f' no cache found for output/{videoname}.npy!!!')
            return False

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
            # print(' self. xy_coordinates : ', self.xy_coordinates)
            # print(np)



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

    # TRAIL LINE 
    def mark_positions2(self, frame, trail_length=5, frame_num=None, ball_color='yellow'):
        base_color = ImageColor.getrgb(ball_color)
        if frame_num is not None:
            # Get positions from the frame number with a limit on trail length
            positions = self.xy_coordinates[max(0, frame_num - trail_length + 1):frame_num + 1, :]
        else:
            # Get the last 'trail_length' positions
            positions = self.xy_coordinates[-trail_length:, :]

        # Apply smoothing
        positions = self.smooth_positions(positions)

        
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

        font = ImageFont.load_default()
        image_width = self.video_width
        image_height = self.video_height

        # thicknesses = [1,2,3,4,5]
        thicknesses = [1,3,3,7,7]

        threshold_percentage = 0.2  # 20% threshold for relative differences
        red = (255,0,0)
        orange = (255,165,0)
        yellow = (255,255,0)
        # gradient_ball_colors = []
        for i in range(1, len(positions)):
            isBad = False
            if positions[i-1][0] is not None and positions[i][0] is not None:
                xdiff = abs(positions[i][0] - positions[i-1][0])
                ydiff = abs(positions[i][1] - positions[i-1][1])
                prev_x = positions[i-1][0]
                prev_y = positions[i-1][1]

                # Calculate relative differences
                x_relative_diff = xdiff / prev_x if prev_x != 0 else 0
                y_relative_diff = ydiff / prev_y if prev_y != 0 else 0

                # Define outlier criteria
                if x_relative_diff > threshold_percentage or y_relative_diff > threshold_percentage:
                    isBad = True
                    print(' OUTLIER!! : >>>(', positions[i][0], positions[i][1], ')')
                    # Use previous position to replace the outlier
                    positions[i][0] = positions[i-1][0]
                    positions[i][1] = positions[i-1][1]
                else:
                    xpercent = x_relative_diff * 100
                    ypercent = y_relative_diff * 100
                    # print(">>>" , positions[i], f"x:{xpercent:.2f}% , y:{ypercent:.2f}%")
                
                fade_factor = i / len(positions)
                # fade_factor = fade_factor//2
                transparency = int(255 * fade_factor)//2
                if 0<i<=2 :
                    color = red
                elif 2<i<=4:
                    color = orange
                else:
                    color = yellow
                faded_color_with_alpha = color + (transparency,)
                line_thickness = thicknesses[min(i-1, len(thicknesses)-1)]

                draw.line(
                    (positions[i-1][0], positions[i-1][1], positions[i][0], positions[i][1]),
                    fill=faded_color_with_alpha,
                    width=line_thickness
                )
                # if isBad: 
                #     text = f"({positions[i][0]:.2f} , {positions[i][1]:.2f})\n{xpercent*100:.2f}%,    {ypercent*100:.2f}%\n BAD!!!!!!"
                # else:
                #     text = f"({positions[i][0]:.2f} , {positions[i][1]:.2f})\n{xpercent*100:.2f}%,    {ypercent*100:.2f}%"

                # Draw a filled rectangle over the previous text
                # draw.rectangle(
                #     [text_position, (image_width - 10, image_height - 10)],
                #     fill=background_color
                # )

                # # Draw the new text on top of the cleared area
                # draw.text(text_position, text, fill="white", font=font)



        # # Determine ROI based on the last known position
        # if positions[-1, 0] is not None and positions[-1, 1] is not None:
        #     last_known_position = positions[-1]
        #     x, y = int(last_known_position[0]), int(last_known_position[1])
        #     roi = (max(0, x - self.search_radius), max(0, y - self.search_radius),
        #            min(frame.shape[1], x + self.search_radius) - max(0, x - self.search_radius),
        #            min(frame.shape[0], y + self.search_radius) - max(0, y - self.search_radius))
            

        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame


    def interpolate_color(self, start_color, end_color, factor):
        """
        Interpolates between two RGB colors.
        :param start_color: The starting color (as an RGB tuple)
        :param end_color: The ending color (as an RGB tuple)
        :param factor: A value between 0 and 1, where 0 means start_color and 1 means end_color
        :return: The interpolated color as an RGB tuple
        """
        return tuple(
            int(start_color[j] + factor * (end_color[j] - start_color[j]))
            for j in range(3)
        )


    def printarray(self):
        print('self.xy_coordinates :', self.xy_coordinates)
        
    def mark_positions1(self, frame, mark_num=7, frame_num=None, ball_color='yellow'):
        bounce_i = None
        
        # Define RGB colors for transitions (50% opacity, alpha = 127)
        ball_color_rgb = (255, 255, 0, 150)   # Yellow with 50% opacity
        orange_rgb = (255, 165, 0, 150)       # Orange with 50% opacity
        red_rgb = (255, 0, 0, 150)            # Red with 50% opacity
        blue_rgb = (0, 0, 255, 130)
        green_rgb = (171, 225, 0, 130)
        pink_rgb = (255,0,154, 130)
        purple_rgb = (230,0,255,130)
        light_yellow_rgb = (255,255,150,130)
        white_rgb = (255,255,255,130)

        
        # If frame number is provided, use the relevant slice of xy_coordinates
        if frame_num is not None:
            # print('frame_num: ', frame_num)
            q = self.xy_coordinates[frame_num-mark_num+1:frame_num+1, :]
            for i in range(frame_num - mark_num + 1, frame_num + 1):
                if i in self.bounces_indices:
                    bounce_i = i - frame_num + mark_num - 1
                    break
        else:
            q = self.xy_coordinates[-mark_num:, :]
        
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))  # Transparent overlay


        # Mark each position by a circle
        for i in range(q.shape[0]):
            if q[i, 0] is not None:
                draw_x = q[i, 0]
                draw_y = q[i, 1]
                
                # # Assign colors: First 2 positions yellow, next 3 orange, last 2 red
                # radius = 2
                # dic = { 7:ball_color_rgb , 6:orange_rgb, 5:red_rgb, 4:pink_rgb, 3:purple_rgb, 2:blue_rgb, 1:green_rgb}
                # if i not in dic:
                #     current_color = green_rgb
                # else:
                #     current_color = dic[i]
                if i < 2:
                    current_color =  red_rgb # Red
                    # current_color = white_rgb
                    radius = 1 # Smaller size
                elif i < 4:
                    current_color = orange_rgb # Orange
                    # current_color = light_yellow_rgb
                    radius = 2 # Smaller size
                else:
                    current_color = ball_color_rgb # Yellow
                    radius = 3
                current_color = ball_color_rgb # Yellow

                bbox = (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
                draw = ImageDraw.Draw(overlay)

                if bounce_i is not None and i == bounce_i:
                    # Draw bounce position with fully opaque red color
                    draw.ellipse(bbox, fill=(255, 0, 0, 255))  # Full opacity red for bounce
                else:
                    # Draw semi-transparent circles with the chosen color
                    # draw.ellipse(bbox, fill=current_color)
                    draw.ellipse(bbox, fill=current_color)

        # Composite the overlay with the original image
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)

        # Convert back to OpenCV format (BGR)
        frame = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        return frame







    # def mark_positions1(self, frame, mark_num=10, frame_num=None, ball_color='yellow'):
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
    
    # def show_y_graph(self, player_1_boxes, player_2_boxes):
    #     """
    #     Display ball y index positions and both players y index positions in all the frames in a graph
    #     :param player_1_boxes: bottom player boxes
    #     :param player_2_boxes: top player boxes
    #     """
    #     player_1_centers = np.array([center_of_box(box) for box in player_1_boxes])
    #     player_1_y_values = player_1_centers[:, 1]
    #     # get y value of top quarter of bottom player box
    #     player_1_y_values -= np.array([(box[3] - box[1]) // 4 for box in player_1_boxes])

    #     # Calculate top player boxes center
    #     player_2_centers = []
    #     for box in player_2_boxes:
    #         if box[0] is not None:
    #             player_2_centers.append(center_of_box(box))
    #         else:
    #             player_2_centers.append([None, None])
    #     player_2_centers = np.array(player_2_centers)
    #     player_2_y_values = player_2_centers[:, 1]

    #     y_values = self.xy_coordinates[:, 1].copy()
    #     x_values = self.xy_coordinates[:, 0].copy()

    #     plt.figure()
    #     plt.scatter(range(len(y_values)), y_values)
    #     plt.plot(range(len(player_1_y_values)), player_1_y_values, color='r')
    #     plt.plot(range(len(player_2_y_values)), player_2_y_values, color='g')
    #     plt.show()


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