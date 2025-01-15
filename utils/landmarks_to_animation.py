import cv2
import numpy as np
import pandas as pd

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index Finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle Finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring Finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17)  # Palm Base
]

def landmarks_to_video(landmarks_df, output_path, fps=30, width=1920, height=1080):
    """
    Convert landmarks DataFrame to a video using OpenCV, including connections between landmarks.
    
    Parameters:
        landmarks_df: pd.DataFrame - DataFrame containing landmarks data
        output_path: str - Path to save the output video
        fps: int - Frames per second for the video
        width: int - Width of the output video
        height: int - Height of the output video
    """
    # Initialize OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define scaling and translation for visualization
    x_scale = min(width, height) * 0.7  # Scale landmarks to fit in frame
    x_offset = width // 2             # Center horizontally
    y_offset = height // 2            # Center vertically

    def draw_landmarks(frame, landmarks, color):
        """
        Draw landmarks and connections on a frame.
        """
        # Draw landmarks
        for x, y, _ in landmarks:
            cv2.circle(frame, (int(x * scale + x_offset), int(-y * scale + y_offset)), 5, color, -1)

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx][0] * scale + x_offset),
                               int(-landmarks[start_idx][1] * scale + y_offset))
                end_point = (int(landmarks[end_idx][0] * scale + x_offset),
                             int(-landmarks[end_idx][1] * scale + y_offset))
                cv2.line(frame, start_point, end_point, color, 2)

    # Process each frame
    for _, row in landmarks_df.iterrows():
        # Create a blank image for the frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Parse the row data
        hand_label = row['hand_label']  # 'Right' or 'Left'
        landmarks = row[5:].values.reshape(-1, 3)  # Reshape (x, y, z) for 21 points

        # Draw landmarks and connections
        if hand_label == 'Right':
            draw_landmarks(frame, landmarks, color=(255, 0, 0))  # Blue for Right Hand
        elif hand_label == 'Left':
            draw_landmarks(frame, landmarks, color=(0, 0, 255))  # Red for Left Hand

        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved at {output_path}")


if __name__ == "__main__":
    # Example usage
    landmarks_csv = pd.read_csv(r'/home/atefeh/AG-MAE/data/asl/test/7673_research.csv')
    landmarks_to_video(landmarks_csv, 'output.mp4')
