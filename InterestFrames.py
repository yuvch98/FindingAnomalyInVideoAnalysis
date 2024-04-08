# importing needed libraries
from DataExploratoryAnalysis import get_label_from_path
import cv2
import numpy as np
import random
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
random.seed(42)

def convert_to_grayscale_and_resize(video_path, resize=(100, 80), sampling_rate=2,median_normal_time=70.70):
    video_path = video_path.decode("utf-8") if isinstance(video_path, bytes) else str(video_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the total number of frames to read, limiting to 70.70 seconds for normal videos
    if get_label_from_path(video_path) == 0:
        total_frames = min(total_frames, int(fps * median_normal_time))
    # Calculate the interval between frames to sample based on the desired sampling rate
    sampling_interval = int(fps / sampling_rate)

    grayscale_frames = []
    rgb_frames = []
    frame_index = 0  # Keep track of the total number of frames processed

    while frame_index < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Process and append frames at the sampling interval
        if frame_index % sampling_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_gray = cv2.resize(gray_frame, resize)  # Ensure correct order: (width, height)
            resized_rgb = cv2.resize(frame, resize)  # Ensure correct order: (width, height)
            grayscale_frames.append(resized_gray)
            rgb_frames.append(resized_rgb)

        frame_index += 1

    cap.release()
    return grayscale_frames, rgb_frames


def extract_interest_frames(grayscale_frames, rgb_frames, max_frames=25):
    # Initialize a list to hold the score and corresponding rgb_frame
    scored_rgb_frames = []

    # Iterate through the grayscale frames to calculate scores
    for i in range(len(grayscale_frames) - 1):
        frame1 = grayscale_frames[i]
        frame2 = grayscale_frames[i + 1]

        # Calculate the Structural Similarity Index (SSIM) score between consecutive frames
        score = ssim(frame1, frame2)

        # Append the score and corresponding rgb_frame to the list
        scored_rgb_frames.append((score, rgb_frames[i + 1]))

    # Sort the list based on scores, in ascending order to have the smallest scores first
    scored_rgb_frames.sort(key=lambda x: x[0])

    # Extract the first max_frames rgb_frames with the smallest scores
    interest_rgb_frames = [frame for score, frame in scored_rgb_frames[:max_frames]]

    return interest_rgb_frames

def load_video_frames(video_path, max_frames=25, resize=(100, 80)):
    grayscale_frames, rgb_frames = convert_to_grayscale_and_resize(video_path, resize)
    interest_rgb_frames = extract_interest_frames(grayscale_frames, rgb_frames)

    # Ensure there's at least one frame available, using the first frame of the video as a fallback
    fallback_frame = rgb_frames[0] if rgb_frames else np.zeros((resize[0], resize[1], 3))

    # If no interest frames were identified, fill the list with 20 copies of the fallback frame
    if len(interest_rgb_frames) == 0:
        interest_rgb_frames = [fallback_frame for _ in range(max_frames)]
    # If there are more than max_frames, select max_frames uniformly
    elif len(interest_rgb_frames) > max_frames:
        selected_frames = np.linspace(0, len(interest_rgb_frames) - 1, max_frames, dtype=int)
        interest_rgb_frames = [interest_rgb_frames[i] for i in selected_frames]
    # If there are fewer frames than required, repeat the last frame until reaching 20 frames
    else:
        while len(interest_rgb_frames) < max_frames:
            interest_rgb_frames.append(interest_rgb_frames[-1])

    # Normalize pixel values to [0, 1]
    interest_rgb_frames = [frame / 255.0 for frame in interest_rgb_frames]
    return np.array(interest_rgb_frames).astype(np.float32)

def preprocess_dataset(file_paths, labels, max_frames=25, resize=(100, 80)):
  def load_and_preprocess(video_path, label):
    video_content = tf.numpy_function(load_video_frames, [video_path, max_frames, resize], tf.float32)
    video_content.set_shape((None, resize[1], resize[0], 3))
    return video_content, label

  path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  dataset = tf.data.Dataset.zip((path_ds, labels_ds))
  dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
  return dataset

def augment_frames(frames):
  # Example of a simple augmentation: horizontal flipping
  frames = tf.image.random_flip_left_right(frames)
  return frames


