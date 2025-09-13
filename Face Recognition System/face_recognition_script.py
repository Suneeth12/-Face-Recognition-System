import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Paths
DATA_FOLDER = r"C:\Users\saiku\OneDrive\Desktop\DV Lab\FaceRecognitionProject\data"
OUTPUT_FOLDER = r"C:\Users\saiku\OneDrive\Desktop\DV Lab\FaceRecognitionProject\output"
TARGET_IMAGE_PATH = os.path.join(DATA_FOLDER, "target.jpeg")
CAM_VIDEOS = [f"cam{i}.mp4" for i in range(1, 8)]  # cam1.mp4 to cam7.mp4

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize MTCNN for face detection and InceptionResnetV1 for embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Process the target face ---
target_img_cv = cv2.imread(TARGET_IMAGE_PATH)
if target_img_cv is None:
    raise FileNotFoundError(f"Target image not found at: {TARGET_IMAGE_PATH}")

target_img_rgb = cv2.cvtColor(target_img_cv, cv2.COLOR_BGR2RGB)
target_pil = Image.fromarray(target_img_rgb)
# Get the aligned face from the target image using MTCNN
target_face = mtcnn(target_pil)
if target_face is None:
    raise ValueError("No face detected in the target image.")

# Ensure target_face has 3 channels (if it's grayscale, repeat the channel)
if target_face.ndim == 3 and target_face.shape[0] == 1:
    target_face = target_face.repeat(3, 1, 1)

with torch.no_grad():
    target_embedding = resnet(target_face.unsqueeze(0).to(device)).cpu().numpy()

# --- Video Processing Function ---
def process_video(video_name):
    """Processes a single video to extract frames with the target person."""
    video_path = os.path.join(DATA_FOLDER, video_name)
    output_path = os.path.join(OUTPUT_FOLDER, video_name.replace(".mp4", "_processed.mp4"))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize VideoWriter (MP4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    camera_label = video_name.replace(".mp4", "")  # e.g., cam1, cam2, etc.
    frame_number = 0
    written_frames = 0  # Count of frames written to output
    threshold = 1.0  # Euclidean distance threshold for a match (adjust as needed)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # Detect faces using MTCNN
        boxes, probs = mtcnn.detect(pil_frame)
        match_found = False
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                # Use MTCNN.extract to get the aligned face. Convert box to a NumPy array with shape (1,4)
                face_tensor_list = mtcnn.extract(pil_frame, np.array([box]), None)
                if face_tensor_list is None or len(face_tensor_list) == 0:
                    continue

                face_tensor = face_tensor_list[0].to(device)
                # Ensure the face tensor has 3 channels
                if face_tensor.ndim == 3 and face_tensor.shape[0] == 1:
                    face_tensor = face_tensor.repeat(3, 1, 1)
                elif face_tensor.ndim == 2:
                    # If shape is (H, W), add a channel dimension and then repeat it
                    face_tensor = face_tensor.unsqueeze(0).repeat(3, 1, 1)

                with torch.no_grad():
                    # Add batch dimension and compute embedding
                    input_tensor = face_tensor.unsqueeze(0)
                    face_embedding = resnet(input_tensor).cpu().numpy()

                # Compare embeddings using Euclidean distance
                dist = np.linalg.norm(face_embedding - target_embedding)
                if dist < threshold:
                    match_found = True
                    break  # Stop after first match

        if match_found:
            timestamp = frame_number / fps
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Camera: {camera_label}, Time: {timestamp:.2f}s"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(frame)
            written_frames += 1

        frame_number += 1

    cap.release()
    out.release()

    if written_frames == 0:
        # Delete the output file if no matching frames were written
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"No matching frames found in {video_name}. Output file removed.")
    else:
        print(f"Processed {video_name} → {output_path} ({written_frames} frames written)")

# Process all videos
for cam in CAM_VIDEOS:
    process_video(cam)
