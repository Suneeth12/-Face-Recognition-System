# Face Recognition System

This project allows you to locate and extract occurrences of a target face from a set of security camera video files, using state-of-the-art deep learning techniques for face detection, alignment, and recognition. Output videos are generated with bounding boxes and camera/time labels where the target face is identified.

## Features

- **Robust Face Detection:** Fast, accurate face detection and alignment using MTCNN pretrained model.
- **Deep Face Recognition:** Compares faces using embeddings from InceptionResnetV1 (VGGFace2 pretrained).
- **Batch Video Search:** Automatically processes multiple surveillance videos (e.g., cam1.mp4 to cam7.mp4).
- **Result Visualization:** Saves new video clips with the target face highlighted, including timestamp and camera label overlays.
- **GPU/CPU Compatible:** Runs on GPU if available, or defaults to CPU.

## Technologies Used

- Python 3.x
- OpenCV
- NumPy
- PyTorch
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MTCNN for detection, InceptionResnetV1 for recognition)
- Pillow (`PIL`)
- Standard Library modules: `os`

