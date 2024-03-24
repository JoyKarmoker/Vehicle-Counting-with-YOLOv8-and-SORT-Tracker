# Vehicle Counting with YOLOv8 and SORT Tracker

![Vehicle Counting Demo](demo.gif)

This project utilizes YOLOv8 for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for tracking to count vehicles passing through a specified region in a video. It detects vehicles such as cars, trucks, buses, and motorbikes, tracks them across frames, and provides a total count of vehicles that have crossed a predefined limit line.

## Overview

In this project, we combine the power of two state-of-the-art algorithms:
- **YOLOv8**: An efficient object detection model capable of real-time inference. It identifies various types of vehicles in each frame of the video.
- **SORT Tracker**: A lightweight and simple online tracking algorithm that associates detections across frames, enabling us to track individual vehicles as they move through the scene.

## Features

- Object detection using YOLOv8 to identify vehicles in a video.
- Real-time vehicle tracking with the SORT algorithm.
- Counting the total number of vehicles passing through a specific region.
- Visualization of bounding boxes and tracking IDs on the video frames.

## Usage

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/your_username/vehicle-counting.git
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download Pretrained YOLOv8 Weights**:

    Download the YOLOv8 weights file (e.g., `yolov8l.pt`) and place it in the `Yolo-Weights` directory.

4. **Run the Code**:

    Replace the video path in the script with your desired video file, and then execute the script:

    ```bash
    python vehicle_counting.py
    ```

5. **View the Output**:

    Once the processing is complete, you'll find the output video (`output_video.mp4`) in the project directory.

## Customization

- Adjust the confidence threshold (`conf > 0.3`) in the code to modify the sensitivity of vehicle detection.
- Modify the region limits (`limits`) to define a specific area for vehicle counting.

## Ouput

Check out the demo video to see the vehicle counting system in action!


https://github.com/JoyKarmoker/YOLOv8-Vehicle-Counting-System-Real-Time-Tracking-and-Counting-of-Vehicles-in-Videos/assets/48152122/18762e2c-0b86-4810-adf0-e6c94c0344fd


## Credits

- **YOLOv8** by Ultralytics for the object detection implementation.
- **SORT Tracker** for the online tracking algorithm.
- **OpenCV** for image and video processing.
- **cvzone** for drawing bounding boxes and text on images.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
