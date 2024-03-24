from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Load the video file
cap = cv2.VideoCapture("../Videos/cars.mp4")

# Load the YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Output folder for saving frames
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

# Initialize the SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define the region limits for counting vehicles
limits = [400, 297, 673, 297]

# List to keep track of the total count of vehicles
totalCount = []

# Define the class names for YOLO detections
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load the mask and graphics images for region masking and overlaying
mask = cv2.imread('mask.png')
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)

# Main loop for processing video frames
while True:
    # Read the next frame from the video
    success, img = cap.read()

    # Break out of the loop if there are no more frames to read
    if not success:
        break

    # Apply mask to the frame to restrict detection to specific region
    imgRegion = cv2.bitwise_and(img, mask)

    # Overlay graphics on the frame
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # Perform object detection using YOLO model
    results = model(imgRegion, stream=True)

    # Initialize array to store detections
    detections = np.empty((0, 5))

    # Process YOLO detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate width and height of bounding box
            w, h = x2 - x1, y2 - y1

            # Extract confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Extract class index
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Check if the detected object is a vehicle and confidence is above threshold
            if currentClass in ['car', 'truck', 'bus', 'motorbike'] and conf > 0.3:
                # Store the detection information in an array
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update the SORT tracker with the detections
    resultsTracker = tracker.update(detections)

    # Draw region limits on the frame
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Process tracked results
    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw bounding box and label for each tracked object
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the tracked object crosses the limit line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display the total count of vehicles on the frame
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Save the frame with YOLO detections applied
    output_path = os.path.join(output_folder, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):06d}.jpg")
    cv2.imwrite(output_path, img)

    # Display the processed frame
    cv2.imshow("Image", img)

    # Wait for a key press to move to the next frame
    cv2.waitKey(1)

# Release video capture
cap.release()

# Recreate video from saved frames
output_video_path = "output_video.mp4"
img_array = []

# Iterate through saved frames and append to img_array
for filename in sorted(os.listdir(output_folder)):
    if filename.endswith(".jpg"):
        img_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

# Initialize video writer
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

# Write frames to output video
for i in range(len(img_array)):
    out.write(img_array[i])

# Release video writer
out.release()

# Print success message
print("Video created successfully.")
