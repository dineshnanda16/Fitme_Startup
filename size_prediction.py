import cv2
import numpy as np
import mediapipe as mp


# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Load the image
image_path = 'img.jpg'
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

# Use Hough Line Transform to detect straight lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Variables to store the best line (representing the scale)
best_line = None
max_length = 0

# Loop through all detected lines and filter based on length
for line in lines:
    for x1, y1, x2, y2 in line:
        length = calculate_distance((x1, y1), (x2, y2))
        if length > max_length:
            max_length = length
            best_line = (x1, y1, x2, y2)

# If a scale line was detected
if best_line is not None:
    x1, y1, x2, y2 = best_line
    
    # Draw the detected scale on the image for visualization
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Calculate the length of the scale in pixels
    scale_length_pixels = calculate_distance((x1, y1), (x2, y2))
    
    # Known length of the scale in centimeters
    scale_length_cm = 23.0
    
    # Calculate the conversion factor (cm per pixel)
    cm_per_pixel = scale_length_cm / scale_length_pixels
    
    print(f"Detected scale length in pixels: {scale_length_pixels:.2f}")
    print(f"Conversion factor: {cm_per_pixel:.4f} cm/pixel")
    
    # The remaining code for pose detection and measurement calculation would follow here.

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Convert to RGB for MediaPipe processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image to detect pose
results = pose.process(image_rgb)

# Draw pose landmarks on the image
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    landmarks = results.pose_landmarks.landmark
    image_height, image_width, _ = image.shape

    # Extract points for shoulder width
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_shoulder_coords = (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height))
    right_shoulder_coords = (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height))
    shoulder_width_pixels = calculate_distance(left_shoulder_coords, right_shoulder_coords)
    shoulder_width_cm = shoulder_width_pixels * cm_per_pixel

    # Estimate neck position as midpoint between shoulders
    neck_coords = (
        (left_shoulder_coords[0] + right_shoulder_coords[0]) // 2,
        (left_shoulder_coords[1] + right_shoulder_coords[1]) // 2
    )

    # Extract points for front length (distance from neck to waist)
    waist_midpoint = (
        int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2 * image_width),
        int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2 * image_height)
    )
    front_length_pixels = calculate_distance(neck_coords, waist_midpoint)
    front_length_cm = front_length_pixels * cm_per_pixel

    # Extract points for waist width
    left_waist_coords = (
        int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_width),
        int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_height)
    )
    right_waist_coords = (
        int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width),
        int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height)
    )
    waist_width_pixels = calculate_distance(left_waist_coords, right_waist_coords)
    waist_width_cm = waist_width_pixels * cm_per_pixel

    # Display results
    print(f'Shoulder Width: {shoulder_width_cm:.2f} cm')
    print(f'Front Length: {front_length_cm:.2f} cm')
    print(f'Waist Width: {waist_width_cm:.2f} cm')

# Display the final image
cv2.imshow('Final Image with Measurements', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
