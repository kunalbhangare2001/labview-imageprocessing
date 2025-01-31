import cv2
import os

def detect_motion(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Error: Failed to open video {video_path}"
    
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    frame_count = 0
    while ret:
        # Calculate the absolute difference between two frames
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the frame
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the processed frame
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.png"), frame1)
        frame_count += 1
        
        frame1 = frame2
        ret, frame2 = cap.read()
    
    cap.release()
    return f"{output_dir}"
