import cv2
import numpy as np
import os
import csv
import time

def apply_kernel_edge_detection(image, kernel_x, kernel_y):
    """Applies custom kernels for edge detection."""
    image_float = image.astype(np.float32)
    
    grad_x = cv2.filter2D(image_float, -1, kernel_x)
    grad_y = cv2.filter2D(image_float, -1, kernel_y)
    
    magnitude = cv2.magnitude(grad_x, grad_y)
    cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    
    return magnitude.astype(np.uint8)

def region_of_interest(img):
    """Applies a mask to keep only the road area."""
    height, width = img.shape
    mask = np.zeros_like(img)
    
    # Define a trapezoid focusing on the bottom half of the image
    polygons = np.array([
        [(int(width * 0.1), height), 
         (int(width * 0.45), int(height * 0.6)), 
         (int(width * 0.55), int(height * 0.6)), 
         (int(width * 0.9), height)]
    ])
    
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 255, 0], thickness=5):
    """Draws lines on the image."""
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img

def process_frame(original_image, algorithm, delay=0):
    """
    Processes a single frame/image.
    Returns: final_image, edges, masked_edges, time_taken
    """
    start_time = cv2.getTickCount()

    # 1. Preprocessing (Grayscale + Blur)
    if len(original_image.shape) == 2:
        gray = original_image
    else:
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Edge Detection
    edges = None
    if algorithm == 'Canny':
        edges = cv2.Canny(blur, 50, 150)
        
    elif algorithm == 'Sobel':
        grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        _, edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    elif algorithm == 'Prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        edges = apply_kernel_edge_detection(blur, kernel_x, kernel_y)
        _, edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    elif algorithm == 'Roberts':
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        edges = apply_kernel_edge_detection(blur, kernel_x, kernel_y)
        _, edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    elif algorithm == 'Laplacian':
        laplacian = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
        edges = cv2.convertScaleAbs(laplacian)
        _, edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    # 3. ROI Masking
    masked_edges = region_of_interest(edges)

    # 4. Hough Transform
    hough_threshold = 50 if algorithm == 'Canny' else 80
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, hough_threshold, minLineLength=50, maxLineGap=10)

    # 5. Overlay Lines
    line_image = draw_lines(original_image, lines)
    
    # Ensure original_image is 3-channel for overlay
    if len(original_image.shape) == 2:
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_image_rgb = original_image
        
    final_image = cv2.addWeighted(original_image_rgb, 0.8, line_image, 1, 0)
    
    # Simulate Processing Delay (Raspberry Pi Mode)
    if delay > 0:
        time.sleep(delay)
    
    end_time = cv2.getTickCount()
    time_taken = (end_time - start_time) / cv2.getTickFrequency()
    
    return final_image, edges, masked_edges, gray, time_taken

def process_video(video_path, algorithm, output_folder, progress_callback=None, delay=0):
    """
    Processes a video file frame by frame.
    Saves metrics to CSV and output video to disk.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error opening video file"

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_folder, f"{video_name}_{algorithm}_out.mp4")
    csv_path = os.path.join(output_folder, "results.csv")
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    metrics = []
    frame_count = 0
    
    # Check if CSV exists to write header
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['Video Name', 'Algorithm', 'Frame', 'Computation Time (s)', 'FPS']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process Frame
            final_img, _, _, _, time_taken = process_frame(frame_rgb, algorithm, delay=delay)
            
            # Convert back to BGR for saving
            final_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            out.write(final_bgr)
            
            # Log metrics
            current_fps = 1.0 / time_taken if time_taken > 0 else 0
            row = {
                'Video Name': video_name,
                'Algorithm': algorithm,
                'Frame': frame_count,
                'Computation Time (s)': f"{time_taken:.4f}",
                'FPS': f"{current_fps:.2f}"
            }
            writer.writerow(row)
            metrics.append(row)
            
            frame_count += 1
            
            if progress_callback:
                progress_callback(frame_count / total_frames)

    cap.release()
    out.release()
    
    return metrics, output_video_path

def process_batch_images(images, filenames, algorithm, output_folder, progress_callback=None, delay=0):
    """
    Processes a batch of images.
    Saves processed images and logs metrics to CSV.
    """
    csv_path = os.path.join(output_folder, "results.csv")
    metrics = []
    
    # Check if CSV exists to write header
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['Input Name', 'Algorithm', 'Type', 'Computation Time (s)', 'FPS']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        for i, (img_array, filename) in enumerate(zip(images, filenames)):
            # Process Frame
            final_img, edges, _, gray, time_taken = process_frame(img_array, algorithm, delay=delay)
            
            # Save processed image (Visual)
            save_path = os.path.join(output_folder, f"processed_{algorithm}_{filename}")
            final_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, final_bgr)
            
            # Save raw edges (For Validation/FOM)
            edge_save_path = os.path.join(output_folder, f"edges_{algorithm}_{filename}")
            cv2.imwrite(edge_save_path, edges)
            
            # Save grayscale image
            gray_save_path = os.path.join(output_folder, f"gray_{algorithm}_{filename}")
            cv2.imwrite(gray_save_path, gray)
            
            # Log metrics
            current_fps = 1.0 / time_taken if time_taken > 0 else 0
            row = {
                'Input Name': filename,
                'Algorithm': algorithm,
                'Type': 'Image',
                'Computation Time (s)': f"{time_taken:.4f}",
                'FPS': f"{current_fps:.2f}"
            }
            writer.writerow(row)
            metrics.append(row)
            
            if progress_callback:
                progress_callback((i + 1) / len(images))
                
    return metrics

def calculate_fom(detected_edges, ground_truth_edges, alpha=1.0/9.0):
    """
    Calculates Pratt's Figure of Merit (FOM).
    detected_edges: Binary image of detected edges (0 or 255).
    ground_truth_edges: Binary or Color image of ideal edges.
    alpha: Scaling constant (typically 1/9).
    """
    # Handle GT image format
    if len(ground_truth_edges.shape) == 3:
        # If Color, convert to grayscale
        gt_gray = cv2.cvtColor(ground_truth_edges, cv2.COLOR_BGR2GRAY)
    else:
        gt_gray = ground_truth_edges

    # Thresholding
    # Note: GT lanes might be colored (e.g., Magenta), which converts to ~105 gray.
    # We use a low threshold (10) to ensure we capture any non-black markings.
    _, detected_bin = cv2.threshold(detected_edges, 127, 255, cv2.THRESH_BINARY)
    _, gt_bin = cv2.threshold(gt_gray, 10, 255, cv2.THRESH_BINARY)
    
    # Count ideal edge points
    N_i = cv2.countNonZero(gt_bin)
    # Count detected edge points
    N_a = cv2.countNonZero(detected_bin)
    
    if N_i == 0:
        return 0.0 # Avoid division by zero if GT is empty
        
    # Distance Transform on inverted GT to get distance to nearest ideal edge pixel
    # Invert GT: edges become 0, background 255
    gt_inv = cv2.bitwise_not(gt_bin)
    dist_map = cv2.distanceTransform(gt_inv, cv2.DIST_L2, 3)
    
    # Summation part
    fom_sum = 0.0
    
    # Iterate over detected edge points
    # We can use numpy masking for speed instead of looping
    # Get coordinates of detected edges
    y_coords, x_coords = np.where(detected_bin > 0)
    
    if len(y_coords) == 0:
        return 0.0
        
    # Get distances for these points from the distance map
    distances = dist_map[y_coords, x_coords]
    
    # Calculate metric
    fom_sum = np.sum(1.0 / (1.0 + alpha * (distances ** 2)))
    
    # Final FOM
    N_max = max(N_i, N_a)
    fom = (1.0 / N_max) * fom_sum
    
    return fom
