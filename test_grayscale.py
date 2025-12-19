import cv2
import numpy as np
from utils import process_frame

def test_process_frame():
    print("Testing process_frame with RGB input...")
    # Create a dummy RGB image (100x100x3)
    rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(rgb_image, (10, 10), (90, 90), (255, 255, 255), 2)
    
    try:
        final_img, edges, masked_edges, gray, time_taken = process_frame(rgb_image, 'Canny')
        print("RGB Input: Success")
    except Exception as e:
        print(f"RGB Input: Failed with error: {e}")

    print("\nTesting process_frame with Grayscale input...")
    # Create a dummy Grayscale image (100x100)
    gray_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(gray_image, (10, 10), (90, 90), 255, 2)
    
    try:
        final_img, edges, masked_edges, gray, time_taken = process_frame(gray_image, 'Canny')
        print("Grayscale Input: Success")
    except Exception as e:
        print(f"Grayscale Input: Failed with error: {e}")

if __name__ == "__main__":
    test_process_frame()
