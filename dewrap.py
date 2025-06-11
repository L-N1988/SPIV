import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class TiltCameraDewarper:
    def __init__(self):
        self.src_points = None
        self.dst_points = None
        self.transform_matrix = None
        
    def manual_point_selection(self, image: np.ndarray, title: str = "Select 4 corners") -> List[Tuple[int, int]]:
        """
        Interactive point selection for manual correction.
        Click 4 corners in order: top-left, top-right, bottom-right, bottom-left
        """
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image_copy, f"{len(points)}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(title, image_copy)
        
        image_copy = image.copy()
        cv2.imshow(title, image_copy)
        cv2.setMouseCallback(title, mouse_callback)
        
        print("Click 4 corners in order: top-left, top-right, bottom-right, bottom-left")
        print("Press 'r' to reset, 'q' when done")
        
        while len(points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                points.clear()
                image_copy = image.copy()
                cv2.imshow(title, image_copy)
            elif key == ord('q'):
                break
                
        cv2.destroyAllWindows()
        return points
    
    def auto_detect_rectangle(self, image: np.ndarray, min_area: int = 1000) -> Optional[np.ndarray]:
        """
        Automatically detect rectangular objects in the image using contour detection.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular contour
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we found a quadrilateral
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        return None
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the order: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        
        return rect
    
    def calculate_output_size(self, src_points: np.ndarray) -> Tuple[int, int]:
        """
        Calculate the output dimensions based on the maximum width and height
        """
        # Calculate width
        width_top = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) + 
                           ((src_points[1][1] - src_points[0][1]) ** 2))
        width_bottom = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + 
                              ((src_points[2][1] - src_points[3][1]) ** 2))
        max_width = max(int(width_top), int(width_bottom))
        
        # Calculate height
        height_left = np.sqrt(((src_points[3][0] - src_points[0][0]) ** 2) + 
                             ((src_points[3][1] - src_points[0][1]) ** 2))
        height_right = np.sqrt(((src_points[2][0] - src_points[1][0]) ** 2) + 
                              ((src_points[2][1] - src_points[1][1]) ** 2))
        max_height = max(int(height_left), int(height_right))
        
        return max_width, max_height
    
    def dewarp_image(self, image: np.ndarray, src_points: np.ndarray, 
                     output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Perform perspective correction on the image
        """
        # Order the source points
        src_points = self.order_points(src_points.astype(np.float32))
        
        # Calculate output size if not provided
        if output_size is None:
            width, height = self.calculate_output_size(src_points)
        else:
            width, height = output_size
        
        # Define destination points (rectangle)
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective correction
        dewarped = cv2.warpPerspective(image, self.transform_matrix, (width, height))
        
        return dewarped
    
    def dewarp_automatic(self, image: np.ndarray, min_area: int = 1000) -> Optional[np.ndarray]:
        """
        Automatically detect and dewarp the largest rectangular object in the image
        """
        detected_points = self.auto_detect_rectangle(image, min_area)
        
        if detected_points is None:
            print("No rectangular object detected. Try manual selection.")
            return None
        
        print(f"Detected rectangle with corners: {detected_points}")
        return self.dewarp_image(image, detected_points)
    
    def dewarp_manual(self, image: np.ndarray) -> np.ndarray:
        """
        Manual point selection and dewarping
        """
        points = self.manual_point_selection(image)
        
        if len(points) != 4:
            raise ValueError("Need exactly 4 points for perspective correction")
        
        src_points = np.array(points, dtype=np.float32)
        return self.dewarp_image(image, src_points)
    
    def dewarp_with_known_points(self, image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Dewarp using predefined points
        """
        if len(points) != 4:
            raise ValueError("Need exactly 4 points for perspective correction")
        
        src_points = np.array(points, dtype=np.float32)
        return self.dewarp_image(image, src_points)

def demo_usage():
    """
    Demonstration of how to use the TiltCameraDewarper class
    """
    # Initialize dewarper
    dewarper = TiltCameraDewarper()
    
    # Load image (replace with your image path)
    image_path = "./figures/ssd_box2.jpg"  # Change this to your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load image: {image_path}")
        print("Creating a sample tilted image for demonstration...")
        
        # Create a sample image with text for demonstration
        sample_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(sample_img, (50, 50), (550, 350), (0, 0, 0), 2)
        cv2.putText(sample_img, "SAMPLE DOCUMENT", (100, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(sample_img, "This is a test image", (100, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(sample_img, "for dewarping demo", (100, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Apply a perspective transform to simulate tilt
        src_pts = np.float32([[50, 50], [550, 50], [550, 350], [50, 350]])
        dst_pts = np.float32([[30, 80], [520, 40], [570, 320], [80, 360]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        image = cv2.warpPerspective(sample_img, matrix, (600, 400))
        
    # Method 1: Automatic detection
    print("Trying automatic detection...")
    dewarped_auto = dewarper.dewarp_automatic(image.copy())
    
    if dewarped_auto is not None:
        cv2.imshow("Original", image)
        cv2.imshow("Auto Dewarped", dewarped_auto)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Automatic detection failed.")
    
    # Method 2: Manual selection (uncomment to use)
    print("Starting manual selection...")
    dewarped_manual = dewarper.dewarp_manual(image.copy())
    cv2.imshow("Manual Dewarped", dewarped_manual)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Method 3: Using known points
    # Example points (adjust for your image)
    # known_points = [(100, 100), (500, 80), (520, 300), (80, 320)]
    # dewarped_known = dewarper.dewarp_with_known_points(image, known_points)

if __name__ == "__main__":
    demo_usage()