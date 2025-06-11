import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from typing import Tuple, List, Optional

class TiltCameraDewarper:
    def __init__(self):
        self.src_points = None
        self.dst_points = None
        self.transform_matrix = None
        self.output_size = None
        
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
        
        # Store calibration data
        self.src_points = src_points
        self.dst_points = dst_points
        self.output_size = (width, height)
        
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
    
    def save_calibration(self, filename: str, format: str = 'numpy') -> None:
        """
        Save the calibration matrix and parameters to file
        
        Args:
            filename: Path to save the calibration file
            format: 'numpy' (npz), 'pickle' (pkl), or 'json' (limited precision)
        """
        if self.transform_matrix is None:
            raise ValueError("No calibration data available. Perform dewarping first.")
        
        calibration_data = {
            'transform_matrix': self.transform_matrix,
            'src_points': self.src_points,
            'dst_points': self.dst_points,
            'output_size': self.output_size
        }
        
        if format.lower() == 'numpy':
            # Save as numpy compressed format (.npz)
            if not filename.endswith('.npz'):
                filename += '.npz'
            np.savez_compressed(filename, **calibration_data)
            
        elif format.lower() == 'pickle':
            # Save as pickle format (.pkl)
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(calibration_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        elif format.lower() == 'json':
            # Save as JSON format (limited precision)
            if not filename.endswith('.json'):
                filename += '.json'
            # Convert numpy arrays to lists for JSON serialization
            json_data = {
                'transform_matrix': self.transform_matrix.tolist(),
                'src_points': self.src_points.tolist(),
                'dst_points': self.dst_points.tolist(),
                'output_size': self.output_size
            }
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
                
        else:
            raise ValueError("Format must be 'numpy', 'pickle', or 'json'")
        
        print(f"Calibration saved to: {filename}")
    
    def load_calibration(self, filename: str, format: str = None) -> None:
        """
        Load calibration matrix and parameters from file
        
        Args:
            filename: Path to the calibration file
            format: 'numpy', 'pickle', or 'json'. If None, inferred from filename
        """
        # Infer format from filename if not specified
        if format is None:
            if filename.endswith('.npz'):
                format = 'numpy'
            elif filename.endswith('.pkl'):
                format = 'pickle'
            elif filename.endswith('.json'):
                format = 'json'
            else:
                raise ValueError("Cannot infer format. Please specify format parameter.")
        
        if format.lower() == 'numpy':
            data = np.load(filename)
            self.transform_matrix = data['transform_matrix']
            self.src_points = data['src_points']
            self.dst_points = data['dst_points']
            self.output_size = tuple(data['output_size'])
            
        elif format.lower() == 'pickle':
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.transform_matrix = data['transform_matrix']
            self.src_points = data['src_points']
            self.dst_points = data['dst_points']
            self.output_size = data['output_size']
            
        elif format.lower() == 'json':
            with open(filename, 'r') as f:
                data = json.load(f)
            self.transform_matrix = np.array(data['transform_matrix'], dtype=np.float32)
            self.src_points = np.array(data['src_points'], dtype=np.float32)
            self.dst_points = np.array(data['dst_points'], dtype=np.float32)
            self.output_size = tuple(data['output_size'])
            
        else:
            raise ValueError("Format must be 'numpy', 'pickle', or 'json'")
        
        print(f"Calibration loaded from: {filename}")
    
    def apply_saved_calibration(self, image: np.ndarray) -> np.ndarray:
        """
        Apply previously saved calibration to a new image
        
        Args:
            image: Input image to dewarp
            
        Returns:
            Dewarped image using saved calibration
        """
        if self.transform_matrix is None:
            raise ValueError("No calibration loaded. Use load_calibration() first.")
        
        # Apply the stored transformation
        dewarped = cv2.warpPerspective(image, self.transform_matrix, self.output_size)
        return dewarped
    
    def batch_process_images(self, image_paths: List[str], output_dir: str = "dewarped/") -> None:
        """
        Apply saved calibration to multiple images
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save dewarped images
        """
        import os
        
        if self.transform_matrix is None:
            raise ValueError("No calibration loaded. Use load_calibration() first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for img_path in image_paths:
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                # Apply calibration
                dewarped = self.apply_saved_calibration(image)
                
                # Generate output filename
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(output_dir, f"{name}_dewarped{ext}")
                
                # Save dewarped image
                cv2.imwrite(output_path, dewarped)
                print(f"Processed: {img_path} -> {output_path}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    def visualize_calibration(self, original_image: np.ndarray) -> None:
        """
        Visualize the calibration points and transformation
        """
        if self.src_points is None:
            print("No calibration data available")
            return
        
        # Create visualization
        vis_img = original_image.copy()
        
        # Draw source points and connections
        pts = self.src_points.astype(int)
        for i, pt in enumerate(pts):
            cv2.circle(vis_img, tuple(pt), 8, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i+1), (pt[0]+10, pt[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw quadrilateral
        cv2.polylines(vis_img, [pts], True, (255, 0, 0), 2)
        
        cv2.imshow("Calibration Points", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
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
    image_path = "./figures/calibration_plate.jpg"  # Change this to your image path
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
    dewarper.save_calibration("calibration", format='numpy')
    
    # Method 3: Using known points
    # Example points (adjust for your image)
    # known_points = [(100, 100), (500, 80), (520, 300), (80, 320)]
    # dewarped_known = dewarper.dewarp_with_known_points(image, known_points)

if __name__ == "__main__":
    demo_usage()