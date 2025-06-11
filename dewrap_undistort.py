import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import glob
from typing import Tuple, List, Optional

class CompleteCameraDewarper:
    def __init__(self):
        # Lens distortion parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.lens_calibrated = False
        
        # Perspective correction parameters
        self.src_points = None
        self.dst_points = None
        self.transform_matrix = None
        self.output_size = None
        
    def calibrate_camera_lens(self, calibration_images: List[str], pattern_size: Tuple[int, int] = (9, 6)) -> bool:
        """
        Calibrate camera to remove lens distortion using checkerboard pattern
        
        Args:
            calibration_images: List of paths to checkerboard calibration images
            pattern_size: (width, height) of inner corners in checkerboard
            
        Returns:
            True if calibration successful, False otherwise
        """
        # Prepare object points (3D coordinates of checkerboard corners)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        successful_detections = 0
        
        for img_path in calibration_images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions for subpixel accuracy
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
                successful_detections += 1
                print(f"✓ Detected pattern in {img_path}")
            else:
                print(f"✗ No pattern found in {img_path}")
        
        if successful_detections < 3:
            print(f"Error: Need at least 3 successful detections, got {successful_detections}")
            return False
        
        print(f"Calibrating with {successful_detections} images...")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                          gray.shape[::-1], None, None)
        
        if ret:
            self.camera_matrix = mtx
            self.distortion_coeffs = dist
            self.lens_calibrated = True
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            mean_error = total_error / len(objpoints)
            print(f"✓ Camera calibration successful!")
            print(f"Mean reprojection error: {mean_error:.3f} pixels")
            
            return True
        else:
            print("✗ Camera calibration failed")
            return False
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from image using calibrated camera parameters
        """
        if not self.lens_calibrated:
            print("Warning: No lens calibration available. Returning original image.")
            return image
        
        # Get optimal new camera matrix
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, 
                                                         self.distortion_coeffs, 
                                                         (w, h), 1, (w, h))
        
        # Undistort image
        undistorted = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs, 
                                   None, newcameramtx)
        
        # Crop the image based on ROI (optional)
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
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
    
    def dewarp_perspective(self, image: np.ndarray, src_points: np.ndarray, 
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
    
    def complete_dewarp(self, image: np.ndarray, src_points: np.ndarray = None, 
                       output_size: Optional[Tuple[int, int]] = None, 
                       method: str = 'auto') -> np.ndarray:
        """
        Complete dewarping pipeline: lens distortion removal + perspective correction
        
        Args:
            image: Input distorted image
            src_points: Source points for perspective correction (if None, will detect/select)
            output_size: Output image size
            method: 'auto', 'manual', or 'points' (if src_points provided)
        """
        # Step 1: Remove lens distortion
        print("Step 1: Removing lens distortion...")
        undistorted = self.undistort_image(image)
        
        # Step 2: Perspective correction
        print("Step 2: Correcting perspective...")
        
        if method == 'auto' or (method == 'points' and src_points is None):
            detected_points = self.auto_detect_rectangle(undistorted)
            if detected_points is None:
                print("Auto detection failed, switching to manual selection...")
                method = 'manual'
            else:
                src_points = detected_points
                
        if method == 'manual':
            src_points = np.array(self.manual_point_selection(undistorted))
        
        if src_points is None or len(src_points) != 4:
            raise ValueError("Need exactly 4 points for perspective correction")
        
        # Apply perspective correction to undistorted image
        final_result = self.dewarp_perspective(undistorted, src_points, output_size)
        
        return final_result
    
    def save_complete_calibration(self, filename: str, format: str = 'numpy') -> None:
        """
        Save both lens calibration and perspective calibration
        """
        calibration_data = {
            # Lens calibration
            'camera_matrix': self.camera_matrix,
            'distortion_coeffs': self.distortion_coeffs,
            'lens_calibrated': self.lens_calibrated,
            # Perspective correction
            'transform_matrix': self.transform_matrix,
            'src_points': self.src_points,
            'dst_points': self.dst_points,
            'output_size': self.output_size
        }
        
        if format.lower() == 'numpy':
            if not filename.endswith('.npz'):
                filename += '.npz'
            # Handle None values for numpy
            save_data = {}
            for key, value in calibration_data.items():
                if value is not None:
                    save_data[key] = value
            np.savez_compressed(filename, **save_data)
            
        elif format.lower() == 'pickle':
            if not filename.endswith('.pkl'):
                filename += '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(calibration_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            raise ValueError("Format must be 'numpy' or 'pickle' for complete calibration")
        
        print(f"Complete calibration saved to: {filename}")
    
    def load_complete_calibration(self, filename: str, format: str = None) -> None:
        """
        Load both lens calibration and perspective calibration
        """
        if format is None:
            if filename.endswith('.npz'):
                format = 'numpy'
            elif filename.endswith('.pkl'):
                format = 'pickle'
            else:
                raise ValueError("Cannot infer format. Please specify format parameter.")
        
        if format.lower() == 'numpy':
            data = np.load(filename, allow_pickle=True)
            self.camera_matrix = data.get('camera_matrix', None)
            self.distortion_coeffs = data.get('distortion_coeffs', None)
            self.lens_calibrated = bool(data.get('lens_calibrated', False))
            self.transform_matrix = data.get('transform_matrix', None)
            self.src_points = data.get('src_points', None)
            self.dst_points = data.get('dst_points', None)
            self.output_size = tuple(data['output_size']) if 'output_size' in data else None
            
        elif format.lower() == 'pickle':
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            for key, value in data.items():
                setattr(self, key, value)
        
        print(f"Complete calibration loaded from: {filename}")
    
    def apply_complete_calibration(self, image: np.ndarray) -> np.ndarray:
        """
        Apply both lens distortion removal and perspective correction to new image
        """
        # Step 1: Remove lens distortion
        undistorted = self.undistort_image(image)
        
        # Step 2: Apply perspective correction
        if self.transform_matrix is None:
            raise ValueError("No perspective calibration loaded.")
        
        dewarped = cv2.warpPerspective(undistorted, self.transform_matrix, self.output_size)
        return dewarped
    
    def generate_checkerboard_calibration_target(self, filename: str = "checkerboard.png", 
                                               pattern_size: Tuple[int, int] = (9, 6),
                                               square_size: int = 50) -> None:
        """
        Generate a checkerboard calibration target for printing
        """
        rows, cols = pattern_size[1] + 1, pattern_size[0] + 1
        
        # Create checkerboard pattern
        board = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)
        
        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 == 0:
                    board[i*square_size:(i+1)*square_size, 
                          j*square_size:(j+1)*square_size] = 255
        
        cv2.imwrite(filename, board)
        print(f"Checkerboard calibration target saved as: {filename}")
        print(f"Print this image and use it for camera calibration")
        print(f"Pattern size: {pattern_size[0]}x{pattern_size[1]} inner corners")

def demo_complete_calibration():
    """
    Complete calibration workflow demonstration
    """
    dewarper = CompleteCameraDewarper()
    
    print("=== Complete Camera Dewarping Demo ===\n")
    
    # Step 1: Generate calibration target (optional)
    print("Step 1: Generate calibration checkerboard...")
    dewarper.generate_checkerboard_calibration_target("calibration_checkerboard.png")
    
    # Step 2: Camera lens calibration
    print("\nStep 2: Camera lens calibration...")
    # You would need multiple checkerboard images for real calibration
    calibration_images = glob.glob("calibration_images/*.jpg")  # Your calibration images
    
    if len(calibration_images) > 0:
        success = dewarper.calibrate_camera_lens(calibration_images)
        if success:
            print("✓ Lens calibration successful!")
        else:
            print("✗ Lens calibration failed, continuing without lens correction...")
    else:
        print("No calibration images found, skipping lens calibration...")
        print("Place checkerboard images in 'calibration_images/' folder")
    
    # Step 3: Load test image
    print("\nStep 3: Processing test image...")
    test_image = cv2.imread("test_image.jpg")  # Your test image
    
    if test_image is None:
        print("No test image found, creating sample...")
        # Create sample distorted image for demo
        test_image = create_sample_distorted_image()
    
    # Step 4: Complete dewarping
    try:
        result = dewarper.complete_dewarp(test_image, method='auto')
        
        # Save calibration for future use
        dewarper.save_complete_calibration("complete_calibration")
        
        # Display results
        cv2.imshow("Original", test_image)
        cv2.imshow("Corrected", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("✓ Complete dewarping successful!")
        
    except Exception as e:
        print(f"Error during dewarping: {e}")

def create_sample_distorted_image():
    """Create a sample image with both lens distortion and perspective distortion"""
    # Create base image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (550, 350), (0, 0, 0), 2)
    cv2.putText(img, "DISTORTED SAMPLE", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Apply perspective distortion
    src_pts = np.float32([[50, 50], [550, 50], [550, 350], [50, 350]])
    dst_pts = np.float32([[30, 80], [520, 40], [570, 320], [80, 360]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img = cv2.warpPerspective(img, matrix, (600, 400))
    
    # Simulate lens distortion (barrel distortion)
    h, w = img.shape[:2]
    # Create fake camera matrix and distortion coefficients
    camera_matrix = np.array([[w*0.7, 0, w/2], [0, h*0.7, h/2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([-0.2, 0.1, 0, 0, 0], dtype=np.float32)  # Barrel distortion
    
    # Apply distortion (reverse of undistortion)
    img_points = np.mgrid[0:w:1, 0:h:1].T.reshape(-1, 2).astype(np.float32)
    distorted_points = cv2.undistortPoints(img_points, camera_matrix, dist_coeffs, P=camera_matrix)
    
    return img

if __name__ == "__main__":
    demo_complete_calibration()