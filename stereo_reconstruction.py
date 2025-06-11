import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import pickle
from typing import Tuple, List, Optional, Dict
from dewrap import TiltCameraDewarper  # Import your existing dewarper

class StereoReconstructor:
    def __init__(self):
        self.InfDepth = 1000000
        self.left_dewarper = TiltCameraDewarper()
        self.right_dewarper = TiltCameraDewarper()

        # Camera calibration parameters: intrinsic matrix and distortion coeffs
        # See: https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html
        self.left_camera_matrix = None
        self.left_dist_coeffs = None
        self.right_camera_matrix = None
        self.right_dist_coeffs = None

        # Stereo calibration parameters
        self.rotation_matrix = None
        self.translation_vector = None
        self.essential_matrix = None
        self.fundamental_matrix = None

        # Rectification parameters
        self.rectification_transform_left = None
        self.rectification_transform_right = None
        self.projection_matrix_left = None
        self.projection_matrix_right = None
        self.disparity_to_depth_mapping = None

        # Stereo matcher
        self.stereo_matcher = None

    def calibrate_single_camera(self, images: List[np.ndarray], 
                                pattern_size: Tuple[int, int] = (9, 6),
                                square_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate a single camera using checkerboard pattern

        Args:
            images: List of calibration images
            pattern_size: (width, height) of checkerboard internal corners
            square_size: Size of checkerboard squares in world units

        Returns:
            camera_matrix, distortion_coefficients
        """
        # Prepare object points (3D points in real world space)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) # Assuming points at z=0 plane
        objp *= square_size

        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                objpoints.append(objp)

                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)

        # Camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

        return camera_matrix, dist_coeffs

    def calibrate_stereo_cameras(self, left_images: List[np.ndarray], 
                                 right_images: List[np.ndarray],
                                 pattern_size: Tuple[int, int] = (9, 6),
                                 square_size: float = 1.0) -> None:
        """
        Calibrate stereo camera system

        Reference: https://stackoverflow.com/a/58188465/18736354
        """
        if len(left_images) != len(right_images):
            raise ValueError("Number of left and right images must be equal")

        # Calibrate individual cameras first
        print("Calibrating left camera...")
        self.left_camera_matrix, self.left_dist_coeffs = self.calibrate_single_camera(
                left_images, pattern_size, square_size)

        print("Calibrating right camera...")
        self.right_camera_matrix, self.right_dist_coeffs = self.calibrate_single_camera(
                right_images, pattern_size, square_size)

        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints = []
        imgpoints_left = []
        imgpoints_right = []

        # Find corresponding points in stereo pairs
        for left_img, right_img in zip(left_images, right_images):
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

            if ret_left and ret_right:
                objpoints.append(objp)

                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1),
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1),
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)

        print("Performing stereo calibration...")
        # Stereo calibration
        image_size = gray_left.shape[::-1]
        ret, _, _, _, _, self.rotation_matrix, self.translation_vector, \
                self.essential_matrix, self.fundamental_matrix = cv2.stereoCalibrate(
                        objpoints, imgpoints_left, imgpoints_right,
                        self.left_camera_matrix, self.left_dist_coeffs,
                        self.right_camera_matrix, self.right_dist_coeffs,
                        image_size,
                        flags=cv2.CALIB_FIX_INTRINSIC
                        )

        print("Computing rectification transforms...")
        # Stereo rectification
        self.rectification_transform_left, self.rectification_transform_right, \
                self.projection_matrix_left, self.projection_matrix_right, \
                self.disparity_to_depth_mapping, _, _ = cv2.stereoRectify(
                        self.left_camera_matrix, self.left_dist_coeffs,
                        self.right_camera_matrix, self.right_dist_coeffs,
                        image_size, self.rotation_matrix, self.translation_vector
                        )

        print("Stereo calibration completed!")

    def setup_stereo_matcher(self, matcher_type: str = 'SGBM') -> None:
        """
        Setup stereo matching algorithm

        Args:
            matcher_type: 'BM' for StereoBM or 'SGBM' for StereoSGBM
        """
        if matcher_type == 'BM':
            self.stereo_matcher = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)
        elif matcher_type == 'SGBM':
            # Semi-Global Block Matching (better quality)
            self.stereo_matcher = cv2.StereoSGBM_create(
                    minDisparity=0,
                    numDisparities=16*5,  # Max disparity minus min disparity
                    blockSize=5,
                    P1=8 * 3 * 5**2,  # Controls disparity smoothness
                    P2=32 * 3 * 5**2,  # Controls disparity smoothness
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32
                    )
        else:
            raise ValueError("matcher_type must be 'BM' or 'SGBM'")

    def rectify_images(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair to align epipolar lines horizontally
        """
        if self.rectification_transform_left is None:
            raise ValueError("Stereo calibration not performed. Call calibrate_stereo_cameras() first")

        h, w = left_img.shape[:2]

        # Generate rectification maps
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
                self.left_camera_matrix, self.left_dist_coeffs,
                self.rectification_transform_left, self.projection_matrix_left,
                (w, h), cv2.CV_16SC2
                )

        right_map1, right_map2 = cv2.initUndistortRectifyMap(
                self.right_camera_matrix, self.right_dist_coeffs,
                self.rectification_transform_right, self.projection_matrix_right,
                (w, h), cv2.CV_16SC2
                )

        # Apply rectification
        left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

        return left_rectified, right_rectified

    def compute_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        Compute disparity map from rectified stereo pair
        """
        if self.stereo_matcher is None:
            self.setup_stereo_matcher()

        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img

        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)

        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0

        return disparity

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map using calibration parameters
        """
        if self.disparity_to_depth_mapping is None:
            raise ValueError("Stereo calibration not performed")

        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.1)

        # Calculate depth: depth = (focal_length * baseline) / disparity
        focal_length = self.disparity_to_depth_mapping[2, 3]  # fx * baseline
        depth = focal_length / disparity_safe

        # Set invalid disparities to a large depth
        depth[disparity <= 0] = self.InfDepth

        return depth

    def triangulate_points(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        Perform 3D reconstruction from stereo image pair

        Returns:
            points_3d: Nx3 array of 3D points
        """
        # First, dewarp images using your existing dewarper if needed
        # (Optional: uncomment if you want to dewarp before stereo processing)
        left_img = self.left_dewarper.apply_saved_calibration(left_img)
        right_img = self.right_dewarper.apply_saved_calibration(right_img)

        # Rectify images
        left_rect, right_rect = self.rectify_images(left_img, right_img)

        # Compute disparity
        disparity = self.compute_disparity(left_rect, right_rect)

        # Convert to depth
        depth = self.disparity_to_depth(disparity)

        # Generate 3D points
        h, w = disparity.shape
        points_3d = []

        # Camera parameters
        fx = self.projection_matrix_left[0, 0]
        fy = self.projection_matrix_left[1, 1]
        cx = self.projection_matrix_left[0, 2]
        cy = self.projection_matrix_left[1, 2]

        for y in range(h):
            for x in range(w):
                if disparity[y, x] > 0:  # Valid disparity
                    z = depth[y, x]
                    if z < self.InfDepth:  # Filter out very far points
                        x_3d = (x - cx) * z / fx
                        y_3d = (y - cy) * z / fy
                        points_3d.append([x_3d, y_3d, z])

        return np.array(points_3d)

    def reconstruct_3d_field(self, left_img: np.ndarray, right_img: np.ndarray,
                             downsample_factor: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct 3D field with color information

        Args:
            left_img: Left camera image
            right_img: Right camera image
            downsample_factor: Factor to downsample for faster processing

        Returns:
            points_3d: Nx3 array of 3D coordinates
            colors: Nx3 array of RGB colors
        """
        # Downsample for faster processing
        if downsample_factor > 1:
            h, w = left_img.shape[:2]
            left_small = cv2.resize(left_img, (w//downsample_factor, h//downsample_factor))
            right_small = cv2.resize(right_img, (w//downsample_factor, h//downsample_factor))
        else:
            left_small = left_img
            right_small = right_img

        # Rectify images
        left_rect, right_rect = self.rectify_images(left_small, right_small)

        # Compute disparity
        disparity = self.compute_disparity(left_rect, right_rect)

        # Convert to depth
        depth = self.disparity_to_depth(disparity)

        # Generate 3D points with colors
        h, w = disparity.shape
        points_3d = []
        colors = []

        # Camera parameters (adjusted for downsampling)
        fx = self.projection_matrix_left[0, 0] / downsample_factor
        fy = self.projection_matrix_left[1, 1] / downsample_factor
        cx = self.projection_matrix_left[0, 2] / downsample_factor
        cy = self.projection_matrix_left[1, 2] / downsample_factor

        for y in range(0, h, 2):  # Skip some points for performance
            for x in range(0, w, 2):
                if disparity[y, x] > 0:  # Valid disparity
                    z = depth[y, x]
                    if 0.1 < z < 50:  # Filter reasonable depth range
                        x_3d = (x - cx) * z / fx
                        y_3d = (y - cy) * z / fy

                        points_3d.append([x_3d, y_3d, z])

                        # Get color from left rectified image
                        if len(left_rect.shape) == 3:
                            color = left_rect[y, x] / 255.0  # Normalize to 0-1
                            colors.append([color[2], color[1], color[0]])  # BGR to RGB
                        else:
                            gray_val = left_rect[y, x] / 255.0
                            colors.append([gray_val, gray_val, gray_val])

        return np.array(points_3d), np.array(colors)

    def visualize_3d_reconstruction(self, points_3d: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
        """
        Visualize 3D reconstruction result
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if colors is not None:
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                       c=colors, s=1, alpha=0.6)
        else:
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                       c=points_3d[:, 2], cmap='viridis', s=1, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Depth)')
        ax.set_title('3D Reconstruction')

        plt.show()

    def save_calibration(self, filename: str) -> None:
        """Save stereo calibration parameters"""
        calibration_data = {
                'left_camera_matrix': self.left_camera_matrix,
                'left_dist_coeffs': self.left_dist_coeffs,
                'right_camera_matrix': self.right_camera_matrix,
                'right_dist_coeffs': self.right_dist_coeffs,
                'rotation_matrix': self.rotation_matrix,
                'translation_vector': self.translation_vector,
                'essential_matrix': self.essential_matrix,
                'fundamental_matrix': self.fundamental_matrix,
                'rectification_transform_left': self.rectification_transform_left,
                'rectification_transform_right': self.rectification_transform_right,
                'projection_matrix_left': self.projection_matrix_left,
                'projection_matrix_right': self.projection_matrix_right,
                'disparity_to_depth_mapping': self.disparity_to_depth_mapping
                }

        np.savez_compressed(filename, **calibration_data)
        print(f"Stereo calibration saved to: {filename}")

    def load_calibration(self, filename: str) -> None:
        """Load stereo calibration parameters"""
        data = np.load(filename)

        self.left_camera_matrix = data['left_camera_matrix']
        self.left_dist_coeffs = data['left_dist_coeffs']
        self.right_camera_matrix = data['right_camera_matrix']
        self.right_dist_coeffs = data['right_dist_coeffs']
        self.rotation_matrix = data['rotation_matrix']
        self.translation_vector = data['translation_vector']
        self.essential_matrix = data['essential_matrix']
        self.fundamental_matrix = data['fundamental_matrix']
        self.rectification_transform_left = data['rectification_transform_left']
        self.rectification_transform_right = data['rectification_transform_right']
        self.projection_matrix_left = data['projection_matrix_left']
        self.projection_matrix_right = data['projection_matrix_right']
        self.disparity_to_depth_mapping = data['disparity_to_depth_mapping']

        print(f"Stereo calibration loaded from: {filename}")

def demo_stereo_reconstruction():
    """
    Demo function showing how to use stereo reconstruction
    """
    # Initialize reconstructor
    reconstructor = StereoReconstructor()

    # Example usage:

    # 1. Calibrate stereo cameras (you need checkerboard calibration images)
    left_calibration_images = [cv2.imread(f"left_calib_{i:02d}.jpg") for i in range(20)]
    right_calibration_images = [cv2.imread(f"right_calib_{i:02d}.jpg") for i in range(20)]
    reconstructor.calibrate_stereo_cameras(left_calibration_images, right_calibration_images,
                                           pattern_size=(8, 6), # A4 - 25mm squares - 8x6 verticies, 9x7 squares
                                           square_size=25) # [mm]
    reconstructor.save_calibration("stereo_calibration.npz")

    # 2. Load pre-saved calibration
    reconstructor.load_calibration("stereo_calibration.npz")

    # 3. Load stereo pair for reconstruction
    left_img = cv2.imread("left_image.jpg")
    right_img = cv2.imread("right_image.jpg")

    # 4. Optionally dewarp images first using your existing dewarper
    reconstructor.left_dewarper.load_calibration("left_camera_dewarp.npz")
    reconstructor.right_dewarper.load_calibration("right_camera_dewarp.npz")
    left_img = reconstructor.left_dewarper.apply_saved_calibration(left_img)
    right_img = reconstructor.right_dewarper.apply_saved_calibration(right_img)

    # 5. Perform 3D reconstruction
    points_3d, colors = reconstructor.reconstruct_3d_field(left_img, right_img)

    # 6. Visualize results
    reconstructor.visualize_3d_reconstruction(points_3d, colors)

    # 7. Save point cloud (optional)
    np.savetxt("point_cloud.xyz", points_3d)

    print("Stereo reconstruction demo setup complete!")
    print("Uncomment the relevant sections and provide your image paths to run.")

if __name__ == "__main__":
    demo_stereo_reconstruction()
