import json
import os
import pickle
import sys
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class TiltCameraDewarper:
    def __init__(self):
        self.transform_matrix = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None

    def manual_point_selection(
        self, image: np.ndarray, title: str = "Select 4 corners"
    ) -> List[Tuple[int, int]]:
        """
        Interactive point selection for manual correction.
        Click 4 corners in order: top-left, top-right, bottom-right, bottom-left
        """
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    image_copy,
                    f"{len(points)}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(title, image_copy)

        image_copy = image.copy()
        cv2.imshow(title, image_copy)
        cv2.setMouseCallback(title, mouse_callback)

        print(
            "Click 4 corners in order: top-left, top-right, bottom-right, bottom-left"
        )
        print("Press 'r' to reset, 'q' when done")

        while len(points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                points.clear()
                image_copy = image.copy()
                cv2.imshow(title, image_copy)
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()
        return points

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the order: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def calculate_output_size(self, src_points: np.ndarray) -> Tuple[int, int]:
        """
        Calculate the output dimensions based on the maximum width and height
        """
        # Calculate width
        width_top = np.sqrt(
            ((src_points[1][0] - src_points[0][0]) ** 2)
            + ((src_points[1][1] - src_points[0][1]) ** 2)
        )
        width_bottom = np.sqrt(
            ((src_points[2][0] - src_points[3][0]) ** 2)
            + ((src_points[2][1] - src_points[3][1]) ** 2)
        )
        max_width = max(int(width_top), int(width_bottom))

        # Calculate height
        height_left = np.sqrt(
            ((src_points[3][0] - src_points[0][0]) ** 2)
            + ((src_points[3][1] - src_points[0][1]) ** 2)
        )
        height_right = np.sqrt(
            ((src_points[2][0] - src_points[1][0]) ** 2)
            + ((src_points[2][1] - src_points[1][1]) ** 2)
        )
        max_height = max(int(height_left), int(height_right))

        return max_width, max_height

    def manual_selection_four_points(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Manual point selection and dewarping
        """
        points = self.manual_point_selection(image)

        if len(points) != 4:
            raise ValueError("Need exactly 4 points for perspective correction")

        src_points = np.array(points, dtype=np.float32)
        return self.order_points(src_points)

    # TODO: update saved variables
    def save_calibration(self, filename: str, format: str = "numpy"):
        """
        Save the calibration matrix and parameters to file

        Args:
            filename: Path to save the calibration file
            format: 'numpy' (npz), 'pickle' (pkl), or 'json' (limited precision)
        """
        if self.transform_matrix is None:
            raise ValueError("No calibration data available. Perform dewarping first.")

        calibration_data = {
            "transform_matrix": self.transform_matrix,
            "src_points": self.src_points,
            "dst_points": self.dst_points,
            "output_size": self.output_size,
        }

        if format.lower() == "numpy":
            # Save as numpy compressed format (.npz)
            if not filename.endswith(".npz"):
                filename += ".npz"
            np.savez_compressed(filename, **calibration_data)

        elif format.lower() == "pickle":
            # Save as pickle format (.pkl)
            if not filename.endswith(".pkl"):
                filename += ".pkl"
            with open(filename, "wb") as f:
                pickle.dump(calibration_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif format.lower() == "json":
            # Save as JSON format (limited precision)
            if not filename.endswith(".json"):
                filename += ".json"
            # Convert numpy arrays to lists for JSON serialization
            json_data = {
                "transform_matrix": self.transform_matrix.tolist(),
                "src_points": self.src_points.tolist(),
                "dst_points": self.dst_points.tolist(),
                "output_size": self.output_size,
            }
            with open(filename, "w") as f:
                json.dump(json_data, f, indent=2)

        else:
            raise ValueError("Format must be 'numpy', 'pickle', or 'json'")

        print(f"Calibration saved to: {filename}")

    # TODO: update saved variables
    def load_calibration(self, filename: str, format: str = None):
        """
        Load calibration matrix and parameters from file

        Args:
            filename: Path to the calibration file
            format: 'numpy', 'pickle', or 'json'. If None, inferred from filename
        """
        # Infer format from filename if not specified
        if format is None:
            if filename.endswith(".npz"):
                format = "numpy"
            elif filename.endswith(".pkl"):
                format = "pickle"
            elif filename.endswith(".json"):
                format = "json"
            else:
                raise ValueError(
                    "Cannot infer format. Please specify format parameter."
                )

        if format.lower() == "numpy":
            data = np.load(filename)
            self.transform_matrix = data["transform_matrix"]
            self.src_points = data["src_points"]
            self.dst_points = data["dst_points"]
            self.output_size = tuple(data["output_size"])

        elif format.lower() == "pickle":
            with open(filename, "rb") as f:
                data = pickle.load(f)
            self.transform_matrix = data["transform_matrix"]
            self.src_points = data["src_points"]
            self.dst_points = data["dst_points"]
            self.output_size = data["output_size"]

        elif format.lower() == "json":
            with open(filename, "r") as f:
                data = json.load(f)
            self.transform_matrix = np.array(data["transform_matrix"], dtype=np.float32)
            self.src_points = np.array(data["src_points"], dtype=np.float32)
            self.dst_points = np.array(data["dst_points"], dtype=np.float32)
            self.output_size = tuple(data["output_size"])

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
        dewarped = cv2.warpPerspective(
            image, self.transform_matrix, image.shape[:2][::-1]
        )
        return dewarped

    def dewarp_single_camera(
        self,
        images: List[np.ndarray],
        target_image: np.ndarray,
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 1.0,
    ) -> np.ndarray:
        """
        Dewarp a single camera using checkerboard pattern

        Args:
            image: List of Calibration image
            target_image: Image of Calibration Plate in Target Plane
            pattern_size: (width, height) of checkerboard internal corners
            square_size: Size of checkerboard squares in world units

        Returns:
            output_image
        """
        # Prepare object points (3D points in real world space)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
            -1, 2
        )  # Assuming points at z=0 plane
        objp *= square_size

        # Arrays to store object points and image points
        # cv2.calibrateCamera need points in format of list not ndarray
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        target_objp = [] # 3D points
        target_imgp = [] # 2D points

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret is True:
                objpoints.append(objp)

                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
            else:
                print("=" * 40)
                print("Bad calibration result.")
                print("=" * 40)
                return None

        # Search matched points in raw target image (FIXME: redundent code)
        gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        # FIXME: Calibration plate is not chessboard
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret is True:
            target_objp = objp
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            target_imgp = corners2
        else:
            print("=" * 40)
            print("Bad calibration result.")
            print("=" * 40)
            return None

        # Camera calibration, need multiple images to estimate camera intrinsic matrix and distortion coefficients
        # stdev: This is the primary return value and represents the root mean square (RMS) reprojection error. Values between 0.1 and 1.0 are considered good.
        # camera_matrix: This is a 3x3 matrix that represents the intrinsic parameters of the camera.
        # dist_coeffs: This is a vector (k1, k2, p1, p2, k3) that represents the distortion coefficients of the camera.
        # rvecs: This is a vector that represents the rotation vector of the camera.  Each vector represents the rotation of the camera coordinate system relative to the world coordinate system for that specific image.
        # tvecs: This is a vector that represents the translation vector of the camera.  Each vector represents the translation of the camera coordinate system relative to the world coordinate system for that specific image.
        stdev, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = (
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        )
        # Note: https://stackoverflow.com/a/69161589/18736354
        # "focal length" (unit pixels) in the camera matrix: it describes a scale factor for mapping the real world to a picture of a certain resolution

        # Estimate extrinsic matrix from target image
        # rotation vector and translation vector of the camera coordinate system relative to the world coordinate system
        # roatation vector is independent with scale_factor, 
        # translation vector varies linearly with scale_factor
        ret, self.rvec, self.tvec = (
            cv2.solvePnP(target_objp, target_imgp, self.camera_matrix, self.dist_coeffs)
        )
        rotation_matrix, _ = cv2.Rodrigues(self.rvec)

        # Apply rotation matrix and translation vector to object points (in world coordinate), the results are in camera coordinate

        # FIXME: how to determine scale_factor from calibration
        # zoom target_objp in image plane for visulization, no more need for estimate camera parameters
        scale_factor = 1/2 * (self.camera_matrix[0, 0] + self.camera_matrix[1, 1]) / self.tvec[2]  # not good estimation [pixels per mm]
        # Compute the homography matrix
        self.transform_matrix, _ = cv2.findHomography(target_imgp[:, 0, :], target_objp[:, :2] * scale_factor)
        print("Camera scale factor: ", scale_factor[0])
        print("Camera homography matrix: \n", self.transform_matrix)

        # Temp solution 2
        # Take the four corners of the input image
        h, w = target_image.shape[:2]
        img_corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        # Transform them using the homography
        corners_transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), self.transform_matrix)
        corners_transformed = corners_transformed.reshape(-1, 2)
        # Find the bounding box
        min_x, min_y = np.min(corners_transformed, axis=0).astype(int)
        max_x, max_y = np.max(corners_transformed, axis=0).astype(int)
        width_out = max_x - min_x + 1
        height_out = max_y - min_y + 1

        # Apply the homography transformation (without correction for distortion)
        output_image = cv2.warpPerspective(
            target_image, self.transform_matrix, (width_out, height_out)
            # target_image, self.transform_matrix, target_image.shape[:2][::-1]
        )

        # DEUBG code
        cv2.imwrite("saved_cv_image.png", output_image)
        cv2.imshow("Dewarped Image", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print calibration results in pretty format
        np.set_printoptions(precision=3)
        print("=" * 60)
        print('-'*10 + " Calibration Results " + '-'*10)
        print(f"Calibration RMS reprojection error: {stdev:.3f} pixels")
        print("Camera intrinsic matrix: \n", self.camera_matrix)
        print("\t principal point: ", self.camera_matrix[:2, 2])
        print("Camera distorion coeffs [k1, k2, p1, p2, k3]: ", self.dist_coeffs[0])
        # According to Tsai's model, only radial distortion needs to be condsidered for industrial machine vision application[s].
        print(
            "\t radial [k1, k2, k3]: ", np.append(self.dist_coeffs[0][:2], self.dist_coeffs[0][-1])
        )
        print("\t tangential [p1, p2]: ", self.dist_coeffs[0][2:4])
        print("Rotation vector (x, y, z): ", self.rvec.T[0])
        print("Rotation Matrix:\n", rotation_matrix)
        print("Translation vector (x, y, z): ", self.tvec.T[0])
        print("=" * 60)
        return output_image

    def demo_tilted_image(self):
        # Create a sample image with text for demonstration
        sample_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(sample_img, (50, 50), (550, 350), (0, 0, 0), 2)
        cv2.putText(
            sample_img,
            "SAMPLE DOCUMENT",
            (100, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            sample_img,
            "This is a test image",
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            sample_img,
            "for dewarping demo",
            (100, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        # Apply a perspective transform to simulate tilt
        src_pts = np.float32([[50, 50], [550, 50], [550, 350], [50, 350]])
        dst_pts = np.float32([[30, 80], [520, 40], [570, 320], [80, 360]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        image = cv2.warpPerspective(sample_img, matrix, (600, 400))

        print("Starting manual selection...")
        dewarped_manual = self.dewarp_manual(image.copy())
        cv2.imshow("Manual Dewarped", dewarped_manual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_usage(image_path: str):
    """
    Demonstration of how to use the TiltCameraDewarper class
    """
    # Initialize dewarper
    dewarper = TiltCameraDewarper()

    calibration_images = [cv2.imread(f"./left_calibration_figs/{i:02d}.jpg") for i in range(1, 8)]
    # Load image (replace with your image path)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        print("Creating a sample tilted image for demonstration...")
        dewarper.demo_tilted_image()
        exit(0)

    dewarp_image = dewarper.dewarp_single_camera(
        calibration_images, image, pattern_size=(8, 6), square_size=25
    )

    cv2.imshow("Dewarped Image", dewarp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # if image_path is not None:
    #     save_filename = os.path.splitext(image_path)[0]
    #     dewarper.save_calibration(save_filename, format="numpy")


if __name__ == "__main__":
    # Read image path from cmd line
    if len(sys.argv) < 2:
        print("Usage: python dewarp.py <image_path>\nUsing default image.\n")
        image_path = "./figures/checkerboard.jpg"
    else:
        image_path = sys.argv[1]
    demo_usage(image_path)
