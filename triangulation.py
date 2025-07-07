import cv2
import numpy as np

class Camera:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.projection_matrix = self.compute_projection_matrix()

    def compute_projection_matrix(self, rotation_vector=None, translation_vector=None):
        """ Compute the projection matrix from the camera matrix and distortion coefficients. """
        # For simplicity, we assume no rotation or translation for this example.
        # In a real scenario, you would include rotation and translation vectors.
        if rotation_vector is None or translation_vector is None:
            return np.hstack((self.camera_matrix, np.zeros((3, 1), dtype=np.float64)))
        else:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))
            return self.camera_matrix @ extrinsic_matrix

    def display_info(self):
        print(f"Camera Matrix: {self.camera_matrix}")
        print(f"Distortion Coefficients: {self.dist_coeffs}")
        print(f"Projection Matrix: {self.projection_matrix}")
        

class SPIVSelfCalibration:
    def __init__(self, left_camera, right_camera):
        """ Initialize the SPIV self-calibration with two cameras.
        left_camera: Camera object for the left camera.
        right_camera: Camera object for the right camera.
        """
        self.left_camera = left_camera
        self.right_camera = right_camera

    def ensemble_correlation(self, left_image, right_image):
        """
        Find correspondences between left and right images. 
        Return disparity map.
        """
        # Placeholder for correlation logic
        print("Ensembling correlation between left and right images...")
        # In a real scenario, you would implement a method to find correspondences
        # between the two sets of points, possibly using RANSAC or another method.
        pass
    
    def triangulate(self, points1, points2):
        """
        Triangulate 3D points from two sets of 2D points.
        points1: 2D points from the first camera (shape: 2xN)
        points2: 2D points from the second camera (shape: 2xN)
        Returns: 3D points in homogeneous coordinates (shape: 4xN)
        """
        print("Triangulating points...")
        # Placeholder for triangulation logic
        # Triangulate the 3D points
        # The output 'points4D' will be 4xN, representing homogeneous coordinates (x, y, z, w)
        points4D = cv2.triangulatePoints(self.left_camera.projection_matrix,
                                         self.right_camera.projection_matrix,
                                         points1, points2)

        # Convert homogeneous coordinates to 3D Cartesian coordinates
        # Divide by the 'w' component
        points3D = points4D[:3] / points4D[3]

        print("Reconstructed 3D points:")
        print(points3D)
        return points3D
    
    def fitted_plane(self, points3D):
        """
        Fit a plane to the 3D points.
        points3D: 3D points in Cartesian coordinates (shape: 3xN)
        Returns: Plane coefficients (A, B, C, D) for the equation Ax + By + Cz + D = 0
        """
        print("Fitting a plane to the 3D points...")
        # Use SVD to fit a plane to the points
        A = np.c_[points3D[0], points3D[1], points3D[2], np.ones(points3D.shape[1])]
        _, _, Vt = np.linalg.svd(A)
        plane_coeffs = Vt[-1]  # [A, B, C, D]

        print("Plane coefficients:", plane_coeffs)
        return plane_coeffs
    
    def plane_to_xy_transform(self, plane_coeffs):
        """
        Given plane coefficients (A, B, C, D), return a 4x4 transformation matrix
        that maps the plane to the XY-plane (z'=0).
        """
        A, B, C, D = plane_coeffs
        normal = np.array([A, B, C])
        norm = np.linalg.norm(normal) # L2 norm of the plane normal vector
        if norm == 0:
            raise ValueError("Invalid plane: zero normal vector")
        z_axis = normal / norm  # new Z direction base vector

        # Pick an x direction from left camera (#1)
        # Ref: https://math.stackexchange.com/a/2967611/1063060
        x_axis_left = self.left_camera.camera_matrix[:3, 0]  # x-axis of the left camera
        # Project x_axis_left onto the plane
        x_axis = x_axis_left - np.dot(x_axis_left, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        p0 = -D / (norm ** 2) * normal # Project the calibration plate origin onto the new fitted plane

        # Rotation matrix from world to local frame
        dR = np.vstack([x_axis, y_axis, z_axis])
        # Translation vector from world to local frame
        dT = -dR @ p0
        return dR, dT


    def calibrate(self):
        # Placeholder for calibration logic
        print("Calibrating stereo camera...")
        # In a real scenario, you would use cv2.stereoCalibrate here
        # to compute the essential matrix and rectify the images.
        pass
