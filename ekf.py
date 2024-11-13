import numpy as np

class EKF:
    def __init__(self):
        # State vector [px, py, vx, vy]
        self.x = np.zeros(4)
        
        # State covariance matrix P
        self.P = np.eye(4)
        
        # Process noise covariance matrix Q
        self.Q = np.eye(4)
        
        # State transition matrix F (initialized for constant velocity model)
        self.F = np.eye(4)
        
        # Measurement noise matrices for LiDAR and Radar
        self.R_lidar = np.array([[0.0225, 0], [0, 0.0225]])
        self.R_radar = np.array([[0.09, 0, 0], [0, 0.0009, 0], [0, 0, 0.09]])
        
        # Measurement matrix for LiDAR (H)
        self.H_lidar = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]])
    
    def predict(self, dt):
        """Predicts the state vector and covariance matrix."""
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        
        # Predict the next state
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update_lidar(self, z):
        """Updates the state using LiDAR measurements."""
        y = z - self.H_lidar.dot(self.x)
        S = self.H_lidar.dot(self.P).dot(self.H_lidar.T) + self.R_lidar
        K = self.P.dot(self.H_lidar.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4) - K.dot(self.H_lidar)).dot(self.P)
    
    def update_radar(self, z):
        """Updates the state using Radar measurements (non-linear)."""
        rho, phi, rho_dot = z
        
        # Convert state to polar coordinates
        px, py, vx, vy = self.x
        rho_pred = np.sqrt(px**2 + py**2)
        phi_pred = np.arctan2(py, px)
        rho_dot_pred = (px*vx + py*vy) / rho_pred
        
        # Predicted measurement in polar coordinates
        z_pred = np.array([rho_pred, phi_pred, rho_dot_pred])
        y = z - z_pred
        
        # Normalize the angle
        y[1] = self.normalize_angle(y[1])
        
        H_j = self.calculate_jacobian(self.x)
        S = H_j.dot(self.P).dot(H_j.T) + self.R_radar
        K = self.P.dot(H_j.T).dot(np.linalg.inv(S))
        
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4) - K.dot(H_j)).dot(self.P)

    def calculate_jacobian(self, x_state):
        """Calculates the Jacobian matrix for Radar measurement update."""
        px, py, vx, vy = x_state
        rho2 = px**2 + py**2
        rho = np.sqrt(rho2)
        rho3 = rho2 * rho
        
        if np.abs(rho2) < 1e-6:
            return np.zeros((3, 4))
        
        H_j = np.array([[px / rho, py / rho, 0, 0],
                        [-py / rho2, px / rho2, 0, 0],
                        [py * (vx * py - vy * px) / rho3, px * (vy * px - vx * py) / rho3, px / rho, py / rho]])
        return H_j

    def normalize_angle(self, angle):
        """Normalizes an angle to be within -pi and pi."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
