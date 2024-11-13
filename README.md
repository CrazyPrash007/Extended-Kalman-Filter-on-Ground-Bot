# Ground Bot Navigation using Extended Kalman Filter (EKF)

## Overview
This project implements an Extended Kalman Filter (EKF) to help a Ground Bot estimate its position and velocity based on data from radar and lidar sensors. The EKF processes noisy sensor measurements and updates the bot's position estimates in real-time.

## File Structure
- `ekf.py`: Contains the EKF implementation with predict and update steps.
- `main.py`: Main script for reading input data, processing it using EKF, and saving the results to an output file.
- `input.txt`: Contains the sensor data (lidar and radar) and ground truth values.
- `output.txt`: Generated output file with estimated positions and velocities.
- `requirements.txt`: List of dependencies for Python (e.g., NumPy).

## Requirements
- Python 3.x
- NumPy

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```
**To run the project, simply execute the following command:**
```bash
python main.py
```
**Input Format**

- Laser measurement (L): L meas_px meas_py timestamp gt_px gt_py gt_vx gt_vy
- Radar measurement (R): R meas_rho meas_phi meas_rho_dot timestamp gt_px gt_py gt_vx gt_vy

**Output Format**

- After processing the sensor data, the output file (output.txt) will contain:

est_px est_py est_vx est_vy meas_px meas_py gt_px gt_py gt_vx gt_vy

**Assumptions**

    -The Ground Bot follows a constant velocity model.
    -Radar measurements are provided in polar coordinates (rho, phi, rho_dot), while LiDAR measurements are provided in Cartesian coordinates (px, py).
    -The filter can handle asynchronous sensor inputs, with Radar and LiDAR measurements arriving at different intervals.

**Design Decisionss**

    -The prediction step uses the constant velocity model for state transition.
    -Update steps differ for Radar (non-linear, using a Jacobian matrix) and LiDAR (linear).
    -The EKF is designed to handle both sensor modalities seamlessly, deciding which update function to use based on the input sensor type.