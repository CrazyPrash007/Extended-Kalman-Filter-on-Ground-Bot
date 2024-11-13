import numpy as np
from ekf import EKF

def read_input_file(input_file):
    measurements = []
    with open(input_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            sensor_type = data[0]
            timestamp = int(data[4] if sensor_type == 'R' else data[3])
            
            if sensor_type == 'L':
                meas_px = float(data[1])
                meas_py = float(data[2])
                gt_px = float(data[4])
                gt_py = float(data[5])
                gt_vx = float(data[6])
                gt_vy = float(data[7])
                measurements.append(('L', np.array([meas_px, meas_py]), timestamp, np.array([gt_px, gt_py, gt_vx, gt_vy])))
            
            elif sensor_type == 'R':
                meas_rho = float(data[1])
                meas_phi = float(data[2])
                meas_rho_dot = float(data[3])
                gt_px = float(data[5])
                gt_py = float(data[6])
                gt_vx = float(data[7])
                gt_vy = float(data[8])
                measurements.append(('R', np.array([meas_rho, meas_phi, meas_rho_dot]), timestamp, np.array([gt_px, gt_py, gt_vx, gt_vy])))
    
    return measurements

def save_output_file(output_file, results):
    with open(output_file, 'w') as f:
        for result in results:
            f.write(" ".join(map(str, result)) + "\n")

def process_measurements(measurements):
    ekf = EKF()
    results = []
    previous_timestamp = None

    for measurement in measurements:
        sensor_type, sensor_data, timestamp, ground_truth = measurement
        
        if previous_timestamp is None:
            previous_timestamp = timestamp
            continue

        dt = (timestamp - previous_timestamp) / 1e6
        previous_timestamp = timestamp

        ekf.predict(dt)

        if sensor_type == 'L':
            ekf.update_lidar(sensor_data)
        elif sensor_type == 'R':
            ekf.update_radar(sensor_data)

        # Store estimated state, measurement, and ground truth
        est_px, est_py, est_vx, est_vy = ekf.x
        meas_px, meas_py = (sensor_data[0], sensor_data[1]) if sensor_type == 'L' else (None, None)
        gt_px, gt_py, gt_vx, gt_vy = ground_truth

        result = [est_px, est_py, est_vx, est_vy, meas_px, meas_py, gt_px, gt_py, gt_vx, gt_vy]
        results.append(result)

    return results

def main():
    input_file = 'input.txt'
    output_file = 'output.txt'

    measurements = read_input_file(input_file)
    results = process_measurements(measurements)
    save_output_file(output_file, results)

if __name__ == '__main__':
    main()
