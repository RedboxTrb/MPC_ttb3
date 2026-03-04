# MPC TurtleBot3 Navigator

Replaces Pure Pursuit with MPC tracker and cubic spline path smoothing.

## Nodes
- path_smoother.py: Loads CSV waypoints, fits cubic spline, publishes smoothed path
- mpc_tracker.py: MPC-based path tracker with reactive obstacle avoidance

## Launch
ros2 launch nav sim_bringup.py

## Features
- MPC optimization using scipy SLSQP
- Cubic spline smoothing via scipy splprep/splev  
- Reactive obstacle avoidance: TRACKING → AVOIDING → RECOVERING state machine
- 2D lidar-based obstacle detection with global frame direction locking
