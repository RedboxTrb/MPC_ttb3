# MPC TurtleBot3 Navigator

Replaces Pure Pursuit with MPC-based trajectory tracking and cubic spline path smoothing.

## Dependencies

pip3 install casadi

## Nodes

- path_smoother.py: Loads CSV waypoints, fits cubic spline, publishes smoothed path
- mpc_tracker.py: MPC path tracker with obstacle avoidance using CasADi/IPOPT solver

## Features

- Nonlinear MPC with CasADi/IPOPT (12-step horizon, 30Hz)
- Cubic spline path smoothing
- Exponential obstacle soft cost for smooth avoidance
- Handles static and dynamic obstacles via 2D lidar

## Usage

export TURTLEBOT3_MODEL=burger
ros2 launch nav sim_bringup.py
