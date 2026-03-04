import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    turtlebot3_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'empty_world.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            [os.path.join(get_package_share_directory('nav'), 'config', 'sim.rviz')]
        ],
        parameters=[{'use_sim_time': True}]
    )

    path_smoother = Node(
        package='nav',
        executable='path_smoother.py',
        name='path_smoother',
        parameters=[{
            'use_sim_time': True,
            'path_file': os.path.join(
                get_package_share_directory('nav'), 'waypoints', 'waypoint.csv'
            ),
            'frame_id': 'odom',
            'path_resolution': 0.05,
            'spline_smoothing': 1.0,
        }]
    )

    mpc_tracker = Node(
        package='nav',
        executable='mpc_tracker.py',
        name='mpc_tracker',
        parameters=[{
            'use_sim_time': True,
            'desired_linear_vel': 0.22,
            'max_linear_vel': 0.4,
            'max_angular_vel': 0.5,
            'horizon': 12,
            'dt': 0.1,
            'goal_tolerance': 0.2,
            'obstacle_threshold': 1.0,
            'avoidance_gain': 2.0,
            'control_hz': 30.0,
        }]
    )

    return LaunchDescription([
        turtlebot3_gazebo,
        path_smoother,
        mpc_tracker,
        rviz,
    ])
