import os
from setuptools import find_packages, setup

package_name = 'cube_grasp_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 文件
        (os.path.join('share', package_name, 'launch'),
            [os.path.join('launch', 'grasp_bridge.launch.py')]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tptwo',
    maintainer_email='tptwo@todo.todo',
    description='YOLO-seg 到机械臂抓取的 ROS2 桥接层',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_pose_publisher = cube_grasp_bridge.yolo_pose_publisher:main',
            'grasp_controller = cube_grasp_bridge.grasp_controller_node:main',
        ],
    },
)
