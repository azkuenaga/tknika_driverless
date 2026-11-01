from setuptools import find_packages, setup

package_name = 'tknika_driverless'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mirena',
    maintainer_email='mirena@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "sensor_monitor = tknika_driverless.sensor_monitor:main",
            "lidar_visualizer = tknika_driverless.lidar_visualizer:main",
            "cone_detector = tknika_driverless.cone_detector:main",
            "camera_detector = tknika_driverless.camera_detector:main",
            'sensor_fusion = tknika_driverless.sensor_fusion:main',
            'pid_controller = tknika_driverless.pid_controller:main',
        ],
    },
)
