from setuptools import find_packages, setup

package_name = 'assignment1_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/rviz', ['rviz/cylinders.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eva',
    maintainer_email='akulka96@asu.edu',
    description='Perception assignment for cylinder detection using ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cylinder_pipeline = assignment1_perception.cylinder_pipeline:main',
        ],
    },
)
