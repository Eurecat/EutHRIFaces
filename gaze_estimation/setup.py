from setuptools import find_packages, setup

package_name = 'gaze_estimation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/gaze_estimation.yaml']),
        ('share/' + package_name + '/launch', ['launch/gaze_estimation.launch.py']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy',
        'PyYAML',
    ],
    zip_safe=True,
    maintainer='Josep Bravo',
    maintainer_email='josep.bravo@eurecat.org',
    description='Gaze estimation package for HRI applications',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'gaze_estimation_node = gaze_estimation.gaze_estimation_node:main',
        ],
    },
)
