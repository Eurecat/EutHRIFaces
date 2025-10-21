from setuptools import find_packages, setup

package_name = 'face_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/face_detection.launch.py',
        ]),
        # ('share/' + package_name + '/config', ['config/face_detection.yaml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy',
        'PyYAML',
        'onnxruntime',
    ],
    zip_safe=True,
    maintainer='Josep Bravo',
    maintainer_email='josep.bravo@eurecat.org',
    description='Face detection package for HRI applications using YOLO',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'face_detector = face_detection.face_detector:main',
        ],
    },
)
