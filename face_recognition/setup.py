from setuptools import find_packages, setup

package_name = 'face_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/face_recognition.launch.py',
        ]),
        ('share/' + package_name + '/config', ['config/face_recognition_params.yaml']),
        ('share/' + package_name + '/weights', []),  # Placeholder for weights directory
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy',
        'PyYAML',
        'torch',
        'torchvision',
        'Pillow',
        'scikit-learn',
        'scipy',
        'facenet-pytorch',
    ],
    zip_safe=True,
    maintainer='Josep Bravo',
    maintainer_email='josep.bravo@eurecat.org',
    description='Face recognition package for HRI applications using face embeddings and identity management',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_recognition_node = face_recognition.face_recognition_node:main',
        ],
    },
)
