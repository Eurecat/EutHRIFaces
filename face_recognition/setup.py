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
    description='Face recognition package for HRI applications',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # TODO: Add entry points when implemented
        ],
    },
)
