from setuptools import find_packages, setup

package_name = 'visual_speech_activity'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/visual_speech_activity.launch.py',
        ]),
        ('share/' + package_name + '/config', ['config/visual_speech_activity_params.yaml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy',
        'PyYAML',
        'scipy',
    ],
    zip_safe=True,
    maintainer='Josep Bravo',
    maintainer_email='josep.bravo@eurecat.org',
    description='Visual speech activity detection for HRI applications using lip movement analysis',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'visual_speech_activity_node = visual_speech_activity.visual_speech_activity_node:main',
        ],
    },
)
