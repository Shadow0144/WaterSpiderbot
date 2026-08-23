"""Set up Spiderbot locomotion."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'spiderbot_locomotion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'model_weights'),
         glob(os.path.join('model_weights', '*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shadow0144',
    maintainer_email='1351027+Shadow0144@users.noreply.github.com',
    description='Spiderbot locomotion',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simple_sin_locomotion_node = '
            'spiderbot_locomotion.simple_sin.main:main',
            'handcrafted_angles_locomotion_node = '
            'spiderbot_locomotion.handcrafted_angles.main:main',
            'handcrafted_points_locomotion_node = '
            'spiderbot_locomotion.handcrafted_points.main:main',
            'deep_actor_critic_locomotion_node = '
            'spiderbot_locomotion.deep_actor_critic.main:main',
        ],
    },
)
