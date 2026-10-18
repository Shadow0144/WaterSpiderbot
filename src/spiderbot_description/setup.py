"""Set up Spiderbot description."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'spiderbot_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'),
         [f for f in glob('models/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'models', 'meshes'),
         [f for f in glob('models/meshes/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'models', 'textures'),
         [f for f in glob('models/textures/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shadow0144',
    maintainer_email='1351027+Shadow0144@users.noreply.github.com',
    description='Spiderbot description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'description_node = spiderbot_description.main:main',
        ],
    },
)
