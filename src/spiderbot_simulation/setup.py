"""Set up Spiderbot simulation."""

from setuptools import find_packages, setup

package_name = 'spiderbot_simulation'

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
    maintainer='Shadow0144',
    maintainer_email='1351027+Shadow0144@users.noreply.github.com',
    description='Spiderbot simulation',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simulation_node = spiderbot_simulation.main:main',
        ],
    },
)
