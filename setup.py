from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='robotdatapy',
    version='1.1.12',
    description='Python package for interfacing with robot data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mbpeterson70/robotdatapy',
    author='Mason Peterson',
    author_email='masonbp@mit.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'rosbags',
                        'rosbags-image',
                        'pykitti',
                        'evo',
                        'opencv-python',
                        'utm',
                        'gtsam',
                        'argcomplete',
                      ],
    entry_points={
        'console_scripts': [
            'rdp-plot-trajectory=robotdatapy.cli.plot_trajectory:main',
            'rdp-path-length=robotdatapy.cli.path_length:main',
        ],
    },

)
