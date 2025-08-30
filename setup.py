from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='robotdatapy',
    version='1.1.1',    
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
                      ],

)