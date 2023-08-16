from setuptools import setup, find_packages

setup(
    name='robot_utils',
    version='0.1.0',    
    description='Robot Utilities Package',
    url='',
    author='Mason Peterson',
    author_email='masonbp@mit.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'rosbags',
                      ],

)