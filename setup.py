from setuptools import setup, find_packages

setup(
    name='robotdatapy',
    version='1.0.0',    
    description='Python package for interfacing with robot data',
    url='',
    author='Mason Peterson',
    author_email='masonbp@mit.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'rosbags',
                        'pykitti'
                      ],

)
