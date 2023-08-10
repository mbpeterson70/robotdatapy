from setuptools import setup

setup(
    name='robot_data',
    py_modules=['robot_data'],
    version='0.1.0',    
    description='Robot Data Utilities Package',
    url='',
    author='Mason Peterson',
    author_email='masonbp@mit.edu',
    license='MIT',
    install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'rosbags'
                      ],

    # classifiers=[
    #     'Development Status :: 1 - Planning',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: BSD License',  
    #     'Operating System :: POSIX :: Linux',        
    #     'Programming Language :: Python :: 2',
    #     'Programming Language :: Python :: 2.7',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.4',
    #     'Programming Language :: Python :: 3.5',
    # ],
)