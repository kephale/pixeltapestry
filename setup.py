from setuptools import setup, find_packages

setup(
    name='pixeltapestery',
    version='0.0.1',
    description='Pixel Tapestery models',
    author='Kyle Harrington',
    author_email='kyle@kyleharrington.com',
    url='https://github.com/kephale/pixeltapestery',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'monai'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.9',
)
