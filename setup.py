from setuptools import setup, find_packages

setup(
    name='cs229',
    version='0.0.1',
    description='CS229 Project',
    url='https://github.com/sgherbst/cs229-project',
    author='Steven Herbst',
    author_email='sherbst@stanford.edu',
    packages=['cs229'],
    install_requires=[
        'scipy',
        'numpy',
        'tqdm'
    ],
    include_package_data=True,
    zip_safe=False,
)
