from setuptools import setup, find_packages

setup(
    name='csdl_om',
    packages=find_packages(),
    #packages=['csdl_om'],
    install_requires=[
        'csdl<1',
        'openmdao',
        'numpy>=1.21',
        'pint',
        'guppy3',
        'scipy>=1.8.0',
    ],
)

