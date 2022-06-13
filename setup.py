from setuptools import setup

setup(
    name='csdl_om',
    packages=[
        'csdl_om',
    ],
    install_requires=[
        'csdl<1',
        'openmdao',
        'numpy>=1.21',
        'pint',
        'guppy3',
        'scipy>=1.7.1',
    ],
)

