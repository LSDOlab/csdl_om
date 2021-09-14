from setuptools import setup

setup(
    name='csdl_om',
    packages=[
        'csdl_om',
    ],
    install_requires=[
        'csdl<1',
        'openmdao==3.10.0',
        'numpy>=1.20,<1.21',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'guppy3',
    ],
)
