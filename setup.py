from setuptools import setup

setup(
    name='csdl_om',
    packages=[
        'csdl_om',
    ],
    install_requires=[
        'csdl<1',
        # taking length of inputs.values() within components is broken
        # in later versions of openmdao
        'openmdao==3.10.0',
        'numpy>=1.21',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'guppy3',
        'scipy>=1.7.1',
    ],
)
