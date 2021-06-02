from setuptools import setup

setup(
    name='csdl_om',
    packages=[
        'csdl_om',
    ],
    install_requires=[
        'csdl<1',
        'openmdao',
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'guppy3',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)
