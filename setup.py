from distutils.core import setup

setup(
    name='csdl_om',
    version='0.1',
    packages=[
        'csdl_om',
    ],
    install_requires=[
        # 'csdl@git+https://github.com/lsdolab/csdl@master',
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
