# CSDL-OM

This is a compiler back end for the Computational System Design Language
(CSDL) that uses OpenMDAO to generate executable objects from CSDL model
code.

Please read the documentation for
[CSDL](https://github.com/LSDOlab/csdl) to learn how to use CSDL.

To use this backend in CSDL code, install this package.
Then, in your project file, include

```py
from csdl_om import Simulator
```

to use the `Simulator` class.

## Installation

First install the latest version of
[CSDL](https://github.com/LSDOlab/csdl).
Then clone this repository and install using `pip`.
