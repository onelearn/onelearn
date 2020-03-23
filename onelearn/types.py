# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause
import os
from numba import types


class String(str):
    """A weird callable string type. This just allows to keep intact the numba class
    declarations intact."""

    def __new__(cls, *args, **kw):
        return str.__new__(cls, *args, **kw)

    def __call__(self, *args, **kwargs):
        pass


if os.getenv("NUMBA_DISABLE_JIT", None) == "1":
    boolean = String("bool")
    uint8 = String("uint8")
    uint32 = String("uint32")
    float32 = String("float32")
    int32 = String("int32")
    void = String("void")
else:
    boolean = types.boolean
    uint8 = types.uint8
    uint32 = types.uint32
    int32 = types.int32
    float32 = types.float32
    void = types.void

string = types.string
Tuple = types.Tuple


def get_array_2d_type(dtype=None):
    if os.getenv("NUMBA_DISABLE_JIT", None) == "1":
        return String("")
    else:
        return dtype[:, ::1]
