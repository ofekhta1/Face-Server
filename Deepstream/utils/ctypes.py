import numpy as np
import ctypes
import pyds

def convert_to_array(buffer,shape,type=ctypes.c_float)->np.ndarray:
    pointer=ctypes.cast(pyds.get_ptr(buffer), ctypes.POINTER(type))
    embs = np.ctypeslib.as_array(pointer, shape=shape)
    return embs;
def convert_to_num(buffer,type=ctypes.c_float):
    pointer=ctypes.cast(pyds.get_ptr(buffer), ctypes.POINTER(type))
    value = pointer.contents.value
    return value