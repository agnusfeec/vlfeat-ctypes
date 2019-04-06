#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 08:29:06 2019

@author: agnus
"""

from __future__ import print_function

from collections import namedtuple

from ctypes import (c_int, c_float, c_double, c_void_p, c_bool,
                    POINTER, CFUNCTYPE, cast, byref)

import numpy as np

import numpy.ctypeslib as npc

from .vl_ctypes import (LIB, Enum, vl_type, vl_size, np_to_c_types, c_to_vl_types)

from numpy.ctypeslib import ndpointer

class VlFVNormalization(Enum):
    VL_FISHER_FLAG_SQUARE_ROOT = (0x1 << 0)
    VL_FISHER_FLAG_NORMALIZED = (0x1 << 1)
    VL_FISHER_FLAG_IMPROVED = (VL_FISHER_FLAG_NORMALIZED|VL_FISHER_FLAG_SQUARE_ROOT)
    VL_FISHER_FLAG_FAST = (0x1 << 2)


VL_FISHER_FLAG_SQUARE_ROOT = (0x1 << 0)
VL_FISHER_FLAG_NORMALIZED = (0x1 << 1)
VL_FISHER_FLAG_IMPROVED = (VL_FISHER_FLAG_NORMALIZED|VL_FISHER_FLAG_SQUARE_ROOT)
VL_FISHER_FLAG_FAST = (0x1 << 2)

def ptr(data):
    num_data = len(data)
    array_type = c_float * num_data
    data_p = array_type(*data)
    return data_p

def vl_fisher_encode(enc, dataType, means, dimension, numClusters, covariances, priors, data, numData, flags): 
    
    if(dataType == vl_type.FLOAT.value):
        p_np = ndpointer(dtype=np.float32)
        p_vec = POINTER((c_float *(2*dimension*numClusters)))
        #aux = (c_float*(2*dimension*numClusters))()
    else:
        p_np = ndpointer(dtype=np.float64)
        p_vec = POINTER((c_double *(2*dimension*numClusters)))
        #aux = (c_double*(2*dimension*numClusters))()
    
    Py_vl_fisher_encode = LIB['vl_fisher_encode']
    Py_vl_fisher_encode.argtypes = [  p_vec, vl_type, p_np, vl_size, vl_size, p_np, p_np, p_np, vl_size, c_int ]
    Py_vl_fisher_encode.restype = vl_size
    numTerms = Py_vl_fisher_encode(byref(enc), dataType, means, dimension, numClusters, covariances, priors, data, numData, flags)
    
    #if(dataType == vl_type.FLOAT.value):
    #    enc = np.asarray(enc,dtype=np.float32)
    #else:
    #    enc = np.asarray(enc,dtype=np.float64)
    #enc = [aux[i] for i in range(2*dimension*numClusters)]
    
    return numTerms