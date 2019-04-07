#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:32:23 2019

@author: agnus
"""

from __future__ import print_function

from collections import namedtuple

from ctypes import (c_int, c_float, c_double, c_void_p, c_bool,
                    POINTER, CFUNCTYPE, cast)

import numpy as np

import numpy.ctypeslib as npc

from .vl_ctypes import (LIB, CustomStructure, Enum,
                        vl_type, vl_size, np_to_c_types, c_to_vl_types)

from .kmeans import VlKMeans_p

class VlGMMInitialization(Enum):
    VLGMMKMEANS = 0
    VLGMMRAND = 1
    VLGMMCUSTOM = 2

class VlGMM(CustomStructure):
    _fields_ = [
            ("dataType", vl_type),
            ("dimension", vl_size),
            ("numClusters", vl_size),
            ("numData", vl_size),
            ("maxNumIterations", vl_size),
            ("numRepetitions", vl_size),

            ("verbosity", c_int),

            ("means", c_void_p),
            ("covariances", c_void_p),
            ("priors", c_void_p),
            ("posteriors", c_void_p),
            ("sigmaLowBound", c_void_p),

            ("initialization", c_int),

            ("kmeansInit", VlKMeans_p),

            ("LL", c_double),
            ("kmeansInitIsOwner", c_bool)
    ]

# Initializations modes
VlGMMKMeans = 0 #Initialize GMM from KMeans clustering.
VlGMMRand 	= 1 #Initialize GMM parameters by selecting points at random.
VlGMMCustom = 2 #User specifies the initial GMM parameters.

# Data types
VL_TYPE_DOUBLE =  2
VL_TYPE_FLOAT = 1
VL_TYPE_INT16 = 5
VL_TYPE_INT32 = 7
VL_TYPE_INT64 = 9
VL_TYPE_INT8 = 3
VL_TYPE_UINT16 = 6
VL_TYPE_UINT32 = 8
VL_TYPE_UINT64 = 10
VL_TYPE_UINT8 = 4

VlGMM_p = POINTER(VlGMM)
c_double_p = POINTER(c_double)
c_float_p = POINTER(c_float)


################################################################################
### functions in the SO

# create and destroy
vl_gmm_new = LIB['vl_gmm_new']
vl_gmm_new.argtypes = [ vl_type, vl_size, vl_size ]
vl_gmm_new.restype = VlGMM_p

vl_gmm_new_copy = LIB['vl_gmm_new_copy']
vl_gmm_new_copy.argtypes = [ VlGMM_p ]
vl_gmm_new_copy.restype = VlGMM_p

vl_gmm_delete = LIB['vl_gmm_delete']
vl_gmm_delete.argtypes = [ VlGMM_p ]
vl_gmm_delete.restype = None

vl_gmm_reset = LIB['vl_gmm_reset']
vl_gmm_reset.argtypes = [ VlGMM_p ]
vl_gmm_reset.restype = None

# Basic data processing

def vl_gmm_cluster(gmm, data, numData):

    p_np = np.ctypeslib.ndpointer(dtype=np.float32)

    Py_vl_gmm_cluster = LIB['vl_gmm_cluster']
    Py_vl_gmm_cluster.argtypes = [ VlGMM_p, p_np, vl_size ]
    Py_vl_gmm_cluster.restype = c_double

    return Py_vl_gmm_cluster(gmm, data, numData)

# Fine grained data processing

vl_gmm_init_with_rand_data = LIB['vl_gmm_init_with_rand_data']
vl_gmm_init_with_rand_data.argtypes = [ VlGMM_p, c_void_p, vl_size ]
vl_gmm_init_with_rand_data.restype = None

vl_gmm_init_with_kmeans = LIB['vl_gmm_init_with_kmeans']
vl_gmm_init_with_kmeans.argtypes = [ VlGMM_p, c_void_p, vl_size, VlKMeans_p ]
vl_gmm_init_with_kmeans.restype = None

def vl_gmm_em(gmm, data, numData):
    """
    	Args:
        	gmm	     GMM object instance. (VlGMM *)
          	data	 data points which should be clustered. (void const *)
          	numData	 number of data points.(vl_size)
		Return:
			(double)
    """

    Py_vl_gmm_em = LIB['vl_gmm_em']
    Py_vl_gmm_em.argtypes = [ VlGMM_p, c_void_p, vl_size ]
    Py_vl_gmm_em.restype = c_double

    num_data = len(data)
    array_type = c_float * num_data
    data_p = array_type(*data)

    return Py_vl_gmm_em(gmm, data_p, numData)

vl_gmm_set_means = LIB['vl_gmm_set_means']
vl_gmm_set_means.argtypes = [ VlGMM_p, c_void_p ]
vl_gmm_set_means.restype = None

vl_gmm_set_covariances = LIB['vl_gmm_set_means']
vl_gmm_set_covariances.argtypes = [ VlGMM_p, c_void_p ]
vl_gmm_set_covariances.restype = None

vl_gmm_set_priors = LIB['vl_gmm_set_priors']
vl_gmm_set_priors.argtypes = [ VlGMM_p, c_void_p ]
vl_gmm_set_priors.restype = None

vl_get_gmm_data_posteriors_f = LIB['vl_get_gmm_data_posteriors_f']
vl_get_gmm_data_posteriors_f.argtypes = [ c_float_p, vl_size, vl_size,
                                        c_float_p, c_float_p, vl_size,
                                        c_float_p, c_float_p ]
vl_get_gmm_data_posteriors_f.restype = c_double

vl_get_gmm_data_posteriors_d = LIB['vl_get_gmm_data_posteriors_d']
vl_get_gmm_data_posteriors_d.argtypes = [ c_double_p, vl_size, vl_size,
                                        c_double_p, c_double_p, vl_size,
                                        c_double_p, c_double_p ]
vl_get_gmm_data_posteriors_d.restype = c_double

# Set parameters
vl_gmm_set_num_repetitions = LIB['vl_gmm_set_num_repetitions']
vl_gmm_set_num_repetitions.argtypes = [ VlGMM_p, vl_size ]
vl_gmm_set_num_repetitions.restype = None

vl_gmm_set_max_num_iterations = LIB['vl_gmm_set_max_num_iterations']
vl_gmm_set_max_num_iterations.argtypes = [ VlGMM_p, vl_size ]
vl_gmm_set_max_num_iterations.restype = None

vl_gmm_set_verbosity = LIB['vl_gmm_set_verbosity']
vl_gmm_set_verbosity.argtypes = [ VlGMM_p, c_int ]
vl_gmm_set_verbosity.restype = None

vl_gmm_set_initialization = LIB['vl_gmm_set_initialization']
vl_gmm_set_initialization.argtypes = [ VlGMM_p, VlGMMInitialization ]
vl_gmm_set_initialization.restype = None

vl_gmm_set_kmeans_init_object = LIB['vl_gmm_set_kmeans_init_object']
vl_gmm_set_kmeans_init_object.argtypes = [ VlGMM_p, VlKMeans_p ]
vl_gmm_set_kmeans_init_object.restype = None

vl_gmm_set_covariance_lower_bound = LIB['vl_gmm_set_covariance_lower_bound']
vl_gmm_set_covariance_lower_bound.argtypes = [ VlGMM_p, c_double ]
vl_gmm_set_covariance_lower_bound.restype = None

vl_gmm_set_covariance_lower_bounds = LIB['vl_gmm_set_covariance_lower_bounds']
vl_gmm_set_covariance_lower_bounds.argtypes = [ VlGMM_p, c_double_p ]
vl_gmm_set_covariance_lower_bounds.restype = None

def p2List(p, num):
    my_list = [p[i] for i in range(num)]
    return my_list

# Get Parameters
def vl_gmm_get_means(gmm):

    Py_vl_gmm_get_means = LIB['vl_gmm_get_means']
    Py_vl_gmm_get_means.argtypes = [ VlGMM_p ]

    size = gmm.contents.dimension * gmm.contents.numClusters

    if(gmm.contents.dataType.value == vl_type.FLOAT.value):
        restype =  c_float_p
        np_type = np.float32
    else:
        restype =  c_double_p
        np_type = np.float64

    Py_vl_gmm_get_means.restype = restype
    means = Py_vl_gmm_get_means(gmm)

    return np.array(np.fromiter(means, dtype=np_type, count=size))

def vl_gmm_get_covariances(gmm):

    Py_vl_gmm_get_covariances = LIB['vl_gmm_get_covariances']
    Py_vl_gmm_get_covariances.argtypes = [ VlGMM_p ]

    size = gmm.contents.dimension * gmm.contents.numClusters

    if(gmm.contents.dataType.value == vl_type.FLOAT.value):
        restype =  c_float_p
        np_type = np.float32
    else:
        restype =  c_double_p
        np_type = np.float64

    Py_vl_gmm_get_covariances.restype = restype
    covariances = Py_vl_gmm_get_covariances(gmm)

    return np.array(np.fromiter(covariances, dtype=np_type, count=size))

def vl_gmm_get_priors(gmm):
    Py_vl_gmm_get_priors = LIB['vl_gmm_get_priors']
    Py_vl_gmm_get_priors.argtypes = [ VlGMM_p ]

    size = gmm.contents.numClusters

    if(gmm.contents.dataType.value == vl_type.FLOAT.value):
        restype =  c_float_p
        np_type = np.float32
    else:
        restype =  c_double_p
        np_type = np.float64

    Py_vl_gmm_get_priors.restype = restype
    priors = Py_vl_gmm_get_priors(gmm)

    return np.array(np.fromiter(priors, dtype=np_type, count=size))

def vl_gmm_get_posteriors(gmm):

    Py_vl_gmm_get_posteriors = LIB['vl_gmm_get_posteriors']
    Py_vl_gmm_get_posteriors.argtypes = [ VlGMM_p ]

    size = gmm.contents.numData * gmm.contents.numClusters

    if(gmm.contents.dataType.value == vl_type.FLOAT.value):
        restype =  c_float_p
        np_type = np.float32
    else:
        restype =  c_double_p
        np_type = np.float64

    Py_vl_gmm_get_posteriors.restype = restype
    posteriors = Py_vl_gmm_get_posteriors(gmm)

    return np.array(np.fromiter(posteriors, dtype=np_type, count=size))


vl_gmm_get_data_type = LIB['vl_gmm_get_data_type']
vl_gmm_get_data_type.argtypes = [ VlGMM_p ]
vl_gmm_get_data_type.restype = vl_type

vl_gmm_get_dimension = LIB['vl_gmm_get_dimension']
vl_gmm_get_dimension.argtypes = [ VlGMM_p ]
vl_gmm_get_dimension.restype = vl_size

vl_gmm_get_num_repetitions = LIB['vl_gmm_get_num_repetitions']
vl_gmm_get_num_repetitions.argtypes = [ VlGMM_p ]
vl_gmm_get_num_repetitions.restype = vl_size

vl_gmm_get_num_data = LIB['vl_gmm_get_num_data']
vl_gmm_get_num_data.argtypes = [ VlGMM_p ]
vl_gmm_get_num_data.restype = vl_size

vl_gmm_get_num_clusters = LIB['vl_gmm_get_num_clusters']
vl_gmm_get_num_clusters.argtypes = [ VlGMM_p ]
vl_gmm_get_num_clusters.restype = vl_size

vl_gmm_get_loglikelihood = LIB['vl_gmm_get_loglikelihood']
vl_gmm_get_loglikelihood.argtypes = [ VlGMM_p ]
vl_gmm_get_loglikelihood.restype = c_double

vl_gmm_get_verbosity = LIB['vl_gmm_get_verbosity']
vl_gmm_get_verbosity.argtypes = [ VlGMM_p ]
vl_gmm_get_verbosity.restype = c_int

vl_gmm_get_max_num_iterations = LIB['vl_gmm_get_max_num_iterations']
vl_gmm_get_max_num_iterations.argtypes = [ VlGMM_p ]
vl_gmm_get_max_num_iterations.restype = vl_size

vl_gmm_get_num_repetitions = LIB['vl_gmm_get_num_repetitions']
vl_gmm_get_num_repetitions.argtypes = [ VlGMM_p ]
vl_gmm_get_num_repetitions.restype = vl_size

vl_gmm_get_initialization = LIB['vl_gmm_get_initialization']
vl_gmm_get_initialization.argtypes = [ VlGMM_p ]
vl_gmm_get_initialization.restype = VlGMMInitialization

vl_gmm_get_kmeans_init_object = LIB['vl_gmm_get_kmeans_init_object']
vl_gmm_get_kmeans_init_object.argtypes = [ VlGMM_p ]
vl_gmm_get_kmeans_init_object.restype = VlKMeans_p

def vl_gmm_get_covariance_lower_bounds(gmm):
    """
        Args:
            gmm	object
        Returns:
            lower bound on covariances. (numpy array of doubles)

    """
    Py_vl_gmm_get_covariance_lower_bounds = LIB['vl_gmm_get_covariance_lower_bounds']
    Py_vl_gmm_get_covariance_lower_bounds.argtypes = [ VlGMM_p ]

    size = gmm.contents.dimension
    restype =  c_double_p
    np_type = np.float64

    Py_vl_gmm_get_covariance_lower_bounds.restype = restype
    covs_lb = Py_vl_gmm_get_covariance_lower_bounds(gmm)

    return np.array(np.fromiter(covs_lb, dtype=np_type, count=size))
