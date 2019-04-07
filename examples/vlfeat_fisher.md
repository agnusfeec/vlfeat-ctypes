
# Fisher Vector encoding (FV)


```python
from vlfeat.gmm import *
from vlfeat.fisher import *
from random import random
import numpy as np
```


```python
# generate some data for training
numData = 1000
dimension = 3

data = np.random.rand(numData*dimension).astype(np.float32)
```


```python
# create a GMM object and cluster input data to get means, covariances
# and priors of the estimated mixture
numClusters = 10
gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numClusters) ;
```


```python
# cluster the data, i.e. learn the GMM
vl_gmm_cluster (gmm, data, numData);
```

to deal with the by ref passing, at this moment was used the ctype float array


```python
# allocate space for the encoding    
enc = (c_float *(2*dimension*numClusters))()
```


```python
# generate some data for encoding
numDataToEncode = 50

dataToEncode = np.random.rand(numDataToEncode*dimension).astype(np.float32)
print type(dataToEncode), type(dataToEncode[0]), dataToEncode.shape
```

    <type 'numpy.ndarray'> <type 'numpy.float32'> (150,)



```python
# run fisher encoding
num_ops = vl_fisher_encode(enc, VL_TYPE_FLOAT, vl_gmm_get_means(gmm), dimension, 
                 numClusters, vl_gmm_get_covariances(gmm), 
                 vl_gmm_get_priors(gmm), dataToEncode, numDataToEncode, 
                 VL_FISHER_FLAG_IMPROVED)
```


```python
print type(enc)
```

    <class '__main__.c_float_Array_60'>



```python
fv = np.asarray(enc,dtype=np.float32)
print type(fv), fv.shape
print fv
```

    <type 'numpy.ndarray'> (60,)
    [-0.04287256 -0.06507428 -0.10020301 -0.18214983  0.1346008   0.17186874
     -0.13652709 -0.15981176 -0.19029722  0.19615485  0.07016775  0.1551748
     -0.1453222   0.1388649  -0.02093714 -0.07870242 -0.05408803  0.07281471
      0.03494111  0.10511477  0.17566758  0.20364644  0.13085678  0.2145811
     -0.09440219  0.17349534 -0.17443787  0.18611111 -0.22595568 -0.15456152
      0.07751302 -0.10568154  0.07260053  0.10071547 -0.1104122  -0.16936974
     -0.059111    0.02460173 -0.06640872 -0.05691591 -0.10911946 -0.13553801
      0.16017193  0.04012848 -0.05773102  0.08928745  0.0981173   0.08585645
     -0.08392892 -0.04438896  0.0790972  -0.06793284  0.02541483  0.28036758
      0.15831137 -0.170934   -0.07495537 -0.11649156 -0.09496763  0.14295271]



```python

```
