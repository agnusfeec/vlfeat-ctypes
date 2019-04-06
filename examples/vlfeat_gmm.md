
# Gaussian Mixture Models (GMM)


```python
from vlfeat.gmm import *
from random import random
import numpy as np
```


```python
# generate some data for running
numData = 1000
dimension = 3

data = [random() for x in range(numData*dimension)]
array_data = np.asarray(data,dtype=np.float32)
```


```python
print type(array_data), array_data.shape
print array_data
```

    <type 'numpy.ndarray'> (3000,)
    [0.9849168  0.54030895 0.65507126 ... 0.12165529 0.5624702  0.00500118]



```python
# create a new instance of a GMM object for float data
numClusters = 10
gmm = vl_gmm_new (VL_TYPE_FLOAT, dimension, numClusters) ;
```


```python
# set the maximum number of EM iterations to 100
vl_gmm_set_max_num_iterations (gmm, 100) ;
# set the initialization to random selection
vl_gmm_set_initialization (gmm,VlGMMRand);
```


```python
# cluster the data, i.e. learn the GMM
vl_gmm_cluster (gmm, array_data, numData);
```


```python
print "num_data                 = ", vl_gmm_get_num_data(gmm)
print "dimension                = ", vl_gmm_get_dimension(gmm)
print "num_clusters             = ", vl_gmm_get_num_clusters(gmm)
print "verbosity                = ", vl_gmm_get_verbosity(gmm)
print "max_number_of_iterations = ", vl_gmm_get_max_num_iterations(gmm)
```

    num_data                 =  1000
    dimension                =  3
    num_clusters             =  10
    verbosity                =  0
    max_number_of_iterations =  100



```python
# get the means, covariances, and priors of the GMM
means = vl_gmm_get_means(gmm);
covariances = vl_gmm_get_covariances(gmm);
priors = vl_gmm_get_priors(gmm);
```


```python
print type(means), means.shape
print means
```

    <type 'numpy.ndarray'> (30,)
    [0.59462    0.47205934 0.10689756 0.4447806  0.37388813 0.47974092
     0.78566086 0.48852962 0.891163   0.8006425  0.14578111 0.5456001
     0.1572077  0.46888632 0.17956966 0.8868934  0.5445453  0.43605942
     0.55303186 0.90410006 0.8977413  0.30663288 0.43697834 0.8857432
     0.06496482 0.46936834 0.6526975  0.43784773 0.85828644 0.46145633]



```python
print type(covariances), covariances.shape
print covariances
```

    <type 'numpy.ndarray'> (30,)
    [0.04363025 0.06872921 0.00366006 0.03857541 0.05444678 0.03853368
     0.01827892 0.04793352 0.00516289 0.01584681 0.00625306 0.0629904
     0.00738199 0.07383958 0.01035189 0.00474495 0.04713492 0.05620383
     0.07131839 0.00338993 0.00469328 0.01826449 0.05774087 0.00527173
     0.00166457 0.08558749 0.02379693 0.05407924 0.00641314 0.0392633 ]



```python
print type(priors), priors.shape
print priors
```

    <type 'numpy.ndarray'> (10,)
    [0.09733869 0.25367123 0.06070291 0.06040585 0.0876085  0.12913285
     0.03963008 0.0780322  0.08314638 0.11033134]



```python
#get loglikelihood of the estimated GMM
loglikelihood = vl_gmm_get_loglikelihood(gmm)
print loglikelihood
```

    -196.650281592



```python
# get the soft assignments of the data points to each cluster
posteriors = vl_gmm_get_posteriors(gmm)
```


```python
print type(posteriors), posteriors.shape
print posteriors
```

    <type 'numpy.ndarray'> (10000,)
    [8.17337887e-19 3.71038467e-02 4.70439671e-03 ... 1.55579881e-32
     2.99011910e-04 1.11990754e-04]



```python

```
