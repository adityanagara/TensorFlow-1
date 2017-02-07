# TensorFlow
Learning TensorFlow, Installation and Examples

## Installing TensorFlow in Windows with Anaconda 
This tutorial is based on the TensorFlow Anaconda install tutorial [here](https://www.tensorflow.org/get_started/os_setup), but needed some changes to make it work.

### 1. Install [Anaconda](https://www.continuum.io/downloads) 
### 2. Create a conda enviroment 

Create a conda environment called tensorflow:
```
# Python 2.7
$ conda create -n tensorflow python=2.7

# Python 3.4
$ conda create -n tensorflow python=3.4

# Python 3.5
$ conda create -n tensorflow python=3.5
```

### 3. Activate the conda environment and install TensorFlow in it.
#### Activation
```
$ activate tensorflow
(tensorflow)$  # Your prompt should change
```

#### Install TensorFlow
Google has recently launched a newer version of Tesnsorflow r0.12 which include support of Windows both CPU and GPU version can now be installed using Python >=3.5.2 (only 64-bit) version.

For CPU only version open command prompt and enter follow command:
```
(tensorflow)$ pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.0rc0-cp35-cp35m-win_amd64.whl
```
(Source: http://stackoverflow.com/questions/37130489/installing-tensorflow-with-anaconda-in-windows)

### 4. Test
With the conda environment activated, you can now test your installation.

``` 
$ python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```

#### Deactivate
```
(tensorflow)$ deactivate
$  # Your prompt should change back
```
