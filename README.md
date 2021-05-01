# **CARTF**

## usage

##### Dataset
download data from  [baiduyun](https://pan.baidu.com/s/1NiN7SHEtK2yG1dEP7tKS-A ) (the password is 08mo) and save them to the ./data folder.



##### TensorD
download the [tensorD](https://github.com/Large-Scale-Tensor-Decomposition/tensorD.git), which is a very sophisticated tensor decomposition tool. And add it to our tool.

##### Test
Then you can run the model by:


```
python class-levelRecommendation.py

```

```
python method-levelRecommendation.py
```


The outputs will be will be saved  in the ./data/0%, ./data/20%, etc. path.
And you can change the k of top-k and the different stages of development (eg., 0%, 20%, etc.) to get the result you need.