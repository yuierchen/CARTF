# **CARTF**

## usage

##### Dataset
download data from  [baiduyun](https://pan.baidu.com/s/1X3mVyt5oFMRHAWT1HAhtwg) (the password is s7rf) and save them to the ./data folder
the password is 08mo

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
And you can change the k of top-k and the different stages of developemnt (eg., 0%, 20%, etc.) to get the result you need.


##### Data Process
If you want to repeat the complete data preprocessing part, you need download the jar package, the Java project and original test dataset from [baiduyun](https://pan.baidu.com/s/1X3mVyt5oFMRHAWT1HAhtwg) (the password is s7rf).

After configuring the file paths, you need to run the following files in order:

processbikerdata.py

gettitles.py

idf.py

trainW2V.py

get_processed_records.py
