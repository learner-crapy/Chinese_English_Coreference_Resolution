# 项目介绍
具体项目代码和项目文件介绍以及部署请查看master分支，注意本项目基于 https://github.com/taishan1994/pytorch_bert_coreference_resolution 进行部署，做了以下几点工作：

1、更新了数据集，基于OntoNote5.0数据集重构了数据集，训练数据量可以达到1.7GB左右；

2、加入了机器学习方法SVM、决策树、k-means，深度学习模型VGG16、Inception、LeNet5、LSTM、TextCNN进行分类；

3、重建了原项目中的数据处理方法，分别从BERT的最后一层和每一层抽取数据合并输入到上述模型中，根据处理方法不同，上述的几个深度学习模型也有有一维卷积和二维卷积的区别，细节请查看代码和原理图；

4、加入了BERT输出向量可视化的代码；

5、加入了训练过程的日志记录，包括train_acc、train_loss、dev_loss、dev_acc、recall、precision、recall等


# 预训练模型
## 1、中文模型
&emsp;&emsp;在中文中我们将使用到下图所示的模型

<div align="center">
  <img src="./img/chinese_pretrained_model.jpg" />
</div>

<p align="center">中文选用预训练模型</p>

&emsp;&emsp;使用transformers下载上述模型,使用BertTokenizer以及BertModel加载，而不是RobertaTokenizer/RobertaModel，您可以在[这里](https://github.com/ymcui/Chinese-BERT-wwm#%E4%B8%AD%E6%96%87%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)找到这些信息。

```python
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```

## 2、英文模型

&emsp;&emsp;英文模型使用从huggingface下载的预训练模型，你可以通过下面的python代码获取

```python
# Import generic wrappers
from transformers import AutoModel, AutoTokenizer
# Define the model repo
model_name = "SpanBERT/spanbert-base-cased"
# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Transform input tokens
inputs = tokenizer("Hello world!", return_tensors="pt")
# Model apply
outputs = model(**inputs)
```
# 对BERT的输出进行处理
<div align="center">
  <img src="./img/mention_feature.jpg" />
</div>

<p align="center">从BERT的最后一层获取向量</p>


<div align="center">
  <img src="./img/mention_feature_2d.jpg" />
</div>

<p align="center">从BERT的每一层获取向量</p>

# 一维训练日志

<div align="center">
  <img src="./img/chinese_1d_5_5_indicator_chinese.jpg" />
</div>

<p align="center">从最后一层获取输入到一维卷积的网络中中文</p>

<div align="center">
  <img src="./img/chinese_1d_5_5_indicator_english.jpg" />
</div>

<p align="center">从最后一层获取输入到一维卷积的网络中英文</p>

# 二维训练日志

<div align="center">
  <img src="./img/chinese_2d_5_5_indicator_chinese.jpg" />
</div>

<p align="center">从每一层获取输入到一维卷积的网络中中文</p>

<div align="center">
  <img src="./img/chinese_2d_5_5_indicator_english.jpg" />
</div>

<p align="center">从每一层获取输入到一维卷积的网络中英文</p>

# 一维混淆矩阵

<div align="center">
  <img src="./img/contrifusion_matrix_5_5_1d_small.jpg" />
</div>

<p align="center">从每一层获取输入到一维卷积的网络中混淆矩阵中文</p>


# 二维混淆矩阵


<div align="center">
  <img src="./img/contrifusion_matrix_5_5_2d_small.jpg" />
</div>

<p align="center">从每一层获取输入到一维卷积的网络中混淆矩阵英文</p>

# 一层和两层全连接网络日志

<div align="center">
  <img src="./img/CRModel_CRModel_2dense_chinese_5_5.jpg" />
</div>

<p align="center">从最后一层获取输入到全连接网络中文</p>

<div align="center">
  <img src="./img/CRModel_CRModel_2dense_english_5_5.jpg" />
</div>

<p align="center">从最后一层获取输入到全连接网络英文</p>

# 一层和两层全连接网络混淆矩阵

<div align="center">
  <img src="./img/CRModel_CRModel_2dense_small.jpg" />
</div>

<p align="center">从最后一层获取输入到全连接网络混淆矩阵</p>
