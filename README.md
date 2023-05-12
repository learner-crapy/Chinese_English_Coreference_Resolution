## 文件介紹

bert_output_visualization.py： 可視化BERT輸出的向量及對應的mention，中英文需要指定不同的預訓練模型

machine_main.py: 實驗中我們採用了幾種機器學習模型，包括SVN, decision tree， k-means，這些方法不在這個文件裡進行訓練和測試

main.py: 所有的深度學習模型都在這個文件裡訓練和測試

jsonlines2json4.py： 將jsonlines格式的文件轉換成json格式，並重構數據集，4的版本主要是限制了文本長度，時候這為300,也就是只有文本長度小於300的樣本才能被選擇進行重構，其他jsonlines2json3.py則不對此做出限制，這兩個版本都是為了使得數據平衡而設置的

ModelForImg/:這個文件夾下的文件是用MNIST數據集對幾個模型進行驗證的，於NLP本身沒什麼關係

log_dir: 記錄訓練日誌

checkpoints：保存模型

data：存放數據集

pre_model: 存放預訓練模型

rebuild_confusion_matrix.py： 有時圖片參數設置的不太對，導致圖片佈局奇怪，根據原圖片的數據重新設置參數，並重新生成混淆矩陣

textclassfication_bert_textcnn.py： 使用textcnn進行情感分類的例子，僅供在編寫TextCNN時進行參考，非被項目構成部分

## 部署和運行
1、安裝依賴包
```bash
pip install -r requirements.txt
```

2、下載預訓練模型，並放到pre_model文件夾下

3、運行main.py文件，訓練模型，並在訓練過程中進行測試，訓練日誌會保存在log_dir文件夾下
查看訓練日誌，運行
```bash
tensorboard --logdir=log_dir
```
運行指令示例
```bash
python main.py --ensemble_model_list
CRModel
CRModel_2dense
--en_cn
cn
--exp_name
5-5-chinese-300-deep-25-epoch-for-rebuild-confusion_matrix
--model_type
2d
--lr
2e-5
--other_lr
2e-5
--bert_dir
./pre_model/chinese/chinese_roberta_wwm_large_ext_pytorch/
--data_dir
./data/chinese/
--train_batch_size
16
--eval_batch_size
16
--train_epochs
25
--max_seq_len
300
--dropout_prob
0.1
```

4、運行機器學習模型進行分類，運行machine_main.py文件，沒有採用參數更新的機器學習模型，調用python庫進行分類，機制為隨機梯度下降，只支持一次生成參數並利用該參數進行預測
```bash
python machine_main.py 
```
