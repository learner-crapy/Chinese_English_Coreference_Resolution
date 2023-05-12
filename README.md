## 文件介紹

bert_output_visualization.py： 可視化BERT輸出的向量及對應的mention，中英文需要指定不同的預訓練模型

machine_main.py: 實驗中我們採用了幾種機器學習模型，包括SVN, decision tree， k-means，這些方法不在這個文件裡進行訓練和測試

main.py: 所有的深度學習模型都在這個文件裡訓練和測試

jsonlines2json4.py： 將jsonlines格式的文件轉換成json格式，並重構數據集，4的版本主要是限制了尋去的文本長度，時候這為300，其他jsonlines2json3.py則不對此做出限制，這兩個版本都是為了使得數據平衡而設置的

ModelForImg/:這個文件夾下的文件是用MNIST數據集對幾個模型進行驗證的，於NLP本身沒什麼關係

log_dir: 記錄訓練日誌

checkpoints：保存模型

data：存放數據集

pre_model: 存放預訓練模型