import pywt
import re
from preprocess import CRBertFeature, CRProcessor, get_data
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, RandomSampler
import dataset
import config

class SOMEUTILS:

    def __init__(self, args, language):
        self.args = args
        if language == 'en':
            self.pattern = r'\w+|[^\w\s]'
        else:
            self.pattern = r'([\u4e00-\u9fa5]|[^\u4e00-\u9fa5\s])'
        self.train_json_data = None
        self.test_json_data = None
        self.dev_json_data = None
        self.json_data = None

    def get_my_data(self, processor, data_file, mode):
        raw_examples = processor.read_json(data_file)
        if mode == 'train':
            self.train_json_data = processor.get_examples(raw_examples, mode)
            self.json_data = self.train_json_data
        elif mode == 'test':
            self.test_json_data = processor.get_examples(raw_examples, mode)
            self.json_data = self.test_json_data
        else:
            self.dev_json_data = processor.get_examples(raw_examples, mode)
            self.json_data = self.dev_json_data
        dev_features = get_data(processor, data_file, mode, self.args)
        my_dataset = dataset.CRDataset(dev_features)
        return my_dataset

    def supervised_kmeans_clustering(self, vectors, labels, dev_vectors, dev_labels, num_clusters, batch_size, epochs):
        # Perform k-means clustering with 5 clusters and the given labels
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)

        for epoch in range(epochs):
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                kmeans.partial_fit(batch_vectors, batch_labels)

            # Output validation results after each epoch
            dev_predictions = kmeans.predict(dev_vectors)
            accuracy = sum(dev_predictions == dev_labels) / len(dev_labels)
            print(f"Epoch {epoch + 1} validation accuracy: {accuracy}")


    # 写一个函数，将pytorch中dataset提取bert的inputs中，返回经过bert之后的结果和每个token中的标签信息
    def multi_get_bert_encode(self, my_dataset):
        # 加载预训练模型和tokenizer
        tokenizer = BertTokenizer.from_pretrained('./model/')
        model = BertModel.from_pretrained('./model')
        my_data_dict = {'data':{'for_bert':[], 'for_kmeans':[]}, 'label':[]}

        for i in range(len(my_dataset.features)):
            for_bert = {'input_ids':None, 'tiken_type_ids':None, 'attention_mask':None, 'pst_lst':None}
            for_bert['input_ids'] = my_dataset.features[i].token_ids
            for_bert['tiken_type_ids'] = my_dataset.features[i].token_type_ids
            for_bert['attention_mask'] = my_dataset.features[i].attention_masks
            for_bert['pst_lst'] = [[self.json_data[i].span1[1], self.json_data[i].span1[1]+len(re.findall(self.pattern, self.json_data[i].span1[0]))], [self.json_data[i].span2[1], self.json_data[i].span2[1]+len(re.findall(self.pattern, self.json_data[i].span2[0]))]]
            my_data_dict['data']['for_bert'].append(for_bert)


        # 对句子进行tokenize和padding
        # inputs = tokenizer(dataset, return_tensors='pt', padding=True)
        # 使用BERT模型进行编码
        with torch.no_grad():
            for i, data in enumerate(my_data_dict['data']['for_bert']):
                outputs = model(input_ids=torch.tensor(data['input_ids']).unsqueeze(0), token_type_ids=torch.tensor(data['tiken_type_ids']).unsqueeze(0), attention_mask=torch.tensor(data['attention_mask']).unsqueeze(0))
                # my_data_dict['data']['for_kmeans'].append(outputs.last_hidden_state)
                vector = self.get_mention_vector(outputs.last_hidden_state[0], data['pst_lst'])
                concatenated_tensor = torch.cat((vector[0], vector[1]), dim=0)
                my_data_dict['data']['for_kmeans'].append(concatenated_tensor)
                my_data_dict['label'].append(self.json_data[i].label)
        # 返回BERT模型的输出结果
        return my_data_dict


    def bert_encode(self, sentence):
        # 加载预训练模型和tokenizer
        tokenizer = BertTokenizer.from_pretrained('./model/')
        model = BertModel.from_pretrained('./model')

        # 对句子进行tokenize和padding
        inputs = tokenizer(sentence, return_tensors='pt', padding=True)

        # 使用BERT模型进行编码
        with torch.no_grad():
            outputs = model(**inputs)

        # 返回BERT模型的输出结果
        return outputs.last_hidden_state


    def fourier_analysis(self, vector):
        """
        This function takes a vector as input and performs Fourier analysis on it.
        It then displays the resulting Fourier transform.
        """
        # Compute the Fourier transform of the input vector
        fourier_transform = np.fft.fft(vector)

        # Compute the power spectrum of the Fourier transform
        power_spectrum = np.abs(fourier_transform) ** 2

        # Create a frequency axis for the power spectrum
        frequency_axis = np.fft.fftfreq(len(vector))

        # Plot the power spectrum
        plt.plot(frequency_axis, power_spectrum)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.show()

    # fourier_analysis(bert_encode(sentence_en))

    def fourier_transform(self, bert_output, fft_result_path=None, high_filtered_path=None, low_filtered_path=None):
        # 进行傅里叶变换
        # bert_output = torch.squeeze(bert_output)
        fft_result = np.fft.fft(bert_output)
        # 计算频率
        freq = np.fft.fftfreq(len(bert_output))
        # 绘制频谱图
        plt.plot(freq, np.abs(fft_result))
        plt.title('fft_result')
        if fft_result_path is not None:
            plt.savefig(fft_result_path)
        # plt.show()

        # 高通滤波
        fft_result_high = np.fft.fftshift(fft_result)
        fft_result_high[:len(fft_result_high) // 2 - 10] = 0
        fft_result_high[len(fft_result_high) // 2 + 10:] = 0
        high_filtered = np.fft.ifft(np.fft.ifftshift(fft_result_high))
        plt.plot(freq, np.abs(high_filtered))
        plt.title('high_filtered')
        if high_filtered_path is not None:
            plt.savefig(high_filtered_path)
        # plt.show()

        # 低通滤波
        fft_result_low = np.fft.fftshift(fft_result)
        fft_result_low[:len(fft_result_low) // 2 + 10] = 0
        fft_result_low[len(fft_result_low) // 2 - 10:] = 0
        low_filtered = np.fft.ifft(np.fft.ifftshift(fft_result_low))
        plt.plot(freq, np.abs(low_filtered))
        plt.title('low_filtered')
        if low_filtered_path is not None:
            plt.savefig(low_filtered_path)
        # plt.show()

        return freq, high_filtered, low_filtered

    # 使用mathplotlib的subplots函数绘制多个子图，传入(子图行数，子图列数，标题列表， 矩阵列表， save_path=None)的参数，绘图并显示
    def plot_subplots(self, rows, cols, titles, matrix, save_path=None):
        fig, axs = plt.subplots(rows, cols)
        for ax in axs.flat:
            ax.tick_params(axis='x', labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
        if rows == 1 and cols == 1:
            axs.set_title(titles[0])
            axs.plot(matrix[0])
        elif cols == 1 and rows > 1:
            for i in range(rows):
                if i >= len(titles):
                    break
                axs[i].set_title(titles[i])
                axs[i].plot(matrix[i])
        elif rows == 1 and cols > 1:
            for j in range(cols):
                if j >= len(titles):
                    break
                axs[j].set_title(titles[j])
                axs[j].plot(matrix[j])
        else:
            for i in range(rows):
                for j in range(cols):
                    if i * cols + j >= len(titles):
                        break
                    axs[i, j].set_title(titles[i * cols + j])
                    axs[i, j].plot(matrix[i * cols + j])
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()



    # 写一个函数，对bert输出的结果进行小波变换处理，并显示结果
    def wavelet_transform(self, bert_output):
        bert_output = torch.squeeze(bert_output)
        # 将bert输出结果转换为numpy数组
        bert_output_np = bert_output.detach().numpy()
        # 对bert输出结果进行小波变换
        coeffs = pywt.dwt2(bert_output_np, 'haar')
        # 显示小波变换结果
        plt.imshow(coeffs[0], cmap='gray')
        plt.show()

    def analyze_bert_output_similarity(self, bert_output):
        """
        分析bert输出结果哪些行之间的关系比较紧密，通过各种相似度方法来比较，分别返回这些相似度矩阵和这些相似矩阵加权平均的结果

        :param bert_output: bert输出结果，numpy数组类型
        :return: 相似度矩阵列表和相似度矩阵加权平均的结果
        """
        bert_output = torch.squeeze(bert_output).numpy()
        # 计算余弦相似度矩阵
        cos_sim_matrix = np.dot(bert_output, bert_output.T) / (
                np.linalg.norm(bert_output, axis=1) * np.linalg.norm(bert_output.T, axis=0))

        # 计算欧几里得距离相似度矩阵
        euclidean_sim_matrix = 1 / (
                1 + np.sqrt(np.sum((bert_output[:, np.newaxis, :] - bert_output[np.newaxis, :, :]) ** 2, axis=-1)))

        # 计算曼哈顿距离相似度矩阵
        manhattan_sim_matrix = 1 / (
                1 + np.sum(np.abs(bert_output[:, np.newaxis, :] - bert_output[np.newaxis, :, :]), axis=-1))

        # 计算相似度矩阵加权平均的结果
        similarity_matrices = [cos_sim_matrix, euclidean_sim_matrix, manhattan_sim_matrix]
        weights = [0.4, 0.3, 0.3]
        weighted_average = np.average(similarity_matrices, axis=0, weights=weights)

        return cos_sim_matrix, euclidean_sim_matrix, manhattan_sim_matrix, weighted_average

    def plot_similarity_matrix_list(self, similarity_matrix_list):
        """
        传入一个二维的相似度矩阵列表，根据不同位置的值大小绘制出不同深浅的颜色

        :param similarity_matrix_list: 二维的相似度矩阵列表
        """
        names = ['cos_sim_matrix', 'euclidean_sim_matrix', 'manhattan_sim_matrix', 'weighted_average']
        for name, similarity_matrix in zip(names, similarity_matrix_list):
            # 绘制热力图
            plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
            plt.title(name)
            plt.savefig('./img/' + name + ".jpg")
            plt.show()

    def bert_encode(self, sentence, model_path='./model/', tokenizer_path='./model/'):
        # 加载预训练模型和tokenizer
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        model = BertModel.from_pretrained(model_path)

        # 对句子进行tokenize和padding
        inputs = tokenizer(sentence, return_tensors='pt', padding=True)

        print(inputs['input_ids'].shape)
        # 使用BERT模型进行编码
        with torch.no_grad():
            outputs = model(**inputs)

        # 返回BERT模型的输出结果
        print(outputs.last_hidden_state.shape)
        return outputs.last_hidden_state

    def is_start_of_sublist(self, num, lst):
        for lst_i in lst:  # iterate through each tuple in the list
            if lst_i[0] == num:  # check if the first element of the tuple is equal to the given number
                return True  # if so, return True
        return False  # if not found in any tuple, return False

    # 遍历列表中的元素和对应的下标
    def get_index_elements(self, lst, ele):
        result = []
        for index, element in enumerate(lst):
            # 如果元素等于给定的k
            if element == ele:
                # 将下标添加到结果列表中
                result.append(index)
        # 返回结果列表
        return result

    def find_positions(self, list1, list2):
        '''
        param list1: 一个句子的token列表，被分割后的
        param list2: 一个句子的mention列表
        '''
        positions = []
        for list2_str in list2:
            lst2_str_split = re.findall(r'\w+|[^\w\s]', list2_str)
            # lst2_str_split = list2_str.split(' ')
            index_list = self.get_index_elements(list1, lst2_str_split[0])
            for i in range(len(index_list)):
                start = index_list[i]
                if self.is_start_of_sublist(start, positions):
                    continue
                else:
                    if list1[start:start + len(lst2_str_split)] == lst2_str_split:
                        # print(list1[start:start + len(lst2_str_split)])
                        # print(lst2_str_split)
                        # print('-----------------------------')
                        end = start+ len(lst2_str_split)
                        positions.append([start, end])
        return positions

    # 获取每个mention在bert输出结果中对应的向量
    def get_mention_vector(self, bert_outputs, positions):
        mention_vector = []
        for position in positions:
            mention_vector.append(bert_outputs[position[0]:position[1]].mean(dim=0))  # 在行上取平均值，这样就得到一个向量，而不是多个向量
        return mention_vector


if __name__ == '__main__':
    args = config.Args().get_parser()
    some_utils = SOMEUTILS(args, 'en')
    #
    # # my_data_train = some_utils.get_my_data(processor=CRProcessor(), data_file='../data/train.json', mode='train')
    # # my_data_dict_train = some_utils.multi_get_bert_encode(my_data_train)
    #
    # my_data_dev = some_utils.get_my_data(processor=CRProcessor(), data_file='../data/dev.json', mode='dev')
    # my_data_dict_dev = some_utils.multi_get_bert_encode(my_data_dev)
    #
    # for_kmeans_data_train = torch.stack([torch.tensor(x).clone() for x in my_data_dict_train['data']['for_kmeans']], dim=0)
    # # for_kmeans_data = torch.cat(my_data_dict['data']['for_kmeans'], dim=1)
    # for_kmeans_label_train = torch.tensor(my_data_dict_train['label'])
    # for_kmeans_data_dev = torch.stack([torch.tensor(x).clone() for x in my_data_dict_dev['data']['for_kmeans']], dim=0)
    # # for_kmeans_data = torch.cat(my_data_dict['data']['for_kmeans'], dim=1)
    # for_kmeans_label_dev = torch.tensor(my_data_dict_dev['label'])
    #
    # some_utils.supervised_kmeans_clustering(vectors=for_kmeans_data_train, labels=for_kmeans_label_train, dev_vectors=for_kmeans_data_dev, dev_labels=for_kmeans_label_dev, num_clusters=2, batch_size=32, epochs=10)
    # # result = some_utils.supervised_kmeans_clustering(for_kmeans_data_train, for_kmeans_label, 2)
    # # print(result)
    # # initialize the
    # # print(my_data_dict)
    sentence_en = '''
    It is said that she is so generous and clever at the table that successful men who hit on her are still giggling when they are rejected, and that she is so drunk that she can keep drinking and making those clients fall under the table, and those clients who are so drunk like to be drunk again by Li Qing, they will tell our president on the phone when they book the next dinner party: " Don't forget to bring Li Qing.
    '''
    sentence_en = sentence_en.replace('\n', '')
    bert_output = some_utils.bert_encode(sentence_en)
    #
    sentence_en_split = [''] + re.findall(r'\w+|[^\w\s]', sentence_en) + ['']  # 占开头CLS和结束SEP的位置
    # sentence_en_split = ['']+sentence_en.split(' ')+['']
    sentence_en_split_2 = ['she', 'the table', 'that successful men', 'her', 'they', 'she', 'she', 'those clients', 'the table', 'those clients', 'they', 'they', 'the next dinner party', 'Li Qing', 'Li Qing']
    print(len(sentence_en_split_2))

    print(len(some_utils.find_positions(sentence_en_split, sentence_en_split_2)))


    my_index_str = ''.join(str(elem) for elem in some_utils.find_positions(sentence_en_split, sentence_en_split_2))
    dic_key = [sentence_en_split_2[i] + my_index_str[i] for i in range(len(sentence_en_split_2))]

    vectors = some_utils.get_mention_vector(bert_output[0], some_utils.find_positions(sentence_en_split, sentence_en_split_2))
    dict_result = dict(zip(dic_key, some_utils.get_mention_vector(bert_output[0], some_utils.find_positions(sentence_en_split, sentence_en_split_2))))
    for vector in dict_result.values():
        some_utils.fourier_transform(vector)
    matrix = np.zeros((6, 24, 24))

    some_utils.plot_subplots(4, 4, sentence_en_split_2, vectors)
    some_utils.plot_subplots(10, 10, sentence_en_split, some_utils.bert_encode(sentence_en)[0])
    #
    #
    #
    #
    # # plot_similarity_matrix_list(list(analyze_bert_output_similarity(bert_encode(sentence_en))))
    # # print(analyze_bert_output_similarity(bert_encode(sentence_en)))
    # # wavelet_transform(bert_encode(sentence_en))
    # # fourier_transform(bert_encode(sentence_en))
    # print(len(sentence_en_split))

