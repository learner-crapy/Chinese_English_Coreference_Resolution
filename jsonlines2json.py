import json
import os
import sys

from ProcessData import *


def replace_values_with_none(d):
    """
    This function takes a dictionary as input and returns a new dictionary with the same structure, 
    but with all values replaced with None.
    """
    return {k: None if not isinstance(v, dict) else replace_values_with_none(v) for k, v in d.items()}


def read_json_line(file_path, k):
    """
    This function takes a file path and an integer k as input and returns the kth line of the json file as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == k:
                return json.loads(line)


# 将一个字符串的1维列表合成一个字符串
def sum_list_to_str(list_1d):
    result = ''
    for i in range(0, len(list_1d)):
        result += list_1d[i]
    return result


# 将一个字符串的1维列表合成一个字符串,并在每个字符串后面加上一个空格
def sum_list_to_str_with_space(list_1d):
    result = ''
    for i in range(0, len(list_1d)):
        result += list_1d[i] + ' '
    return result


# 判断两个一维列表是否在同一个二维列表中，入参为一个三维列表和两个一维列表
def is_in_list(list_3d, list_1d_1, list_1d_2):
    for i in range(0, len(list_3d)):
        if list_1d_1 in list_3d[i] and list_1d_2 in list_3d[i]:
            return True
    return False


# 将一个字典数据添加到json文件中，入参为json_path，dict
def add_dict_to_json(json_path, dict):
    with open(json_path, 'a', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False)
        f.write('\n')


# 写一个函数判断文件是否超过xGB，入参为文件路径和x的值，返回值为True或False
def is_file_over_xGB(file_path, x):
    file_size = os.path.getsize(file_path)
    if file_size > x * 1024 * 1024 * 1024:
        return True
    else:
        return False

# 写一个函数将指定文件列表按照列表的顺序合并到一个文件中，入参为文件列表和目标文件
# 例如：merge(['a.json', 'b.json', 'c.json'], 'd.json')
def merge(file_list, target_file):
    with open(target_file, 'w', encoding='utf-8') as f:
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f1:
                for line in f1:
                    f.write(line)
                f.write('\n')
            f1.close()
    f.close()

# 获取指定目录下后缀为xxxd的文件列表，入参为目录路径和后缀名
def get_file_list(path, suffix):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                file_list.append(os.path.join(root, file))
    return file_list



def main(language, jsonlines_path, example_dict, json_path, x):
    print('processing file: ----------------', jsonlines_path, '------------------')
    # jsonlines文本的行数
    file_index = 0
    # json_path = json_path.split('.')[0] + str(file_index) + '.json'
    idx = 0
    line_nums = count_lines_in_jsonlines_file(jsonlines_path)
    for i in range(0, line_nums):
        print('processing file: ', jsonlines_path, '---------------- line', i, '------------------')
        new_line = read_jsonlines_line(jsonlines_path, i)
        senteces_i = new_line['sentences']
        # sentece_list是将一行jsonlines中senteces所有元素放在一个list中的结果，也就是长度为1维的list
        sentece_list = sum_list(senteces_i)
        # index_list是将一行jsonlines中clusters所有元素放在一个list中的结果，也就是长度为2维的list
        index_list = sum_list(new_line['clusters'])
        # 假设len(index_list)=n，每一个jsonlines文本将会产生n*(n+1)/2个新的json数据样本
        for j in range(0, len(index_list)):
            for jj in range(j, len(index_list)):
                if language == 'chinease':
                    example_dict['target']['span1_text'] = sum_list_to_str(sentece_list[index_list[j][0]:index_list[j][1] + 1])
                    # print(sentece_list[index_list[j][0]:index_list[j][1]+1])
                    example_dict['target']['span2_text'] = sum_list_to_str(sentece_list[index_list[jj][0]:index_list[jj][1] + 1])
                    # print(sentece_list[index_list[jj][0]:index_list[jj][1]+1])
                    example_dict['target']['span1_index'] = index_list[j][0]
                    example_dict['target']['span2_index'] = index_list[jj][0]
                else:
                    example_dict['target']['span1_text'] = sum_list_to_str_with_space(sentece_list[index_list[j][0]:index_list[j][1] + 1])
                    # print(sentece_list[index_list[j][0]:index_list[j][1]+1])
                    example_dict['target']['span2_text'] = sum_list_to_str_with_space(sentece_list[index_list[jj][0]:index_list[jj][1] + 1])
                    # print(sentece_list[index_list[jj][0]:index_list[jj][1]+1])
                    example_dict['target']['span1_index'] = index_list[j][0]
                    example_dict['target']['span2_index'] = index_list[jj][0]

                if_in_same_list_result = is_in_list(new_line['clusters'], index_list[j], index_list[jj])
                if if_in_same_list_result:
                    example_dict['label'] = "true"
                else:
                    example_dict['label'] = "false"
                idx += 1
                example_dict['idx'] = idx
                if language == 'chinease':
                    example_dict['text'] = sum_list_to_str(sentece_list)
                else:
                    example_dict['text'] = sum_list_to_str_with_space(sentece_list)
                add_dict_to_json(json_path, example_dict)
                # if is_file_over_xGB(json_path, x):
                #     file_index += 1
                #     json_path = json_path.split('.')[0] + str(file_index) + '.json'


if __name__ == "__main__":
    # Define the file path
    json_path_exp = "data/train.json"
    example_dict = replace_values_with_none(read_json_line(json_path_exp, 1))
    # language = sys.argv[1]
    language = 'chinese'
    if language == 'chinese':
        file_llist = ['data/chinese/train/train.chinese.128.jsonlines', 'data/chinese/dev/dev.chinese.128.jsonlines',
                      'data/chinese/test/test.chinese.128.jsonlines']
        json_path = ['data/chinese/train/train.json', 'data/chinese/dev/dev.json', 'data/chinese/test/test.json']
    else:
        jsonlines_path = "data/train/train.jsonlines"
        file_llist = ['data/english/train/train.english.128.jsonlines', 'data/english/dev/dev.english.128.jsonlines',
                      'data/english/test/test.english.128.jsonlines']
        json_path = ['data/english/train/train.json', 'data/english/dev/dev.json', 'data/english/test/test.json']
    for i in range(0, len(file_llist)):
        main(language, file_llist[i], example_dict, json_path[i], 2)
    # filelist = get_file_list('./data/chinese/dev/', '.json')
    # merge(filelist, './data/chinese/dev/dev.json')
    #
    # filelist = get_file_list('./data/chinese/test/', '.json')
    # merge(filelist, './data/chinese/test/test.json')
    #
    # filelist = get_file_list('./data/chinese/train/', '.json')
    # merge(filelist, './data/chinese/train/train.json')

