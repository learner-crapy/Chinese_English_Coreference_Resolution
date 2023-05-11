#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jsonlines
import json

# In[2]:




def count_lines_in_jsonlines_file(file_path):
    """
    Counts the number of lines in a jsonlines file.
    :param file_path: The path to the jsonlines file.
    :return: The number of lines in the file.
    """
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
    return count


def sum_list(arg):
    O_List = sum(arg, [])
    return O_List


def count_sublist_lengths(lst):
    """
    Counts the length of each sublist in a 2D list and returns a dictionary with the index of each sublist as the key
    and the length of the corresponding sublist as the value.
    :param lst: The 2D list to count the lengths of sublists for.
    :return: A dictionary with the index of each sublist as the key and the length of the corresponding sublist as the value.
    """
    lengths_dict = {}
    for i, sublst in enumerate(lst):
        lengths_dict[i] = len(sublst)
    return lengths_dict


def split_list(lst, split_dict):
    """
    Splits a 1D list into a 2D list according to a dictionary where the keys represent the indices of the resulting
    sublists and the values represent the length of each sublist. Raises an error if the total length of the sublists
    does not match the length of the input list.
    :param lst: The 1D list to split.
    :param split_dict: The dictionary representing the indices and lengths of the resulting sublists.
    :return: The 2D list resulting from the split.
    """
    # Check if the total length of the sublists matches the length of the input list
    if sum(split_dict.values()) != len(lst):
        raise ValueError("Target lengths do not match input list length.")

    # Initialize variables
    result = []
    start_index = 0

    # Iterate over the split_dict and create sublists according to the specified lengths
    for index, length in split_dict.items():
        end_index = start_index + length
        result.append(lst[start_index:end_index])
        start_index = end_index

    return result


def get_jsonlines_content(jsonlines_path, key, k):
    with jsonlines.open(jsonlines_path) as reader:
        for i, obj in enumerate(reader):
            if i == k:
                if key in obj:
                    return {key: obj[key]}
                else:
                    return {}


# Example usage:

# content = get_jsonlines_content(jsonlines_path, key,1)['sentences']

def read_jsonlines_line(file_path, k):
    """
    Reads the kth line of a jsonlines file.
    :param file_path: The path to the jsonlines file.
    :param k: The index of the line to read (0-indexed).
    :return: The contents of the kth line as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == k:
                return json.loads(line)
    raise ValueError("Line index out of range.")


# print(read_jsonlines_line(jsonlines_path,1))

def RemoveData(sentence_list, index_list):
    '''
    sentence_list: list of lists of strings, 1D list
    index_list: list of lists of tuples, 2D list
    function: remove the data from the sentence list based on the index list
    '''
    for i in index_list:
        for j in range(i[0], i[1] + 1):
            sentence_list[j] = ''
    return sentence_list


def main(jsonlines_path):
    print('processing file: ----------------', jsonlines_path, '------------------')
    line_nums = count_lines_in_jsonlines_file(jsonlines_path)
    for i in range(0, line_nums):
        print('processing file: ', jsonlines_path, '---------------- line', i, '------------------')
        new_line = read_jsonlines_line(jsonlines_path, i)
        senteces_i = new_line['sentences']
        sentece_list = sum_list(senteces_i)
        index_list = sum_list(new_line['clusters'])
        new_sentece_list = RemoveData(sentece_list, index_list)
        new_sentece_list_2D = split_list(new_sentece_list, count_sublist_lengths(senteces_i))
        new_line['sentences'] = new_sentece_list_2D
        new_line['clusters'] = []
        with open(jsonlines_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_line, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':
    jsonlines_path = ['./data/train/train.chinese.128.jsonlines', './data/dev/dev.chinese.128.jsonlines',
                      './data/test/test.chinese.128.jsonlines']
    for i in jsonlines_path:
        main(i)
