import random

def shuffle_file(old_path, new_path, k_lines):
    with open(old_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # random.shuffle(lines)
    with open(new_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[:k_lines])


shuffle_file('./english/test.json', './english/test.json', 2000)
shuffle_file('./english/dev.json', './english/dev.json', 2000)