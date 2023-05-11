# generate a function to count how many lines in a json file
def count_lines_in_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return len(lines)


# generate a function to remove the whitespace lines in a json file
def remove_whitespace_lines_in_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(json_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip() != '':
                f.write(line)


# generate a function to replace some characters in a json file
def replace_characters_in_json_file(json_path, old, new):
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(json_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.replace(old, new))

# replace_characters_in_json_file('./test.json', '"label": false', '"label": "false"')
# print(remove_whitespace_lines_in_json_file('./test.json'))
print(count_lines_in_json_file('./dev.json'))
print(count_lines_in_json_file('./test.json'))
# print(remove_whitespace_lines_in_json_file('./train.json'))
# print(count_lines_in_json_file('./train.json'))