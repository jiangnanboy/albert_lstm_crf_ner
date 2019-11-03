# -*- coding:utf-8 -*-

def format_result(result, text, tag): 
    entities = [] 
    for i in result: 
        begin, end = i 
        entities.append({ 
            "start":begin, 
            "stop":end + 1, 
            "word":text[begin:end+1],
            "type":tag
        }) 
    return entities

def get_tags(path, tag, tag_map):
    begin_tag = tag_map.get("B_" + tag)
    mid_tag = tag_map.get("I_" + tag)
    tags = []

    for index_1 in range(len(path)):
        if path[index_1] == begin_tag:
            ner_index = 0
            for index_2 in range(index_1 + 1, len(path)):
                if path[index_2] == mid_tag:
                    ner_index += 1
                else:
                    break
            if ner_index != 0:
                tags.append([index_1, index_1 + ner_index])
    return tags

def f1_score(tar_path, pre_path, tag, tag_map):
    '''
    :param tar_path:  real tag
    :param pre_path:  predict tag
    :param tag: [ORG, PER, LOC, T]
    :param tag_map: { 'B_T': 0,
                        'I_T': 1,
                        'B_LOC': 2,
                        'I_LOC': 3,
                        'B_ORG': 4,
                        'I_ORG': 5,
                        'B_PER': 6,
                        'I_PER': 7,
                        'O': 8}
    :return:
    '''
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(tar, tag, tag_map)
        pre_tags = get_tags(pre, tag, tag_map)

        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    print("\t{}\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(tag, recall, precision, f1))
    return recall, precision, f1
