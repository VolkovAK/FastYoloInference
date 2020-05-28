import re


def get_net_input_param(cfg, param):
    cfg = cfg.split('\n')
    try:
        for line in cfg:
            if param in line:
                if '#' in line:
                    if line.find('#') > line.find(param):
                        param_line = line
                        break
                else:
                    param_line = line
                    break
        numbers_in_line = re.findall('[0-9]*', param_line) # get all numbers in line 
        params = list(map(int, filter(lambda x: len(x) > 0, numbers_in_line))) # some filtering
        input_param = params[0]  # get first param (second maybe commented value)
    except ValueError:
        raise ValueError('Can not get {} of network input, check .cfg file!'.format(param))
    return input_param

def get_net_width(cfg):
    return get_net_input_param(cfg, 'width')

def get_net_height(cfg):
    return get_net_input_param(cfg, 'height')


def get_all_occurrences(cfg, to_find):
    cfg = cfg.split('\n')
    occurs = []
    for line in cfg:
        if to_find in line:
            if '#' in line:
                if line.find('#') > line.find(to_find):
                    occurs.append(line)
            else:
                occurs.append(line)
    return occurs

def filter_comments(lines_list):
    filtered_list = []
    for line in lines_list:
        if line.find('#') != -1:
            line = line[:line.find('#')]
        filtered_list.append(line)
    return filtered_list



def get_anchors(cfg): # all anchors, however in the most cases they would be the same
    anchors = get_all_occurrences(cfg, 'anchors')
    anchors = filter_comments(anchors)
    parsed_anchors = []
    for anchor in anchors:
        anchors = re.findall('[0-9]+', anchor)
        if len(anchors) == 0:
            raise ValueError('Can not get anchors, check .cfg file!')
        anchors = list(map(int, anchors))
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
        parsed_anchors.append(anchors)
    return parsed_anchors


def get_masks(cfg):  # masks for anchors
    masks = get_all_occurrences(cfg, 'mask')    
    masks = filter_comments(masks)
    parsed_masks = []
    for mask in masks:
        anchors_idx = re.findall('[0-9]+', mask)
        if len(anchors_idx) == 0:
            raise ValueError('Can not get masks, check .cfg file!')
        parsed_masks.append([int(a) for a in anchors_idx])
    return parsed_masks 


def get_classes(cfg):
    classes = get_all_occurrences(cfg, 'classes')
    classes = filter_comments(classes)
    parsed_classes = []
    for cl in classes:
        cl = re.findall('[0-9]+', cl)
        if len(cl) == 0:
            raise ValueError('Can not get classes, check .cfg file!')
        parsed_classes.append(int(cl[0]))
    if len(set(parsed_classes)) != 1:
        raise ValueError('There must be the same "classes" in each [yolo], check .cfg file!')
    return parsed_classes[0] # we need only one number
 

