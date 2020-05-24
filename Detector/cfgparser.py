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



def get_anchors(cfg):
    cfg = cfg.split('\n')

    anchors = []
    for line in cfg:
        if 'anchors' in line:
            if '#' in line:
                if line.find('#') > line.find('anchors'):
                    anchors.append(line)
            else:
                anchors.append(line)

    parsed_anchors = []
    for anchor in anchors:
        anchor = re.search('[0-9][0-9, ]*[0-9]', anchor)
        if anchor is None:
            raise ValueError('Can not get anchors, check .cfg file!')
        else:
            anchor = anchor.group()
            anchor = list(map(int, anchor.replace(' ', '').split(',')))
            anchor = [(anchor[i], anchor[i+1])for i in range(0, len(anchor), 2)]
            parsed_anchors.append(anchor)

    return parsed_anchors


def get_masks(cfg):  # masks for anchors
    pass
