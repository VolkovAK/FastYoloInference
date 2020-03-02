import re


def get_net_width(cfg):
    cfg = cfg.split('\n')
    widths = list(filter(lambda x: (x.find('#') > x.find('width')) & ('width' in x), cfg))  # get all lines with word "width" which not commented
    width = widths[0]  # use only first match
    try:
        net_width = int(re.sub('[^0-9]', '', width))  # get width (and height, since input is square)
    except ValueError:
        raise ValueError('Can not get width of network input, check .cfg file!')
    return net_width


def get_anchors(cfg):
    cfg = cfg.split('\n')
    anchors = list(filter(lambda x: (x.find('#') > x.find('anchors')) & ('anchors' in x), cfg))
    anchors = anchors[0]
    try:
        anchors = re.search('[0-9][0-9, ]*[0-9]', anchors)
        if anchors is None:
            raise ValueError('Can not get acnhors, check .cfg file!')
        else:
            anchors = anchors.group()
            anchors = list(map(int, anchors.replace(' ', '').split(',')))
            anchors = [(anchors[i], anchors[i+1])for i in range(0, len(anchors), 2)]

    return anchors


def get_masks(cfg):  # masks for anchors
    pass
