import torch
from collections import OrderedDict
import numpy as np

def parse_cfg(cfgfile):
    def erase_comment(line):
        line = line.split('#')[0]
        return line
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = OrderedDict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            line = erase_comment(line)
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks

def save_cfg(layers,filename):
    not_write = ['type', 'weight', 'bias', 'var', 'mean', 'scale','shift']
    with open(filename,'w') as f:
        for block in layers:
            f.write('[%s]\n' % (block['type']))
            for key,value in block.items():
                if key != 'type' and key not in not_write:
                    f.write('%s=%s\n' % (key, value))
            f.write('\n')


def save_weights(layers,filename):
    buf = np.array([], dtype=np.float32)
    for id, block in enumerate(layers):
        if block['type'] in ["convolutional","deconvolutional"] :
            batch_normalize = int(block['batch_normalize'])
            bias = block['bias']
            if batch_normalize==1 and bias is not None or batch_normalize==0 and bias is None:
                print(block['type'])
            if batch_normalize:
                buf = np.append(buf, np.reshape(block['shift'], [-1]).astype(np.float32))
            else:
                buf = np.append(buf, np.reshape(block['bias'], [-1]).astype(np.float32))
            if batch_normalize == 1:
                buf = np.append(buf, np.reshape(block['scale'], [-1]).astype(np.float32))
                buf = np.append(buf, np.reshape(block['mean'], [-1]).astype(np.float32))
                buf = np.append(buf, np.reshape(block['var'], [-1]).astype(np.float32))
            buf = np.append(buf, np.reshape(block['weight'], [-1]).astype(np.float32))

    fp = open(filename, 'wb')
    header = np.array([0, 2, 0, 0, 0], dtype=np.int32)
    header.tofile(fp)
    buf.tofile(fp)

def print_cfg(layers):
    not_write = ['type', 'weight', 'bias', 'var', 'mean', 'scale','shift']
    for block in layers:
        print('[%s]' % (block['type']))
        for key,value in block.items():
            if key != 'type' and key not in not_write:
                print('%s=%s' % (key, value))
        print('')

def parse_weights(blocks,weights):
    fp = open(weights,'rb')
    header = np.fromfile(fp, count=5, dtype=np.int32)
    buf = np.fromfile(fp, dtype = np.float32)
    start = 0
    print('layer     filters    size              input                output')
    batch = 64
    prev_height = 608
    prev_width = 608
    prev_filters = 3
    out_filters =[]
    out_widths =[]
    out_heights =[]
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            batch_normalize = int(block['batch_normalize'])
            pad = (kernel_size-1)/2 if is_pad else 0
            width = int((prev_width + 2*pad - kernel_size)/stride + 1)
            height = int((prev_height + 2*pad - kernel_size)/stride + 1)
            numb = filters
            numw = prev_filters*filters*kernel_size*kernel_size
            bias = buf[start:start + numb].copy() ; start+=numb
            block.update({"bias":bias})
            if batch_normalize ==1 :
                scale = buf[start:start + numb].copy(); start+=numb
                block.update({"scale": scale})
                mean = buf[start:start + numb].copy(); start+=numb
                block.update({"mean": mean})
                var = buf[start:start + numb].copy(); start+=numb
                block.update({"var": var})

            weight = np.reshape(buf[start:start + numw].copy(),[filters,prev_filters,kernel_size,kernel_size]); start+= numw
            block.update({"weight": weight})
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d ' % (
            ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width,
            height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width/stride
            height = prev_height/stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->      %3d' % (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
            prev_width = 1
            prev_height = 1
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->      %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->      %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            print('%5d %-6s %s' % (ind, 'route',', '.join([str(layer) for layer in layers])))
            prev_width = out_widths[layers[0]]
            prev_height = out_heights[layers[0]]
            prev_filters = sum(out_filters[layer] for layer in layers)
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'yolo' or block['type'] == 'ctdet':
            print('%5d %-6s' % (ind, block['type']))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            block['prev_filters']=prev_filters
            print('%5d %-6s %d  ' % (ind, 'shortcut', from_id))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s' % (ind, 'softmax'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->      %3d' % (ind, 'connected', prev_filters,  filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            width = prev_width*stride
            height = prev_height*stride
            print('%5d %-6s            %dx   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))

    assert start == len(buf),print(len(buf),start)
    return header[3]/(batch)

if __name__ == '__main__':
    import sys
    cfg = "dla34_ctdet.cfg"
    weights = "dla34_ctdet.weights"
    blocks = parse_cfg(cfg)
    parse_weights(blocks,weights)
    pass