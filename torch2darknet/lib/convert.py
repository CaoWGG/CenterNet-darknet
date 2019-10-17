from tensorboardX.pytorch_graph import NodePyIO,NodePyOP,GraphPy
from collections import OrderedDict
from .cfg import print_cfg
import torch

def parse(graph, args=None, omit_useless_nodes=True):
    n_inputs = len(args)
    nodes_py = GraphPy()
    for i, node in enumerate(graph.inputs()):
        if omit_useless_nodes:
            if len(node.uses()) == 0:
                continue
        if i < n_inputs:
            nodes_py.append(NodePyIO(node, 'input'))
        else:
            nodes_py.append(NodePyIO(node))
    for node in graph.nodes():
        nodes_py.append(NodePyOP(node))

    for node in graph.outputs():
        NodePyIO(node, 'output')
    nodes_py.find_common_root()
    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py

def parse_graph(model, args, verbose=False):
    def _optimize_trace(trace, operator_export_type):
        trace.set_graph(_optimize_graph(trace.graph(), operator_export_type))
    def _optimize_graph(graph, operator_export_type):
        torch._C._jit_pass_constant_propagation(graph)
        torch.onnx.utils._split_tensor_list_constants(graph, graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_peephole(graph, True)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_prepare_division_for_onnx(graph)
        torch._C._jit_pass_erase_number_types(graph)
        torch._C._jit_pass_lower_all_tuples(graph)
        torch._C._jit_pass_peephole(graph, True)
        torch._C._jit_pass_lint(graph)
        if operator_export_type != torch.onnx.utils.OperatorExportTypes.RAW:
            graph = torch._C._jit_pass_onnx(graph, operator_export_type)
            torch._C._jit_pass_lint(graph)
            torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_fixup_onnx_loops(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        return graph
    with torch.onnx.set_training(model, False):
        trace, _ = torch.jit.get_trace_graph(model, args)
    operator_export_type = torch.onnx.utils.OperatorExportTypes.ONNX
    omit_useless_nodes = True
    _optimize_trace(trace, operator_export_type)
    graph = trace.graph()
    node_py = parse(graph,args,omit_useless_nodes)
    startop_offset = int(node_py.nodes_op[0].outputs[0])
    node_dict = node_py.nodes_io
    weights = model.state_dict()
    import re
    pattern = re.compile(r'\[(.*?)\]')
    parse_dict = OrderedDict()
    for key,value  in  node_dict.items():
        key = int(key)
        if key< startop_offset:
            continue
        layer_type = value.kind
        name = '.'.join(pattern.findall(value.uniqueName))
        if layer_type == 'onnx::Conv':
            input = int(value.inputs[0].rsplit('/',1)[-1])
            if input+1 != key and input !=0 :
                block= {'type':'route',
                        'layers':input-key}
                parse_dict[str(key - startop_offset)+'.route'] = block
            weight_offset = int(value.inputs[1].rsplit('/',1)[-1])
            bias_offset = int(value.inputs[2].rsplit('/',1)[-1]) if  len(value.inputs) >2 else False
            attributes = value.attributes.replace(' ','')
            group= re.findall('group:([0-9])',attributes)[0]
            strides = re.findall('strides:(\[.*?\])',attributes)[0]
            pads = re.findall('pads:(\[.*?\])',attributes)[0]
            size = re.findall('kernel_shape:(\[.*?\])',attributes)[0]
            dilations = re.findall('dilations:(\[.*?\])', attributes)[0]
            block = {'type':'conv',
                     'filters':value.tensor_size[1],
                     'strides':strides,
                     'size':size,
                     'dilations':dilations,
                     'group':group,
                     'pads':pads,
                     'weight':weights[name+'.weight'].numpy() ,
                     'bias':weights[name+'.bias'].numpy() if bias_offset else None}
            assert tuple(node_dict[str(weight_offset)].tensor_size)==block['weight'].shape,print(node_dict[str(weight_offset)].tensor_size,block['weight'].shape)
            if bias_offset:
                assert tuple(node_dict[str(bias_offset)].tensor_size) == block['bias'].shape,print(node_dict[str(bias_offset)].tensor_size,block['bias'].shape)
            parse_dict[key - startop_offset] = block
        elif layer_type == 'onnx::BatchNormalization':
            input = int(value.inputs[0].rsplit('/',1)[-1])
            if input+1 != key:
                raise ValueError()
            attributes = value.attributes.replace(' ', '')
            eps= re.findall('epsilon:(.*?)[,}]',attributes)[0]
            mom = re.findall('momentum:(.*?)[,}]',attributes)[0]
            bias = weights[name+'.bias'].numpy()
            weight = weights[name+'.weight'].numpy()
            running_mean = weights[name+'.running_mean'].numpy()
            running_var = weights[name + '.running_var'].numpy()
            block = {'type':'bn',
                     'filters':value.tensor_size[1],
                     'eps':eps,
                     'mom':mom,
                     'weight':weight,
                     'bias':bias,
                     'mean':running_mean,
                     'var':running_var}
            weight_offset = int(value.inputs[1].rsplit('/', 1)[-1])
            bias_offset = int(value.inputs[2].rsplit('/', 1)[-1])
            mean_offset = int(value.inputs[3].rsplit('/', 1)[-1])
            var_offset = int(value.inputs[4].rsplit('/', 1)[-1])
            assert tuple(node_dict[str(weight_offset)].tensor_size) == block['weight'].shape,print(node_dict[str(weight_offset)].tensor_size,block['weight'].shape)
            assert tuple(node_dict[str(bias_offset)].tensor_size) == block['bias'].shape,print(node_dict[str(bias_offset)].tensor_size,block['bias'].shape)
            assert tuple(node_dict[str(mean_offset)].tensor_size) == block['mean'].shape,print(node_dict[str(mean_offset)].tensor_size,block['mean'].shape)
            assert tuple(node_dict[str(var_offset)].tensor_size) == block['var'].shape,print(node_dict[str(var_offset)].tensor_size,block['var'].shape)

            parse_dict[key - startop_offset] = block
        elif layer_type == 'onnx::Relu':
            input = int(value.inputs[0].rsplit('/',1)[-1])
            if input+1 != key:
                raise ValueError()
            block = {'type':'act',
                     'act' : 'relu'}
            parse_dict[key - startop_offset] = block
        elif layer_type == 'onnx::MaxPool':
            input = int(value.inputs[0].rsplit('/',1)[-1])
            if input+1 != key:
                if input + 1 != key and input != 0:
                    block = {'type': 'route',
                             'layers': input - key}
                    parse_dict[str(key - startop_offset) + '.route'] = block
            attributes = value.attributes.replace(' ', '')
            strides = re.findall('strides:(\[.*?\])',attributes)[0]
            pads = re.findall('pads:(\[.*?\])',attributes)[0]
            size = re.findall('kernel_shape:(\[.*?\])',attributes)[0]
            block= {'type':'maxpool',
                    'pads':pads,
                    'strides':strides,
                    'size':size}
            parse_dict[key - startop_offset] = block
        elif layer_type == 'onnx::Add':
            inputs = [int(input.rsplit('/', 1)[-1])-key for input in value.inputs]
            block= {'type':'shortcut',
                    'from':inputs}
            parse_dict[key - startop_offset] = block
        elif layer_type == 'onnx::Concat':
            inputs = [int(input.rsplit('/', 1)[-1]) - key for input in value.inputs]
            block= {'type':'route',
                    'layers':inputs}
            parse_dict[key - startop_offset] = block
        elif layer_type == 'onnx::ConvTranspose':
            input = int(value.inputs[0].rsplit('/', 1)[-1])
            if input + 1 != key and input != 0:
                block = {'type': 'route',
                         'layers': input - key}
                parse_dict[str(key - startop_offset) + '.route'] = block
            weight_offset = int(value.inputs[1].rsplit('/', 1)[-1])
            bias_offset = int(value.inputs[2].rsplit('/', 1)[-1]) if len(value.inputs) > 2 else False
            attributes = value.attributes.replace(' ', '')
            group = re.findall('group:([0-9])', attributes)[0]
            strides = re.findall('strides:(\[.*?\])', attributes)[0]
            pads = re.findall('pads:(\[.*?\])', attributes)[0]
            size = re.findall('kernel_shape:(\[.*?\])', attributes)[0]
            dilations = re.findall('dilations:(\[.*?\])', attributes)[0]
            block = {'type': 'deconv',
                     'filters': value.tensor_size[1],
                     'strides': strides,
                     'size': size,
                     'dilations': dilations,
                     'group': group,
                     'pads': pads,
                     'weight': weights[name + '.weight'].numpy(),
                     'bias': weights[name + '.bias'].numpy() if bias_offset else None}
            assert tuple(node_dict[str(weight_offset)].tensor_size) == block['weight'].shape,print(node_dict[str(weight_offset)].tensor_size,block['weight'].shape)
            if bias_offset:
                assert tuple(node_dict[str(bias_offset)].tensor_size) == block['bias'].shape,print(node_dict[str(bias_offset)].tensor_size,block['bias'].shape)
            parse_dict[key - startop_offset] = block
        else:
            raise ValueError()



    id_old2new = {}
    layers = []
    net = {'type':'net',
            'batch' : 64,
            'subdivisions': 16,
            'width': 512,
            'height' : 512,
            'channels' : 3,
            'momentum' : 0.9,
            'decay' : 0.0005,
            'angle' : 0,
            'saturation' : 1.5,
            'exposure' : 1.5,
            'hue' : .1,
            'learning_rate' : 0.001,
            'burn_in' : 1000,
            'max_batches' : 500200,
            'policy' : "steps",
            'steps' : "325130, 425170",
            'scales' : ".1, .1,"}

    new_id = 0

    for key, block in parse_dict.items():
        layer_type = block['type']
        if layer_type == 'conv':
            bias = block['bias']
            weight = block['weight']
            kernel_size = eval(block['size'])[0]
            padding = eval(block['pads'])[0]
            filters = block['filters']
            group = int(block['group'])
            stride = eval(block['strides'])[0]
            pad = 0
            if padding == kernel_size // 2:
                pad = 1
            new_block = {'type': 'convolutional',
                         'filters': filters,
                         'stride': stride,
                         'pad': pad,
                         'padding': padding,
                         'size': kernel_size,
                         'weight': weight,
                         'bias': bias,
                         'activation':'linear',
                         'batch_normalize':0}
            layers.append(new_block)
            id_old2new[key] = new_id
            new_id += 1

        elif layer_type == 'deconv':
            ####  darknet not support group deconv
            # bias = block['bias']
            # weight = block['weight']
            # kernel_size = eval(block['size'])[0]
            # padding = eval(block['pads'])[0]
            # filters = block['filters']
            # group = int(block['group'])
            # stride = eval(block['strides'])[0]
            # pad = 0
            # if padding == kernel_size // 2:
            #     pad = 1
            # new_block = {'type': 'deconvolutional',
            #              'filters': filters,
            #              'stride': stride,
            #              'pad': pad,
            #              'padding': padding,
            #              'size': kernel_size,
            #              'weight': weight,
            #              'bias': bias,
            #              'activation': 'linear',
            #              'batch_normalize': 0}
            new_block = {'type':'upsample',
                         'stride':2}
            layers.append(new_block)
            id_old2new[key] = new_id
            new_id += 1

        elif layer_type == 'bn':
            pre_type = layers[new_id - 1]['type']
            if pre_type not in ['convolutional', 'deconvolutional']:
                raise ValueError()
            scale = block['weight']
            shift = block['bias']
            mean = block['mean']
            var = block['var']
            layers[new_id - 1]['batch_normalize'] = 1
            layers[new_id - 1]['scale'] = scale
            layers[new_id - 1]['shift'] = shift
            layers[new_id - 1]['mean'] = mean
            layers[new_id - 1]['var'] = var
            id_old2new[key] = new_id - 1

        elif layer_type == 'act':
            pre_type = layers[new_id - 1]['type']

            if pre_type not in ['convolutional', 'deconvolutional', 'shortcut']:
                raise ValueError()
            layers[new_id - 1]['activation'] = 'relu'
            id_old2new[key] = new_id - 1

        elif layer_type == 'maxpool':
            kernel_size = eval(block['size'])[0]
            padding = eval(block['pads'])[0]
            stride = eval(block['strides'])[0]
            new_block = {'type': 'maxpool',
                         'stride': stride,
                         'padding': padding,
                         'size': kernel_size, }
            layers.append(new_block)
            id_old2new[key] = new_id
            new_id += 1
        elif layer_type == 'route':
            int_key = key if type(key) == int else int(key.split('.')[0])
            from_id = block['layers']
            from_id_new = []
            if type(from_id) == int:
                from_id = [from_id]
            for f_id in from_id:
                new_f_id = id_old2new[f_id + int_key]
                from_id_new.append(str(new_f_id - new_id))
            new_block = {'type': 'route',
                         'layers': ','.join(from_id_new)}
            layers.append(new_block)
            id_old2new[key] = new_id
            new_id += 1

        elif layer_type == 'shortcut':
            from_id = block['from']
            from_id_new = []
            if type(from_id) == int:
                from_id = [from_id]
            for f_id in from_id:
                new_f_id = id_old2new[f_id + key] - new_id
                if new_f_id <= -2:
                    from_id_new.append(str(new_f_id))
            if len(from_id_new) != 1:
                raise ValueError()
            new_block = {'type': 'shortcut',
                         'from': ''.join(from_id_new)}
            layers.append(new_block)
            id_old2new[key] = new_id
            new_id += 1
        else:
            raise ValueError
    layers.insert(0,net)
    layers.append({'type':'route',
                   'layers':'-1,-4,-7'})

    layers.append({'type':'ctdet',
                   'classes':80,
                   'jitter':0.3,
                   'hm':1.0,
                   'off':1.0,
                   'wh' : 0.1,
                   'stride' :1,
                   'size':3,
                   'padding':0})
    if verbose:
        print_cfg(layers)
    return layers


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model