from collections import  OrderedDict
from lib.convert import parse_graph
from lib.dla_v0 import get_pose_net
from lib.cfg import parse_cfg,parse_weights,save_cfg,save_weights
import torch

dla34_cfg = 'dla34_ctdet.cfg'
dla34_weight = 'dla34_ctdet.weights'
heads = OrderedDict()
heads.update({'wh':2})
heads.update({'off':2})
heads.update({'hm':80})
net = get_pose_net(34,heads)
input = torch.zeros([1,3,512,512])
cfg = parse_graph(net,(input,),verbose=True)
# save_cfg(cfg,dla34_cfg)
# save_weights(cfg,dla34_weight)
#
#
# ## check weight and cfg
# parse_weights(parse_cfg(dla34_cfg),dla34_weight)
print("convert ok")