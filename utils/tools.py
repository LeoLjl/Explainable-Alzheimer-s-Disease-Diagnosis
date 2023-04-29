import torch.distributed as dist
import pdb

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    # pdb.set_trace()
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor