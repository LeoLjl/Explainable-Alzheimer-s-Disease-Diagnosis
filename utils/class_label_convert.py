import torch
def get_cls_label(args):

    if 'ADNI' in args.data:
        # 用一个字典对结果进行保存
        dic = {}
        cls_file = '../data/train_data.txt'
        lines = open(cls_file, 'r').readlines()
        for line in lines:
            # strip 方法用于移除字符串头尾指定字符
            line = line.strip()
            image_id, label = line.split()
            dic[image_id] = label
        return dic


def name_list_to_cls_label(name_list, label_dic):
    # determine what attribute to use
    tensor_list = []
    for name in name_list:
        tensor_list.append(int(label_dic[name.split('.')[0]]))
    return torch.Tensor(tensor_list).long().cuda()