import re

def match_acc_cam(path):
    pattern = "EPOCH: \d+--> training acc is ([0-9]*.[0-9]*) f1:[0-9]*.[0-9]* auc:[0-9]*.[0-9]* recall:[0-9]*.[0-9]* precision:[0-9]*.[0-9]*"

    acc = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            matches = re.finditer(pattern, line)
            for match in matches:
                acc.append(100-float(match.group(1)))
    return acc

def match_acc_uncertain(path):
    pattern = "EPOCH: \d+--> training_set acc is ([0-9]*.[0-9]*) f1:[0-9]*.[0-9]* auc:[0-9]*.[0-9]* recall:[0-9]*.[0-9]* precision:[0-9]*.[0-9]*"

    acc = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            matches = re.finditer(pattern, line)
            for match in matches:
                acc.append(100-float(match.group(1)))
    return acc

def match_acc_without(path):
    pattern = "EPOCH: \d+--> training acc is ([0-9]*.[0-9]*) f1:[0-9]*.[0-9]* auc:[0-9]*.[0-9]* recall:[0-9]*.[0-9]* precision:[0-9]*.[0-9]*"

    acc = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            matches = re.finditer(pattern, line)
            for match in matches:
                acc.append(100-float(match.group(1)))
    return acc