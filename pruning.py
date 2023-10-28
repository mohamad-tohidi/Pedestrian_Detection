import torch, torch.nn as nn
from torch.nn.utils import prune
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

pruning_param = 0.5

for name, m in model.model.named_modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
        prune.l1_unstructured(m, name='weight', amount=pruning_param)  # prune
        prune.remove(m, 'weight')  # make permanent

print(f'Model pruned to {sparsity(model.model):.3g} global sparsity')

ckpt = {
            'model': model.model,
            'train_args': {},  # save as dict
}

torch.save(ckpt, 'model_pruned_5.pt')

model = YOLO("model_pruned_5.pt")
# results = model.val("model_pruned_3.pt")
