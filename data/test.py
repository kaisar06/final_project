from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter('runs/test_run')
for i in range(100):
    writer.add_scalar('DummyLoss', 100 - i, i)
writer.close()
