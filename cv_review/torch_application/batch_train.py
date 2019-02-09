import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as Data

torch.manual_seed(1)

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
	dataset=torch_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=2,
)


def show_batch():
	for epoch in range(3):
		for step, (batch_x, batch_y) in enumerate(loader):
			print('epoch: ', epoch, '|step ', step, '|batch_x', batch_x, '|batch_y', batch_y)


if __name__ == '__main__':
	show_batch()
