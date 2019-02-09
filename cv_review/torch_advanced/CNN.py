import os
import torch
import torchvision
import torch.utils.data as Data
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
import matplotlib.pyplot as plt

# hyper params
LR = 0.001  # when LR=0.1 destructive.... :(
EPOCH = 10
BATCH_SIZE = 64
DOWNLOAD_MNIST = False

# mnist digits dataset

if not os.path.exists('./mnist') or not os.listdir('./mnist/'):
	DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
	root='./mnist/',
	train=True,
	transform=torchvision.transforms.ToTensor(),  # convert PIL image to Tensor
	download=DOWNLOAD_MNIST
)

# plot example
print(train_data.train_data.size())
print(train_data.train_labels.size())

plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%d ' % train_data.train_labels[0])
plt.show()

train_loader = Data.DataLoader(
	dataset=train_data,
	shuffle=True,
	batch_size=BATCH_SIZE
)

test_data = torchvision.datasets.MNIST(
	root='./mnist/',
	train=False
)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255
test_y = test_data.test_labels[:2000].cuda()


class CNN(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(  # 1*28*28
				in_channels=1,  # the dimension of Filter
				out_channels=16,  # the number of filter
				kernel_size=5,
				stride=1,  # step
				padding=2,  # zero padding
			),  # 16 * 28 * 28
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)  # 16 * 14 * 14
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(2)  # 32 * 7 * 7
		)
		# need 'compress' here
		self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer,
	
	# output channel = class number
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # x.size(0) = batch_size
		out = self.out(x)
		return out, x  # return x for visualization
		pass


cnn = CNN().cuda()
print(cnn)

optimizer = optim.Adam(cnn.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()  # classification


# from matplotlib import cm

def train():
	for epoch in range(EPOCH):
		for step, (b_x, b_y) in enumerate(train_loader):
			output = cnn(b_x.cuda())[0]
			loss = criterion(output, b_y.cuda())  # move to GPU
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			if step % 50 == 0:
				test_output, last_layer = cnn(test_x)
				pred_y = torch.argmax(test_output, 1).data.cpu().numpy()
				acc = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
				print("EPOCH: {} | train loss: {:.5f} | test acc: {:.5f}".format(epoch, loss.data.cpu().numpy(), acc))
	torch.save(cnn, 'mnist_cnn.pkl')  # save the entire net


def test():
	net = torch.load('mnist_cnn.pkl')
	test_output, _ = net(test_x[:10])
	pred_y = torch.argmax(test_output, 1).data.cpu().numpy()
	print(pred_y, "prediction number")
	print(test_y[:10].cpu().numpy(), 'real number')


if __name__ == '__main__':
	train()
	test()
# torch.save(cnn.state_dict(), 'mnist_cnn_net_params.pkl') # save the params
