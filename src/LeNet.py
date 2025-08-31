import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class LeNet(nn.Module):
	def __init__(self, in_channels=3, num_classes=2):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
		self.relu = nn.ReLU()
		self.pool = nn.AvgPool2d(2)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
		# after two pools, spatial depends on input size; we will flatten dynamically
		self.fc1 = nn.Linear(16 * 5 * 5, 120)  # assumes input resized to 32x32 by default
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, num_classes)

	def forward(self, x):
		x = self.pool(self.relu(self.conv1(x)))
		x = self.pool(self.relu(self.conv2(x)))
		x = x.view(x.size(0), -1)
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def train_one_epoch(model, device, loader, criterion, optimizer):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for imgs, targets in loader:
		imgs = imgs.to(device); targets = targets.to(device)
		optimizer.zero_grad()
		out = model(imgs)
		loss = criterion(out, targets)
		loss.backward(); optimizer.step()
		running_loss += loss.item() * imgs.size(0)
		_, preds = out.max(1)
		correct += (preds == targets).sum().item()
		total += imgs.size(0)
	return running_loss / max(1, total), correct / max(1, total)

def main():
	parser = argparse.ArgumentParser(description="Minimal LeNet training")
	parser.add_argument("--data", "-d", required=True, help="ImageFolder dataset root")
	parser.add_argument("--out", "-o", default="lenet.pth", help="Path to save state_dict")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--resize", type=int, default=32, help="Resize short side to N (use square Resize)")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()

	# transforms: resize -> to tensor -> simple normalization
	trans = transforms.Compose([
		transforms.Resize((args.resize, args.resize)),
		transforms.ToTensor(),
		transforms.Normalize([0.5]*3, [0.5]*3)
	])
	ds = datasets.ImageFolder(args.data, transform=trans)
	loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

	num_classes = len(ds.classes)
	device = torch.device(args.device)
	model = LeNet(in_channels=3, num_classes=num_classes).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	start_time = time.time()
	for epoch in range(1, args.epochs + 1):
		loss, acc = train_one_epoch(model, device, loader, criterion, optimizer)
		print(f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}  acc={acc:.3f}")
		torch.save(model.state_dict(), args.out)  # save after each epoch

	duration = time.time() - start_time
	print(f"Training finished. Model saved to {args.out}. Total time: {duration:.2f}s")

if __name__ == "__main__":
	main()