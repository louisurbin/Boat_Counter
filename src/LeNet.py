import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# TRAINING HYPERPARAMÈTRES 
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_OPTIMIZER = "adam"       # "adam" ou "sgd"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RESIZE = 128             # taille d'entrée (128x128)
DEFAULT_NUM_CLASSES = 7
DEFAULT_DROPOUT = 0.5
DEFAULT_SAVE_PATH = "lenet.pth"
DEFAULT_NUM_WORKERS = 4
RANDOM_SEED = 42
LOG_INTERVAL = 10                # affichage toutes les N itérations

class LeNet(nn.Module):
	def __init__(self, in_channels=3, num_classes=2, dropout=0.5):
		super().__init__()
		# Conv1: 128x128 -> (128 - 5 +1)=124 -> Pool1 -> 62
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=0)
		self.relu = nn.ReLU(inplace=True)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Conv2: 62 -> (62 - 5 +1)=58 -> Pool2 -> 29
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Conv3: 29 -> (29 - 3 +1)=27 -> Pool3 -> 13
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		# After Pool3 spatial size = 13 x 13, channels = 128
		flat_feats = 128 * 13 * 13

		# Fully connected head
		self.fc1 = nn.Linear(flat_feats, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, num_classes)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		# x expected resize -> 128x128 by the preprocessing/transforms
		x = self.relu(self.conv1(x))
		x = self.pool1(x)         # -> 62x62

		x = self.relu(self.conv2(x))
		x = self.pool2(x)         # -> 29x29

		x = self.relu(self.conv3(x))
		x = self.pool3(x)         # -> 13x13

		x = x.view(x.size(0), -1)
		x = self.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.relu(self.fc2(x))
		x = self.dropout(x)
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
	parser.add_argument("--resize", type=int, default=128, help="Resize short side to N (use square Resize). default=128 for 128x128 input")
	parser.add_argument("--num-classes", type=int, default=7, help="Default number of classes if not inferred from dataset (default=7)")
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

	# determine num_classes: prefer dataset classes, else CLI default
	num_classes = args.num_classes
	if hasattr(ds, 'classes') and len(ds.classes) > 0:
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