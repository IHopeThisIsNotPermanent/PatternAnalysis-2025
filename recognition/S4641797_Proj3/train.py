import matplotlib.pyplot as plt
from torch.optim import Adam
import torch
from dataset import load_dataset
from modules import ImprovedUNET

class DiceLoss:
    """
    From:
    https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    """
    def __init__(self, smooth=1):
        self.smooth = smooth
    
    def __call__(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def train(EPOCHS = 100):
    batch_size = 50
    x_dim = 256 * 128

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test = load_dataset()

    model = ImprovedUNET().to(d)

    loss_function = DiceLoss()

    optimizer = Adam(model.parameters(), lr=0.0001)
    print("Start training VAE...")
    model.train()

    for epoch in range(EPOCHS):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(d)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))

    print("Finish! Saving Model To Disk...")
    torch.save(model.state_dict(), 'model.pth')