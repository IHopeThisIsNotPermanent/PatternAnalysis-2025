import matplotlib.pyplot as plt
from torch.optim import Adam
from dataset import load_dataset
from modules import ImprovedUNET
import torch.nn.functional as F 
from IPython.display import clear_output
import torch

class DiceLoss:
    """
    From:
    https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    Modified to handle multi-class targets and match input dimensions.
    """
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)

        targets = targets.long()
        if targets.dim() == 4:
            targets = targets[:, 0, :, :]
        elif targets.dim() != 3:
            raise ValueError(f"Unsupported target shape for DiceLoss: {targets.shape}. Expected [N, 1, H, W] or [N, H, W].")

        targets = F.one_hot(targets, num_classes=6) 

        targets = targets.permute(0, 3, 1, 2).float()

        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


def train(EPOCHS = 50):
    batch_size = 8

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test = load_dataset()

    model = ImprovedUNET(classes=6).to(d)

    loss_function = DiceLoss()

    optimizer = Adam(model.parameters(), lr=0.0001)
    print("Start training VAE...")
    model.train()

    for epoch in range(EPOCHS):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x, label = x[0].to(d),x[1].to(d)

            optimizer.zero_grad()

            x_hat = model(x)

            loss = loss_function(x_hat, label)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (batch_idx % 20 == 0):
                clear_output(wait=True)
                if (batch_idx*batch_size) == 0:
                    print("...")
                else:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
                fig, axs = plt.subplots(1,3)
                axs[0].imshow(x.cpu().numpy()[0][0])
                axs[1].imshow(1-x_hat.cpu().detach().numpy()[0][0])
                axs[2].imshow(label.cpu().detach().numpy()[0][0])
                plt.show()


        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size)) 

    print("Finish! Saving Model To Disk...")
    torch.save(model.state_dict(), 'model.pth')