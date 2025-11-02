import matplotlib.pyplot as plt
from torch.optim import Adam
from dataset import load_dataset
from modules import ImprovedUNET
from IPython.display import clear_output
import torch
from util import DiceLoss

def train(EPOCHS = 10):
    """
    Trains the model
    """
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