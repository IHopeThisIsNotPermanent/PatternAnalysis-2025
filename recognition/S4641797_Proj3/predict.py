import matplotlib.pyplot as plt
from IPython.display import clear_output

from util import DiceLoss
import torchvision.transforms as transforms
from dataset import LabeledImageDataset


def test(model, d):
    """A simple function that calcualtes the accuracy against the validation dataset.
    """
    loss_function = DiceLoss()

    transform = transforms.Compose([transforms.ToTensor(),])
    validate = LabeledImageDataset('./data/keras_slices_data/keras_slices_train',
                                                          './data/keras_slices_data/keras_slices_seg_train',
                                                        transform=transform)
    
    overall_loss = 0
    print(len(validate))
    for total, img in enumerate(validate):
        x, label = img[0].unsqueeze(0).to(d), img[1].unsqueeze(0).to(d)
        x_hat = model(x)
        overall_loss += loss_function(x_hat, label).item()
        print(f"{total}/{len(validate)}")
        clear_output(wait=True)
        
    print(f"test accuracy for all labels: {1 - overall_loss/total}")

def disp(model, d):
    """Prints a couple of example inputs, labeled images, and model results.
    """
    transform = transforms.Compose([transforms.ToTensor(),])
    validate = LabeledImageDataset('./data/keras_slices_data/keras_slices_train',
                                                          './data/keras_slices_data/keras_slices_seg_train',
                                                        transform=transform)
    
    for i in [0, 100, 1000, 10000]:
        img = validate[i]
        x, label = img[0].unsqueeze(0).to(d), img[1].unsqueeze(0).to(d)
        x_hat = model(x)

        fig, axs = plt.subplots(2,6)
        axs[0][0].imshow(x.cpu().numpy()[0][0])
        axs[0][1].imshow(label.cpu().detach().numpy()[0][0])
        axs[1][0].imshow(x_hat.cpu().detach().numpy()[0][0])
        axs[1][1].imshow(x_hat.cpu().detach().numpy()[0][1])
        axs[1][2].imshow(x_hat.cpu().detach().numpy()[0][2])
        axs[1][3].imshow(x_hat.cpu().detach().numpy()[0][3])
        axs[1][4].imshow(x_hat.cpu().detach().numpy()[0][4])
        axs[1][5].imshow(x_hat.cpu().detach().numpy()[0][5])
        plt.show()