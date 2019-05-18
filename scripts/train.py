from dataloaders.hand_tracking_dataset import ImgSeqDataset, ToTensor, RandomCrop, Rescale
from models.img_feat_extractor import img_feat_extractor
from models.patch_feat_extractor import patch_feat_extractor
from models.unet import UNet

import torch
from torchvision import transforms

def train_ucl():

    experiment_name = "test1"
    epochs = 100
    batch_size = 20
    img_size = [3, 240,240]
    patch_size = [3, 80, 80]
    train_new = True
    device = 'cuda'
    lr = 0.0001

    transforms_train = transforms.Compose([
        Rescale(output_size=(256, 256)),
        RandomCrop(input_size=(256, 256), output_size=(img_size[1], img_size[2])),
        ToTensor()
    ])

    weights_img = "..//Weights//img_weigts" + experiment_name + ".pt"
    weights_patch = "..//Weights//patch_weigts" + experiment_name + ".pt"
    weights_attenton = "..//Weights//attention_weigts" + experiment_name + ".pt"

    model_img = img_feat_extractor(weights=weights_img, train_new=train_new)
    model_patch = patch_feat_extractor(weights=weights_patch, train_new=train_new)
    model_attention = UNet(n_channels=img_size[0], n_classes=1, weights=weights_attenton, train_new=train_new)

    model_img = model_img.to(device)
    model_patch = model_patch.to(device)
    model_attention = model_attention.to(device)

    # set up optimizer
    optimizer_img = torch.optim.Adam(model_img.parameters(), lr=lr)
    optimizer_patch = torch.optim.Adam(model_patch.parameters(), lr=lr)
    optimizer_attention = torch.optim.Adam(model_attention.parameters(), lr=lr)

    dataloader = ImgSeqDataset('D:\\Anton\\data\\hand_gesture_sequences\\test', None, batch_size, transform=transforms_train)

    for e in range(epochs):

        for i in range(len(dataloader)):
            processed = dataloader[i]

            imgs, landmark, label =processed



if __name__ == "__main__":
    train_ucl()