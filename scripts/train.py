from dataloaders.hand_tracking_dataset import ImgSeqDataset, ToTensor, RandomCrop, Rescale
from models.img_feat_extractor import img_feat_extractor
from models.patch_feat_extractor import patch_feat_extractor
from models.att_feat_extractor import att_feat_extractor

import torch
import random
from torchvision import transforms
import torch.nn.functional as F

def train_ucl():

    experiment_name = "test1"
    epochs = 100
    batch_size = 20
    random_episodes = 10

    n_feat = 256
    img_size = [3, 240,240]
    patch_size = [50, 50]
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

    model_img = img_feat_extractor(n_feat, weights=weights_img, train_new=train_new)
    model_patch = patch_feat_extractor(n_feat, weights=weights_patch, train_new=train_new)
    model_attention = att_feat_extractor(n_channels=img_size[0], n_classes=1, weights=weights_attenton, train_new=train_new)

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

            imgs = imgs.to(device)

            for j in range(random_episodes):
                random_episode(model_img, model_patch, model_attention, imgs, patch_size, device)




def random_episode(m_img, m_patch, m_att, imgs, patch_size, device):

    img_size = imgs[0].shape

    # extracting random patch location
    h, w = img_size[1:]
    new_h, new_w = patch_size
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)

    # creating mask for patch
    mask = torch.zeros(img_size, device=device)
    mask[:, top: top + new_h, left: left + new_w] = 1

    # extracting nontrivial mask for first image

    img1 = imgs[0][None,:,:,:]

    feat_1, mask1 = transition(m_img, m_patch, m_att, img1, mask, img1)

    # feat_main_img = m_img(img1)
    # feat_main_patch = m_patch(img1)
    #
    # img_seq = imgs[1:]

def transition(m_img, m_patch, m_att, img1, mask, img2):

    img1 = img1 * mask

    f1 = m_patch(img1)
    f2 = m_img(img2)

    att_m = compute_att_m(f1, f2)

    att_mask = m_att(att_m)

    return f1, att_mask

def compute_att_m(patch_f, img_f):


    x = torch.conv2d(img_f, patch_f, padding=3)
    shape = x.shape

    x = x.view(shape[0], -1)
    attn = F.softmax(x, dim=1)
    attn = attn.view(shape)

    return attn





if __name__ == "__main__":
    train_ucl()