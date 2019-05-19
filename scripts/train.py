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
    model_attention = att_feat_extractor(n_feat, weights=weights_attenton, train_new=train_new)

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
                loss = random_episode(model_img, model_patch, model_attention, imgs, patch_size, device)

                loss.backward()
                optimizer_img.step()
                optimizer_img.zero_grad()

                optimizer_patch.step()
                optimizer_patch.zero_grad()

                optimizer_attention.step()
                optimizer_attention.zero_grad()

                print("({}/{}) loss: {:.4f}".format(j + 1, random_episode, float(loss)))

            model_patch.save_model()
            model_img.save_model()
            model_attention.save_model()


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

    img1 = imgs[-1][None,:,:,:]
    img_seq = imgs[:-1]

    feat_1, att_1 = transition(m_img, m_patch, m_att, img1, mask, img1)

    # set up vars for backwards pass
    cur_feat = feat_1
    feats_back = []
    att_back = []

    for i, img in enumerate(img_seq):
        img = img[None,:,:,:]
        cur_feat, att = transition_feat(m_img, m_att, img, cur_feat)
        feats_back.append(cur_feat)
        att_back.append(att)

    # set up vars for backwards pass
    feats_forwards = []
    att_forwards = []

    for i, img in enumerate(reversed(img_seq)):
        img = img[None, :, :, :]
        cur_feat, att = transition_feat(m_img, m_att, img, cur_feat)
        feats_forwards.append(cur_feat)
        att_forwards.append(att)

    feat_final, att_final = transition_feat(m_img, m_att, img1, cur_feat)

    att_info = (att_1, att_final, att_forwards, att_back)
    feat_info = (feat_1, feat_final, feats_forwards, feats_back)

    loss = compute_loss(feat_info, att_info)
    return loss


def compute_loss(feat_info, att_info):
    a_first, a, att_forwards, att_back = att_info
    f_first, f_last, feats_forwards, feats_back = feat_info

    # criterion_frobenius = loss_criteria()
    criterion_l2 = torch.nn.MSELoss()

    long_loss = criterion_l2(f_last, f_first) + criterion_l2

    main_loss = 0

    for f in feats_forwards:
        main_loss += criterion_l2(f, f_first)

    for f in feats_back:
        main_loss += criterion_l2(f, f_first)

    loss = main_loss + 0.1 * long_loss

    return loss


class loss_criteria(torch.nn.HingeEmbeddingLoss):
    def __init__(self, margin=1.0, size_average=True, reduce=True):
        super(loss_criteria, self).__init__(margin, size_average, reduce)

    def forward(self, cs, ct):
        D = 1.0  # constant value
        x = ((torch.abs(cs - ct)) / (2 * D)) ** 2  # formula
        hinge_loss = torch.nn.HingeEmbeddingLoss()
        y = hinge_loss(x)  # sorry, couldn't write the whole correct equation here
        return y











def transition(m_img, m_patch, m_att, img1, mask, img2):

    img1 = img1 * mask

    f1 = m_patch(img1)
    f2 = m_img(img2)

    att_m = compute_attn(f1, f2)

    feat = m_att(torch.cat((att_m, f2), dim=1))

    return feat, att_m

def transition_feat(m_img, m_att, img, f1):

    f2 = m_img(img)

    att_m = compute_attn(f1, f2)

    feat = m_att(torch.cat((att_m, f2), dim=1))

    return feat, att_m

def compute_attn(patch_f, img_f):


    x = torch.conv2d(img_f, patch_f, padding=3)
    shape = x.shape

    x = x.view(shape[0], -1)
    attn = F.softmax(x, dim=1)
    attn = attn.view(shape)

    return attn





if __name__ == "__main__":
    train_ucl()