from dataloaders.hand_tracking_dataset import ImgSeqDataset, ToTensor, RandomCrop, Rescale, collate_fn
from models.img_feat_extractor import img_feat_extractor
from models.patch_feat_extractor import patch_feat_extractor
from models.att_feat_extractor import att_feat_extractor

import matplotlib.pyplot as plt
import numpy as np

import torch
import random
from torchvision import transforms
import torch.nn.functional as F

def train_ucl():

    display = False

    experiment_name = "test1"
    epochs = 100
    batch_size = 5
    random_episodes = 10

    n_feat = 256
    img_size = [3, 240,240]
    patch_size = [50, 50]
    train_new = False
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
    model_patch = model_img #patch_feat_extractor(n_feat, weights=weights_patch, train_new=train_new)
    model_attention = att_feat_extractor(2, weights=weights_attenton, train_new=train_new)

    model_img = model_img.to(device)
    model_patch = model_patch.to(device)
    model_attention = model_attention.to(device)

    # set up optimizer
    optimizer_img = torch.optim.Adam(model_img.parameters(), lr=lr)
    optimizer_patch = torch.optim.Adam(model_patch.parameters(), lr=lr)
    optimizer_attention = torch.optim.Adam(model_attention.parameters(), lr=lr)

    dataloader = ImgSeqDataset('D:\\Anton\\data\\hand_gesture_sequences\\train', None, batch_size, transform=transforms_train)

    train_loader = torch.utils.data.DataLoader(dataloader, batch_size=1, shuffle=True, collate_fn=collate_fn)
    for e in range(epochs):

        for processed in train_loader:
            # processed = dataloader[i]
            # processed = processed[:,:,:,:]
            imgs, landmark, label =processed

            imgs = imgs.view(imgs.shape[1:]).to(device)
            # landmark = landmark.view(imgs.shape[1:])
            # label = label.view(imgs.shape[1:])

            # imgs = imgs.to(device)

            for j in range(random_episodes):

                if j == 0:
                    loss = random_episode(model_img, model_patch, model_attention, imgs, patch_size, device, display=display)
                else:
                    loss = random_episode(model_img, model_patch, model_attention, imgs, patch_size, device, display=False)

                loss.backward()
                optimizer_img.step()
                optimizer_img.zero_grad()

                # optimizer_patch.step()
                # optimizer_patch.zero_grad()

                optimizer_attention.step()
                optimizer_attention.zero_grad()

                print("({}/{}) loss: {:.4f}".format(j + 1, random_episodes, float(loss)))

            model_patch.save_model()
            model_img.save_model()
            model_attention.save_model()


def random_episode(m_img, m_patch, m_att, imgs, patch_size, device, display = False):

    if display:
        fig = plt.figure(1, figsize=(16, 4))  # (figsize=(10,5))
        fig.subplots_adjust(left=0.05, right=0.95)

        ax2 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

    img_size = imgs[0].shape

    # extracting random patch location
    h, w = img_size[1:]
    new_h, new_w = patch_size
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)

    loc_initial = torch.ones([1, 2], device=device)
    loc_initial[:,0] *= (top/h * 2 - 1)
    loc_initial[:,1] *= (left/w * 2 - 1)

    # creating mask for patch
    mask = torch.zeros(img_size, device=device)
    mask = imgs[-1][:, top: top + new_h, left: left + new_w]

    if display:
        pred_img = np.asarray(imgs[-1].data.to('cpu')).transpose((1, 2, 0))
        sample_img_ = np.asarray(mask.data.to('cpu')).transpose((1, 2, 0))
        ax1.imshow(pred_img[:, :, :3], animated=True)
        ax2.imshow(sample_img_[:, :, :3], animated=True)
        plt.show(block=False)
        plt.pause(1 / 20)

    # make sure tracking works on this image
    _first_img = imgs[-1][None, :, :, :]
    _first_loc, _first_ind, _ = transition(m_img, m_patch, m_att, mask[None,:,:,:], _first_img)
    _first_mask = extract_mask(_first_loc, _first_img, patch_size, device)

    if display:

        try:
            sample_img_ = np.asarray(_first_mask[0].data.to('cpu')).transpose((1, 2, 0))
            ax3.imshow(sample_img_[:, :, :3], animated=True)
            plt.show(block=False)
            plt.pause(1 / 20)
        except:
            pass



    # set up vars for backwards pass
    img_seq = imgs[:-1]
    img_prev = imgs[-1][None,:,:,:]
    cur_mask = mask[None,:,:,:]
    feats_back = []
    loc_back = [loc_initial]

    for i, img in enumerate(reversed(img_seq)):
        img = img[None,:,:,:]
        loc, ind, feat = transition(m_img, m_patch, m_att, cur_mask, img)
        # img_prev = img
        cur_mask = extract_mask(loc, img, patch_size, device)
        feats_back.append(feat)
        loc_back.append(loc)

        if display:
            pred_img = np.asarray(img[0].data.to('cpu')).transpose((1, 2, 0))
            sample_img_ = np.asarray(cur_mask[0].data.to('cpu')).transpose((1, 2, 0))
            ax1.imshow(pred_img[:, :, :3], animated=True)
            ax4.imshow(sample_img_[:, :, :3], animated=True)
            plt.show(block=False)
            plt.pause(1 / 20)

    _, ind, feat = transition(m_img, m_patch, m_att, cur_mask, img_prev)
    feats_back.append(feat)


    # set up vars for backwards pass
    img_seq = imgs[1:]
    feats_forward = []
    loc_forward = [loc]

    for i, img in enumerate(img_seq):
        img = img[None, :, :, :]
        loc, ind, feat = transition(m_img, m_patch, m_att, cur_mask, img)
        # img_prev = img
        cur_mask = extract_mask(loc, img, patch_size, device)
        feats_forward.append(feat)
        loc_forward.append(loc)

        if display:
            pred_img = np.asarray(img[0].data.to('cpu')).transpose((1, 2, 0))
            sample_img_ = np.asarray(cur_mask[0].data.to('cpu')).transpose((1, 2, 0))
            ax1.imshow(pred_img[:, :, :3], animated=True)
            ax4.imshow(sample_img_[:, :, :3], animated=True)
            plt.show(block=False)
            plt.pause(1 / 20)

    _, ind, feat = transition(m_img, m_patch, m_att, cur_mask, img_prev)
    feats_back.append(feat)



    loc_info = (loc_back, loc_forward)
    feat_info = (feats_back, feats_forward)

    loss = compute_loss((_first_loc, _first_ind), feat_info, loc_info)
    return loss


def extract_mask(loc, img, patch_size, device):

    dimx, dimy = img[0].shape[1:]

    top = torch.round((loc[:, 0] + 1)/2 * dimx).int()
    left = torch.round((loc[:, 1] + 1)/2 * dimy).int()


    new_h, new_w = patch_size

    mask = torch.zeros([img.shape[0], img.shape[1]] + patch_size, device=device)
    info = img[:, :, top: top + new_h, left: left + new_w]

    mask[:,:,:info.shape[2],:info.shape[3]] = info

    return mask




def compute_loss(first_loc_pred, feat_info, loc_info):
    l_back, l_forward = loc_info

    l_first = l_back[0]
    l_last = l_forward[-1]

    first_loc, first_ind = first_loc_pred

    f_back, f_forward = feat_info

    f_first = f_back[0]
    f_last = f_forward[-1]

    # criterion_frobenius = loss_criteria()
    criterion_l2 = torch.nn.MSELoss()

    # long_loss = criterion_l2(f_last, f_first) + criterion_l2(a_last, a_first)

    main_loss = 0


    for f2, f1 in zip(f_forward, reversed(f_back)):
        main_loss += criterion_l2(f2, f_first) * 0.1

    # for l2, l1 in zip(l_forward, reversed(l_back)):
    #     main_loss += criterion_l2(l2, l1)

    main_loss += criterion_l2(l_last, l_first)
    main_loss += criterion_l2(first_loc, l_first) * 0.5
    # main_loss += criterion_l2(first_loc, first_ind)

    loss = main_loss #+ 0.1 * long_loss

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











def transition(m_img, m_patch, m_att, mask, img2):

    img1 = mask

    f1 = m_patch(img1)
    f2 = m_img(img2)

    att_m = compute_attn(f1, f2)

    try:
        x = m_att(att_m)
    except:
        print(att_m.shape)
        print(f1.shape)
        print(f2.shape)
        exit(1)

    m = torch.argmax(F.softmax(att_m.view(1,-1), dim=1))
    indices = torch.cat(((m / 30).view(-1, 1), (m % 30).view(-1, 1)), dim=1)

    indices = indices.float()
    indices[:,0] = indices[:,0]/30 * 2 - 1
    indices[:,1] = indices[:,1]/30 * 2 - 1

    # m_att(torch.cat((att_m, f2), dim=1))

    #plt.imshow(np.asarray(att_m[0,0].data.to('cpu')))
    #plt.imshow(np.asarray(F.softmax(att_m[0,0].view(1,-1)).view(30,30).data.to('cpu')))
    #plt.imshow(np.asarray(img2[0].data.to('cpu')).transpose(1,2,0))
    return x, indices, f1

# def transition_feat(m_img, m_att, img, f1):
#
#     f2 = m_img(img)
#
#     att_m = compute_attn(f1, f2)
#
#     x = m_att(att_m)  # m_att(torch.cat((att_m, f2), dim=1))
#
#     return x

def compute_attn(patch_f, img_f):


    x = torch.conv2d(img_f, patch_f, padding=3)
    # shape = x.shape
    #
    # x = x.view(shape[0], -1)
    # attn = F.softmax(x, dim=1)
    # attn = attn.view(shape)

    return x





if __name__ == "__main__":
    train_ucl()