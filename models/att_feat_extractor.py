import torch
import torch.nn as nn
import torchvision.models as models

class att_feat_extractor(nn.Module):
    def __init__(self, feat, weights = None, train_new = True):
        super(att_feat_extractor, self).__init__()

        image_modules = [
            nn.Linear(30*30, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, feat),
            nn.Sigmoid(),
        ]
        self.model = nn.Sequential(*image_modules)

        if weights is not None and not train_new:
            self.load_state_dict(torch.load(weights))

        self.weights = weights

    def save_model(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)
        elif self.weights is not None:
            torch.save(self.state_dict(), self.weights)

    def forward(self, input):

        batch_size = input.size(0)
        x = input.view(batch_size, -1)

        return self.model(x)

#
# A = att_feat_extractor(3)
#
# # x = torch.ones([5, 3, 240, 240], device='cpu')
# y = torch.ones([5, 1, 30, 30], device='cpu')
#
# # print(A(x).shape)
# print(A(y).shape)


# x = torch.ones(A(x).shape, device='cpu')
# y = torch.ones(A(y).shape, device='cpu')
#
# print(torch.conv2d(x, y, padding = 3).shape)
