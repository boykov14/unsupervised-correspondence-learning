import torch
import torch.nn as nn
import torchvision.models as models

class att_feat_extractor(nn.Module):
    def __init__(self, feat, weights = None, train_new = True):
        super(att_feat_extractor, self).__init__()

        image_modules = [
            nn.Conv2d(feat + 1, 200, kernel_size=5, stride=1),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size=7, stride=1),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(200, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 200, kernel_size=4, stride=1),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(200, feat, kernel_size=4, stride=1),
            nn.BatchNorm2d(feat, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
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

        return self.model(input)

#
# A = att_feat_extractor(256)
#
# # x = torch.ones([5, 3, 240, 240], device='cpu')
# y = torch.ones([5, 257, 30, 30], device='cpu')
#
# # print(A(x).shape)
# print(A(y).shape)


# x = torch.ones(A(x).shape, device='cpu')
# y = torch.ones(A(y).shape, device='cpu')
#
# print(torch.conv2d(x, y, padding = 3).shape)
