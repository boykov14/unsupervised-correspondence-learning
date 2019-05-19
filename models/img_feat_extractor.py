import torch
import torch.nn as nn
import torchvision.models as models

class img_feat_extractor(nn.Module):
    def __init__(self, feat, weights = None, train_new = True):
        super(img_feat_extractor, self).__init__()

        image_modules = list(models.resnet50(pretrained=True).children())[:6]
        image_modules += [
            nn.Conv2d(512, feat, kernel_size=3, stride=1, padding=1),
        ]
        self.resnet = nn.Sequential(*image_modules)

        if weights is not None and not train_new:
            self.load_state_dict(torch.load(weights))

        self.weights = weights

    def save_model(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)
        elif self.weights is not None:
            torch.save(self.state_dict(), self.weights)

    def forward(self, input):

        return self.resnet(input)


# A = img_feat_extractor(256)
#
# x = torch.ones([5, 3, 240, 240], device='cpu')
# y = torch.ones([5, 3, 50, 50], device='cpu')
#
# print(A(x).shape)
# print(A(y).shape)
# #
#
# x = torch.ones(A(x).shape, device='cpu')
# y = torch.ones(A(y).shape, device='cpu')
#
# print(torch.conv2d(x, y, padding = 3).shape)
