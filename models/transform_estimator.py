import torch
import torch.nn as nn
import torchvision.models as models

class transform_estimator(nn.Module):
    def __init__(self, input_size, n_feat, weights = None, train_new = True):
        super(transform_estimator, self).__init__()

        modules = [
            torch.nn.Linear(input_size, 1024),
            torch.nn.Linear(1024, 512),
            torch.nn.Linear(512, 256),
            torch.nn.Linear(256, 128),
            torch.nn.Linear(128, n_feat),
        ]
        self.model = nn.Sequential(*modules)

        if weights is not None and not train_new:
            self.load_state_dict(torch.load(weights))

    def save_model(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)
        elif self.weights is not None:
            torch.save(self.state_dict(), self.weights)

    def forward(self, input):
        return self.model(input)

#
# A = transform_estimator()
#
# x = torch.ones([5, 3, 240, 240], device='cpu')
# y = torch.ones([5, 3, 80, 80], device='cpu')
#
# print(A(x).shape)
# print(A(y).shape)

