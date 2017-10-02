
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.legacy import nn as nnl

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class VGGM(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3,96,(7, 7),(2, 2)),
            nn.ReLU(),
            Lambda(lambda x,lrn=nnl.SpatialCrossMapLRN(*(5, 0.0005, 0.75, 2)): Variable(lrn.forward(x.data))),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(96,256,(5, 5),(2, 2),(1, 1)),
            nn.ReLU(),
            Lambda(lambda x,lrn=nnl.SpatialCrossMapLRN(*(5, 0.0005, 0.75, 2)): Variable(lrn.forward(x.data))),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(18432,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':

    model = VGGM()
    model.eval()

    state_dict = torch.load('vggm.pth')
    model.load_state_dict(state_dict)

    #import ipdb; ipdb.set_trace()