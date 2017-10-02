import torch
import os
from torch.autograd import Variable
from torch.utils import model_zoo
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
from lib.voc import Voc2007Classification
from lib.util import load_imagenet_classes
from lib.vggm import VGGM
models.__dict__['vggm'] = VGGM

model_urls = {
    # Alexnet
    # Paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    # VGGM
    'vggm': 'http://webia.lip6.fr/~cadene/Downloads/pretrained-models.pytorch/vggm-5bdc9182.pth',
    # VGG
    # Paper: https://arxiv.org/abs/1409.1556
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    # VGG BatchNorm
    # Paper: https://arxiv.org/abs/1502.03167
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    # Inception
    # Paper: https://arxiv.org/abs/1602.07261
    # https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    # Resnet
    # Paper: https://arxiv.org/abs/1512.03385
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

if __name__ == '__main__':
    
    print('Create network')
    model_name = 'alexnet'
    model = models.__dict__[model_name]()
    model.eval()
    print('')

    ##########################################################################

    print('Display modules')
    print(model)
    print('')

    ##########################################################################

    print('Display parameters')
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        print(key, value.size())
    print('')

    print('Display features.0.weight')
    print(state_dict['features.0.weight'])
    print('')

    ##########################################################################

    print('Display inputs/outputs')

    def print_info(self, input, output):
        print('Inside '+ self.__class__.__name__+ ' forward')
        print('input size', input[0].size())
        print('output size', output.data.size())
        print('')

    handles = []
    for m in model.features:
        handles.append(m.register_forward_hook(print_info))
        
    for m in model.classifier:
        handles.append(m.register_forward_hook(print_info))

    input = Variable(torch.randn(1,3,224,224).float(), requires_grad=False)
    output = model(input)

    for h in handles:
        h.remove()

    print('')

    ##########################################################################

    print('Load dataset Voc2007')

    train_data = Voc2007Classification('/tmp/torch/datasets', 'train')

    print('Voc2007 trainset has {} images'.format(len(train_data)))

    print('Voc2007 has {} classes'.format(len(train_data.classes)))
    print(train_data.classes)

    item = train_data[0]
    img_data = item[0] # PIL.Image.Image
    img_name = item[1]
    target = item[2]

    os.system('mkdir -p output')
    path_img = 'output/'+img_name+'.png'
    img_data.save(path_img)
    os.system('open '+path_img)

    print('Write image to '+path_img)
    for class_id, has_class in enumerate(target):
        if has_class == 1:
            print('image {} has object of class {}'.format(img_name, train_data.classes[class_id]))

    ##########################################################################

    print('Load pretrained model on Imagenet')

    model.load_state_dict(model_zoo.load_url(model_urls['alexnet'],
                                   model_dir='/tmp/torch/models'))

    print('Display predictions')

    class ToSpaceBGR(object):

        def __init__(self, is_bgr):
            self.is_bgr = is_bgr

        def __call__(self, tensor):
            if self.is_bgr:
                new_tensor = tensor.clone()
                new_tensor[0] = tensor[2]
                new_tensor[2] = tensor[0]
                tensor = new_tensor
            return tensor

    class ToRange255(object):

        def __init__(self, is_255):
            self.is_255 = is_255

        def __call__(self, tensor):
            if self.is_255:
                tensor.mul_(255)
            return tensor

    tf = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ToSpaceBGR(True),
        ToRange255(True),
        transforms.Normalize(
            mean=[123.68, 116.779, 103.939],#[0.485, 0.456, 0.406],
            std=[1,1,1],#[0.229, 0.224, 0.225]
        )
    ])

    input_data = tf(img_data).unsqueeze(0)
    print('input size', input_data.size())
    print(input_data)

    input = Variable(input_data, requires_grad=False)
    output = model(input)

    print('output size', output.data.size())
    print(output.data)

    # Load Imagenet Synsets

    imagenet_classes = load_imagenet_classes()
    print('Imagenet has {} classes'.format(imagenet_classes))

    max, argmax = output.data.squeeze().max(0)
    class_id = argmax[0]
    print('Image {} is of class "{}"'.format(img_name, imagenet_classes[class_id]))

    #############################################################################

    print('Save normalized input as RGB image')

    os.system('mkdir -p output/activation')

    path_img_input = 'output/activation/input.png'
    print('save input activation to '+path_img_input)
    transforms.ToPILImage()(input_data[0]).save(path_img_input)

    print('')

    #############################################################################

    print('Save activations as Gray image')

    layer_id = 0

    def save_activation(self, input, output):
        global layer_id

        for i in range(10):#output.data.size(1)):
            path_img_output = 'output/activation/layer{}_{}_channel{}.png'.format(layer_id, self.__class__.__name__, i)
            print('save output activation to '+path_img_output)
            torchvision.utils.save_image(output.data.squeeze(0)[i], path_img_output)

        layer_id += 1

    handles = []
    for m in model.features:
        handles.append(m.register_forward_hook(save_activation))

    input = Variable(input_data, requires_grad=False)
    output = model(input)

    for h in handles:
        h.remove()

    print('')

    #############################################################################

    print('Save first layer parameters as RGB image')

    os.system('mkdir -p output/parameters')

    state_dict = model.state_dict()

    weight = state_dict['features.0.weight']
    for filter_id in range(weight.size(0)):
        path_param = 'output/parameters/features.0.weight_filter{}.png'.format(filter_id)
        print('save '+path_param)
        torchvision.utils.save_image(weight[filter_id], path_param)

    for key in state_dict:
        if 'features' in key and 'weight' in key:
            for filter_id in range(3):
                for channel_id in range(3):
                    path_param = 'output/parameters/{}_filter{}_channel{}.png'.format(key, filter_id, channel_id)
                    torchvision.utils.save_image(state_dict[key][filter_id][channel_id], path_param)

    print('')

