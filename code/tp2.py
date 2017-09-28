import os
import torch
from torch.autograd import Variable
from torch.utils import model_zoo

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from tqdm import tqdm

from lib.voc import Voc2007Classification
from lib.util import load_imagenet_classes

model_urls = {
    # Alexnet
    # Paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
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

print('Create network')
model = models.alexnet()
print('')

print('Load pretrained model on Imagenet')
model.load_state_dict(model_zoo.load_url(model_urls['alexnet'],
                               model_dir='/tmp/torch/models'))

tf = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
print('')

def extract_features_targets(split, batch_size, path_data, features_size=4096):
    if os.path.isfile(path_data):
        print('Load features from {}'.format(path_data))
        return torch.load(path_data)

    print('Extract features on {}set'.format(split))

    data = Voc2007Classification('/tmp/torch/datasets', split, transform=tf)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)

    features = torch.Tensor(len(data), features_size)
    targets = torch.Tensor(len(data), len(data.classes))

    def get_features(self, input, output):
        nonlocal features, from_, to_ # https://stackoverflow.com/questions/11987358/why-nested-functions-can-access-variables-from-outer-functions-but-are-not-allo
        features[from_:to_] = output.data

    handle = model.classifier[4].register_forward_hook(get_features)

    for batch_id, batch in enumerate(tqdm(loader)):
        img = batch[0]
        target = batch[2]
        current_bsize = img.size(0)
        from_ = batch_id*batch_size
        to_ = from_ + current_bsize
        targets[from_:to_] = target
        input = Variable(img, requires_grad=False)
        model(input)

    handle.remove()

    os.system('mkdir -p {}'.format(os.path.dirname(path_data)))
    print('save '+path_data)
    torch.save((features, targets), path_data)
    print('')
    return features, targets

features_size = 4096
dir_data = '/tmp/output/data'
path_train_data = '{}/{}set.pth'.format(dir_data, 'train')
path_val_data = '{}/{}set.pth'.format(dir_data, 'val')
path_test_data = '{}/{}set.pth'.format(dir_data, 'test')

features = {}
targets = {}
features['train'], targets['train'] = extract_features_targets('train', 256, path_train_data, features_size=features_size)
features['val'], targets['val'] = extract_features_targets('val', 256, path_val_data, features_size=features_size)
features['test'], targets['test'] = extract_features_targets('test', 256, path_test_data, features_size=features_size)
features['trainval'] = torch.cat([features['train'], features['val']], 0)
targets['trainval'] = torch.cat([targets['train'], targets['val']], 0)

print('')

##########################################################################

def train_multilabel(features, targets, split_train, split_test, C=1.0):
    APs = {}
    APs[split_train] = []
    APs[split_test] = []
    for class_id in range(nb_classes):
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        classifier = SVC(C=C, kernel='linear')
        
        #train_X = torch.masked_select(train_features, (train_targets[:,class_id] == 1).view(-1,1).expand_as(train_features)).view(-1,features_size).numpy()
        train_X = features[split_train].numpy()
        train_y = (targets[split_train][:,class_id] != -1).numpy() # uses hard examples

        test_X = features[split_test].numpy()
        test_y = (targets[split_test][:,class_id] != -1).numpy()

        classifier.fit(train_X, train_y)

        train_preds = classifier.predict(train_X)
        train_acc = accuracy_score(train_y, train_preds) * 100
        train_AP = average_precision_score(train_y, train_preds) * 100
        APs.append(train_AP)

        test_preds = classifier.predict(test_X)
        test_acc = accuracy_score(test_y, test_preds) * 100
        test_AP = average_precision_score(test_y, test_preds) * 100
        test_APs.append(test_AP)

        print('class "{}" ({}/{}):'.format(dataset.classes[class_id], test_y.sum(), test_y.shape[0]))
        print('  - {:7}: acc {:.2f}, AP {:.2f}'.format(split_train, train_acc, train_AP))
        print('  - {:7}: acc {:.2f}, AP {:.2f}'.format(split_test, test_acc, test_AP))

    print('all classes:')
    print('  - {:7}: mAP {:.4f}'.format(split_train, sum(APs)/nb_classes))
    print('  - {:7}: mAP {:.4f}'.format(split_test, sum(test_APs)/nb_classes))
    print('')

print('Hyperparameters search: train multilabel classifiers (on-versus-all) on train/val')

dataset = Voc2007Classification('/tmp/torch/datasets', 'train')
nb_classes = len(dataset.classes) # Voc2007
C = 1.0

train_multilabel(features, targets, 'train', 'val', C=C)

##########################################################################

print('Evaluation: train a multilabel classifier on trainval/test')

train_multilabel(features, targets, 'trainval', 'test', C=C)
