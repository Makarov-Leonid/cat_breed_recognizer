import json
import pickle
import sys
from typing import Tuple, List, Any, AnyStr

sys.path.append(r"Models/PyTorch-YOLOv3/")
from pytorchyolo import models as md, detect as dt

import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import models

device = "cpu"

with open("./animal_types.json") as jf:
    animal_types = json.load(jf)

with open("./dog_breeds.json") as jf:
    dog_breeds = json.load(jf)

with open("./cat_breeds.json") as jf:
    cat_breeds = json.load(jf)


def change_device(name: str):
    device = name
    return device


def model_prep(model):
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    return model


class DogsBreedModel(object):
    def __init__(self):
        super().__init__()
        POOLING = "avg"
        shape = (3, 224, 224)
        # self.xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
        # self.inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=POOLING)
        self.bottleneck = model_prep(torch.nn.Sequential(
            *(list(timm.create_model('resnet152', pretrained=True, num_classes=1000).children())[:-1])))
        self.logreg = pickle.load(open("Models/DogBreedLogReg.pkl", 'rb'))

    def predict(self, images: torch.Tensor) -> Tuple[List[AnyStr], List[AnyStr]]:
        X = []
        Value = np.zeros(images.shape[0])
        with torch.no_grad():
            for i, img in enumerate(images):
                x_bf = self.bottleneck(img.unsqueeze(0))
                # x_bf = self.xception_bottleneck.predict(img, batch_size=1, verbose=1)
                # i_bf = self.inception_bottleneck.predict(img, batch_size=1, verbose=1)
                x = x_bf
                tmp = self.logreg.predict_proba(x).reshape((-1,))
                indices = np.argsort(tmp)
                Value[i] = tmp[indices[-1]] / tmp[indices[-2]]
                X.append((tmp, indices[-3:]))

        using_id = np.argsort(Value)[-1]

        probas = X[using_id][0][X[using_id][1]]
        Breeds_ids = X[using_id][1]
        Breeds_str = [dog_breeds[str(int(breed_id))] for breed_id in Breeds_ids]
        return Breeds_str, probas


class CatBreedsModel(object):
    def __init__(self):
        super().__init__()
        # shape 3,224,224
        self.pre_model = torch.nn.Sequential(*(list(timm.create_model('resnet50',
                                                                      pretrained=True, num_classes=1000).children())[
                                               :-1]))
        self.log_reg = pickle.load(open("Models/Log_Reg_on_Resnet50_cats_features.sav", 'rb'))

    def predict(self, images: torch.Tensor) -> Tuple[List[AnyStr], List[AnyStr]]:
        X = []
        Value = np.zeros(len(images))
        with torch.no_grad():
            for i, img in enumerate(images):
                tmp = self.log_reg.predict_proba(self.pre_model(img.unsqueeze(0))).reshape(-1)
                indices = np.argsort(tmp)
                Value[i] = tmp[indices[-1]] / tmp[indices[-2]]
                X.append((tmp, indices[-3:]))

        using_id = np.argsort(Value)[-1]

        Probas = X[using_id][0][X[using_id][1]]
        Breeds_ids = X[using_id][1]
        print(Breeds_ids)
        Breeds_str = [cat_breeds[str(int(breed_id))] for breed_id in Breeds_ids]
        return Breeds_str, Probas


class AnimalTypeModel(object):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.to(device)
        self.model.load_state_dict(torch.load("./Models/ResNet18TL.pt", map_location=torch.device(device)))
        self.model.eval()

    def predict(self, images: torch.Tensor) -> str:
        ans = np.empty(images.shape[0], dtype=np.float32)
        with torch.no_grad():
            for i, img in enumerate(images):
                model_pred = self.model(img.unsqueeze(0))
                ans[i] = torch.max(model_pred, 1)[1]
        return str(int(np.mean(ans) >= 0.5))


class AnimalDetectionModel(object):
    def __init__(self):
        super().__init__()
        self.model = md.load_model(
            r"./Models/yolov3.cfg",
            r"./Models/yolov3.weights"
        )


    def detectBBOX(self, images: List) -> np.ndarray:
        bboxs = np.array([dt.detect_image(self.model, image) for image in images])
        bboxs = np.array([elem[(elem[:, -1] == 15.) | (elem[:, -1] == 16.)][:, :-2] for elem in bboxs])
        return bboxs
