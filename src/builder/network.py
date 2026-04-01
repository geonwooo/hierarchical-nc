import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone
import modules


class Network(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(Network, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_classes_1 = cfg.dataset.num_classes_1  # 0 means no hierarchy
        self.num_classes_2 = cfg.dataset.num_classes_2
        self.hierarchical = self.num_classes_1 > 0 and self.num_classes_2 > 0
        self.num_features = self.get_num_features()

        self.backbone = getattr(backbone, cfg.backbone.type)(cfg)
        self.pooling = getattr(modules, cfg.pooling.type)()
        self.reshape = getattr(modules, cfg.reshape.type)(cfg, num_features=self.num_features)

        if self.hierarchical:
            # No fine-level classifier; only one per hierarchy level.
            self.classifier_1 = self._get_classifier(self.num_classes_1)
            self.scaling_1 = getattr(modules, cfg.scaling.type)(self.num_classes_1)
            self.classifier_2 = self._get_classifier(self.num_classes_2)
            self.scaling_2 = getattr(modules, cfg.scaling.type)(self.num_classes_2)
        else:
            self.classifier = self._get_classifier(self.num_classes)
            self.scaling = getattr(modules, cfg.scaling.type)(self.num_classes)

    def forward(self, input, **kwargs):
        if 'feature_flag' in kwargs:
            return self.extract_feature(input)
        elif 'classifier_flag' in kwargs:
            return self.classify(input)
        return self.classify(self.extract_feature(input))

    def extract_feature(self, input):
        x = self.backbone(input)
        x = self.pooling(x)
        x = self.reshape(x)
        return x

    def classify(self, input):
        if self.hierarchical:
            output_1 = self.classifier_1(input)          # (B, num_classes_1)
            with torch.no_grad():
                # Softmax-weighted combination of classifier_1 row vectors.
                # prob_1 : (B, num_classes_1)
                # classifier_1.weight : (num_classes_1, num_features)
                # input_2 : (B, num_features)
                prob_1 = F.softmax(output_1, dim=1)
                input_2 = prob_1 @ self.classifier_1.weight
            output_2 = self.classifier_2(input_2)        # (B, num_classes_2)
            return self.scaling_1(output_1), self.scaling_2(output_2)
        return self.scaling(self.classifier(input))

    def get_num_features(self):
        dic_num_features = {
            'SimpleFNN': 300,
            'LeNet5': 84,
            'resnetcifar32_z': 64,
        }
        if self.cfg.backbone.type in dic_num_features:
            num_features = dic_num_features[self.cfg.backbone.type]
        elif 'vgg' in self.cfg.backbone.type:
            num_features = 4096
        else:
            raise NotImplementedError(
                "Update dic_num_features for {}".format(self.cfg.backbone.type))
        return num_features

    def _get_classifier(self, num_classes):
        bias_flag = self.cfg.classifier.bias
        if self.cfg.classifier.type == 'FC':
            return nn.Linear(self.num_features, num_classes, bias=bias_flag)
        else:
            return getattr(modules, self.cfg.classifier.type)(
                self.num_features, num_classes, cfg=self.cfg)
