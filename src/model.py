import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
 
def build_model(num_labels=14):
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
    return model  # apply sigmoid at inference/loss time, not inside the model
