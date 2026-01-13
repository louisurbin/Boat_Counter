### Changer le nom du fichier en fonction de l'architecture utilisée ###

import torch
import torch.nn as nn
from torchvision import models

num_classes = 6
model = None

# dictionary mapping class indices to boat type names
boat_classes = {
    0: 'marchandise',
    1: 'administration',
    2: 'passager_gros',
    3: 'passager_petit',
    4: 'plaisance',
    5: 'restaurant'
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    """
    Will be called once at the start.
    Initializes the ResNet model and loads trained weights.
    """
    global model

    model = models.resnet18(weights=None)
    
    num_features = model.fc.in_features # new head 
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, num_classes)
    )
    
    model_weights = torch.load('./models/boat_recognition_model.pth', map_location=DEVICE) 
    model.load_state_dict(model_weights, strict=True) # put the good weights in the model

    model.to(DEVICE) 
    model.eval() 


def model_prediction(input_images):
    """
    Run inference on a batch of images.
    Returns probabilities for each class.
    """
    global model
    if model is None:
        initialize_model()

    logits = model(input_images) # raw outputs (logits)
    probabilities = torch.nn.functional.softmax(logits, dim=1) # get the probabilities for each class

    return probabilities # output shape: (batch_size, num_classes)


def get_boat_type_name(boat_type_int):
    """
    Return the boat type name associated with a class index.
    """
    return boat_classes.get(boat_type_int, "Unknown")


