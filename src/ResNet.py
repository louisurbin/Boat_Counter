import torch
from torchvision import models

num_classes = 6
boat_classes = None
model = None

def initialize_model(): # will be called once at the start
    global model, boat_classes
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes) # modify the head for our number of classes

    model_weights = torch.load('./models/boat_recognition_model.pth')
    model.load_state_dict(model_weights, strict=False) # put the good weights in the model
    model.eval()

    boat_classes = {
        0: 'administration',
        1: 'marchandise',
        2: 'passager_gros',
        3: 'passager_petit',
        4: 'plaisance',
        5: 'restaurant'
    }

initialize_model() 


def model_prediction(input_image):
    if model is None: # if the model is not initialized yet
        initialize_model()
    probabilities = model(input_image) # get the probabilities for each class, its a tensor of shape (1, num_classes)
    return probabilities


def get_boat_type_name(boat_type_int):
    if boat_classes is None:
        initialize_model()
    return boat_classes.get(boat_type_int, "Unknown")

