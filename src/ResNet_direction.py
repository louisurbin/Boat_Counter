### Changer le nom du fichier en fonction de l'architecture utilisée ###

import torch
import torch.nn as nn
from id_to_boat import load_images

# Constante
MAX_IMAGES_PER_ID = 5  # Nombre maximum d'images à utiliser par ID pour la prédiction de la direction

model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model():
    '''
    Initialize the model for direction prediction.
    This function should load the model architecture and weights.
    '''
    global model

    model = nn.Sequential(
        # Convolutional layers
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Flatten the output for the fully connected layers
        nn.Flatten(),

        # Fully connected layers
        nn.Linear(128 * 32 * 32, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2),
        nn.Softmax(dim=1)
    )
    model_weights = torch.load('./models/boat_direction_model.pth', map_location=DEVICE) 
    model.load_state_dict(model_weights, strict=True) # put the good weights in the model

    model.to(DEVICE)
    model.eval()

def get_direction(id_dir):
    '''
    Predict the direction of the boat using all the images in the id_dir folder.
    Sum the probabilities for each class on each image and return the class with the highest total sum probability.
    '''
    global model
    if model is None:
        initialize_model()

    images = load_images(id_dir, MAX_IMAGES_PER_ID)  # Load all images in the folder (normalization is done in load_images)

    total_probabilities = {0: 0.0, 1: 0.0} # 0: avant, 1: arrière

    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    for prob in probabilities:
        total_probabilities[0] += prob[0]  
        total_probabilities[1] += prob[1]  

    # Return the class with the highest total sum probability
    if total_probabilities[1] > total_probabilities[0]:
        return 1  # arrière
    else:
        return -1  # avant
        
