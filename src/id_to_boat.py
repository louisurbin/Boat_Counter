import os
import numpy as np
from torchvision import transforms
from PIL import Image
from ResNet import model_prediction as boat_recognition_model, get_boat_type_name


def get_prob(image_path):
    image = Image.open(image_path)

    target_size = (256, 256) # dimensions expected by the boat recognition model

    resize_transform = transforms.Resize(target_size)
    resized_image = resize_transform(image)

    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(resized_image)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalized_image = normalize(image_tensor)

    input_image = normalized_image.unsqueeze(0)  # Add batch dimension

    probabilities = boat_recognition_model(input_image)  # Get probabilities from the model

    #print("probabilities:", probabilities)

    return probabilities


def process_id_folder(id_folder_path):
    prob_table = [] # store the probabilities for each image in the folder

    for image_file in os.listdir(id_folder_path):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(id_folder_path, image_file)
        prob = get_prob(image_path)
        prob_table.append(prob.detach().numpy()) # convert tensor to numpy array

    prob_table = np.array(prob_table)
    prob_table = prob_table.reshape(-1, 6)
    #print("prob_table:", prob_table)
    class_sums = np.sum(prob_table, axis=0) # For each class, sum probabilities across all images
    predicted_class = np.argmax(class_sums) # Class with highest total probability
    
    return predicted_class


def add2crossings(id_folder, id_predicted_class):
    txt_file_path = os.path.join(id_folder, 'crossings.txt')
    boat_type_name = get_boat_type_name(id_predicted_class)
    with open(txt_file_path, 'a') as f:
        f.write(f"{boat_type_name}\n") # Append the boat type name associated with the ID
    

def main():

    ids_root = './temp/extractions/'

    for folder in os.listdir(ids_root):
        folder_path = os.path.join(ids_root, folder)
        
        if os.path.isdir(folder_path):
            predicted_class = process_id_folder(folder_path)
            add2crossings(folder_path, predicted_class)

if __name__ == "__main__":
    main()

