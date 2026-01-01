import os
import numpy as np
import random
from torchvision import transforms
from PIL import Image
from ResNet import model_prediction as boat_recognition_model, get_boat_type_name

###
# It's possible to change the number of images evaluated per ID in process_id_folder function. 5 is a good balance between speed and accuracy. 
###

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

    image_files = [f for f in os.listdir(id_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files) # Shuffle the list of image files

    num_samples = min(5, len(image_files)) # Ensure we do not exceed the number of available images
    selected_images = image_files[:num_samples] # Select the first num_eval images after shuffling

    for image_file in selected_images:
        image_path = os.path.join(id_folder_path, image_file)
        prob = get_prob(image_path)
        prob_table.append(prob.detach().numpy()) # convert tensor to numpy array

    prob_table = np.array(prob_table)
    prob_table = prob_table.reshape(-1, 6)
    #print("prob_table:", prob_table)
    class_sums = np.sum(prob_table, axis=0) # For each class, sum probabilities across all images
    predicted_class = np.argmax(class_sums) # Class with highest total probability
    mean_prob = class_sums / num_samples # Calculate the mean probabilities
    
    return predicted_class, mean_prob


def add2crossings(id_folder, id_predicted_class, mean_prob):
    txt_file_path = os.path.join(id_folder, 'crossings.txt')
    boat_type_name = get_boat_type_name(id_predicted_class)
    with open(txt_file_path, 'a') as f:
        f.write(f"{boat_type_name} {mean_prob[id_predicted_class]:.4f}\n") # Append the boat type name and mean probability associated with the ID
    

def main():

    ids_root = './temp/extractions/'

    for folder in os.listdir(ids_root):
        folder_path = os.path.join(ids_root, folder)
        
        if os.path.isdir(folder_path):
            predicted_class, mean_prob = process_id_folder(folder_path)
            add2crossings(folder_path, predicted_class, mean_prob)

if __name__ == "__main__":
    main()

