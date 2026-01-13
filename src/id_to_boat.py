import os
import random
import torch
from PIL import Image
from torchvision import transforms
from ResNet_boat_type import model_prediction, get_boat_type_name

# Paramètres
MAX_IMAGES_PER_ID = 5     # It's possible to change the number of images evaluated per ID in process_id_folder function. 5 is a good balance between speed and accuracy.
MIN_PROB_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations applied to each image before inference
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # dimensions expected by the boat recognition model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # statistics of ImageNet dataset
                         std=[0.229, 0.224, 0.225])
])


def load_images(id_folder_path, max_images):
    """
    Load and preprocess images from an ID folder.
    Randomly selects up to max_images images.
    """
    image_files = [
        f for f in os.listdir(id_folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Shuffle / sample images to avoid always taking the same ones
    num_samples = min(max_images, len(image_files))
    selected_images = random.sample(image_files, num_samples)

    images = []
    for image_file in selected_images:
        image_path = os.path.join(id_folder_path, image_file)

        # Open image and ensure RGB format
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing transforms
        images.append(transform(image))

    # Stack images into a single batch tensor
    return torch.stack(images).to(DEVICE)


@torch.no_grad()  # Disable gradient computation 
def process_id_folder(id_folder_path):
    """
    Process all (or a subset of) images inside an ID folder and
    determine the most probable boat class.
    """
    # Load and preprocess images
    batch = load_images(id_folder_path, MAX_IMAGES_PER_ID)

    # Get probabilities from the model for each image
    # Output shape: (num_images, num_classes)
    probabilities = model_prediction(batch)

    # Compute mean probability for each class across images
    mean_prob = probabilities.mean(dim=0)

    # Class with highest mean probability
    predicted_class = mean_prob.argmax().item()

    return predicted_class, mean_prob


def add2crossings(id_folder, id_predicted_class, mean_prob, min_prob_threshold):
    """
    Append classification result to crossings.txt file.
    """
    txt_file_path = os.path.join(id_folder, 'crossings.txt')

    with open(txt_file_path, 'a') as f:
        # If the mean probability is below the threshold, classify as noise
        if mean_prob[id_predicted_class] < min_prob_threshold:
            f.write("noise\n")
        else:
            boat_type_name = get_boat_type_name(id_predicted_class)
            f.write(f"{boat_type_name} {mean_prob[id_predicted_class]:.4f}\n")


def main():
    ids_root = './temp/extractions/'

    # Iterate through each ID folder
    for folder in os.listdir(ids_root):
        folder_path = os.path.join(ids_root, folder)

        if os.path.isdir(folder_path):
            predicted_class, mean_prob = process_id_folder(folder_path)
            add2crossings(folder_path, predicted_class, mean_prob, MIN_PROB_THRESHOLD)


if __name__ == "__main__":
    main()

