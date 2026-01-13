import os
import random
import torch
from PIL import Image
from torchvision import transforms
from ResNet_boat_type import model_prediction, get_boat_type_name
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

# Paramètres
MAX_IMAGES_PER_ID = 5     # It's possible to change the number of images evaluated per ID in process_id_folder function. 5 is a good balance between speed and accuracy.
MIN_PROB_THRESHOLD = 0.5

DEVICE = torch.device("cpu")  # CPU only

# Use half of CPU cores for processes
PROCESS_WORKERS = max(1, cpu_count() // 2)
# Limit threads per process to avoid oversubscription
THREAD_WORKERS = min(4, PROCESS_WORKERS)

# Transformations applied to each image before inference
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # dimensions expected by the boat recognition model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # statistics of ImageNet dataset
                         std=[0.229, 0.224, 0.225])
])


def load_single_image(image_path):
    """
    Load and preprocess a single image.
    """
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def load_images(id_folder_path, max_images):
    """
    Load and preprocess images from an ID folder using multi-threading.
    Randomly selects up to max_images images.
    """
    image_files = [
        f for f in os.listdir(id_folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Ensure there are images to process
    if len(image_files) == 0:
        return None

    num_samples = min(max_images, len(image_files))
    selected_images = random.sample(image_files, num_samples)

    image_paths = [
        os.path.join(id_folder_path, f) for f in selected_images
    ]

    # Multi-threaded image loading (I/O + preprocessing)
    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
        images = list(executor.map(load_single_image, image_paths))

    return torch.stack(images)


@torch.no_grad()  # Disable gradient computation for inference
def process_id_folder(id_folder_path):
    """
    Process all (or a subset of) images inside an ID folder and
    determine the most probable boat class.
    """
    # Load and preprocess images
    batch = load_images(id_folder_path, MAX_IMAGES_PER_ID)
    if batch is None:
        return None, None  # no images in this folder

    # Move batch to CPU device
    batch = batch.to(DEVICE)

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


def process_one_folder(folder_path):
    """
    Wrapper function for processing a single folder.
    Used in multiprocessing.
    """
    predicted_class, mean_prob = process_id_folder(folder_path)
    if predicted_class is not None:
        add2crossings(folder_path, predicted_class, mean_prob, MIN_PROB_THRESHOLD)


def main():
    ids_root = './temp/extractions/'

    # List all ID folders
    folder_paths = [
        os.path.join(ids_root, f)
        for f in os.listdir(ids_root)
        if os.path.isdir(os.path.join(ids_root, f))
    ]

    # Multiprocessing across folders
    with Pool(PROCESS_WORKERS) as pool:
        pool.map(process_one_folder, folder_paths)


if __name__ == "__main__":
    main()
