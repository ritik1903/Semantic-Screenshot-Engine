import os
import glob
import pickle
from PIL import Image
from sentence_transformers import SentenceTransformer

# 1. Load the OpenAI CLIP model (this downloads it the first time)
print("Loading CLIP Model... (this might take a minute)")
model = SentenceTransformer('clip-ViT-B-32')

# 2. Setup paths
IMAGE_FOLDER = 'photos'  # The folder where your images are
INDEX_FILE = 'image_embeddings.pkl' # Where we save the "math"

def index_images():
    # Find all images (png, jpg, jpeg)
    image_paths = glob.glob(os.path.join(IMAGE_FOLDER, '*.*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER}!")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    # 3. Generate Embeddings
    # We load images and ask the model to convert them to vectors
    # This might take a few seconds depending on your CPU/GPU
    images = [Image.open(filepath) for filepath in image_paths]
    embeddings = model.encode(images, convert_to_tensor=True)

    # 4. Save the data
    # We save both the file paths and the vectors so we can match them later
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({'paths': image_paths, 'embeddings': embeddings}, f)
    
    print(f"Success! Indexed {len(image_paths)} images to {INDEX_FILE}")

if __name__ == "__main__":
    index_images()