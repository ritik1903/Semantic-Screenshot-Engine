import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Page Config
st.set_page_config(page_title="AI Photo Search", layout="wide")
st.title("🔎 AI Photo Search")
st.markdown("Upload photos from your gallery and search them by **meaning**.")

# 2. Load Model (Cached)
@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')

# Download model immediately so we don't wait later
with st.spinner("Loading AI Model..."):
    model = load_model()

# 3. File Uploader (Mobile Friendly)
uploaded_files = st.file_uploader(
    "Tap to upload photos:", 
    accept_multiple_files=True, 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_files:
    # 4. Process Images
    images = []
    valid_files = []
    
    # Show a progress bar because phones might be slower
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            img = Image.open(file)
            images.append(img)
            valid_files.append(file)
        except:
            pass
        # Update progress bar
        progress_bar.progress((i + 1) / len(uploaded_files))
        
    if images:
        # Convert images to numbers (Embeddings)
        image_embeddings = model.encode(images, convert_to_tensor=True)
        st.success(f"Ready! Scanned {len(images)} photos.")

        # 5. Search Bar
        query = st.text_input("What are you looking for?", placeholder="e.g. 'cat', 'food', 'selfie'")
        
        if query:
            # Search Logic
            text_vec = model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(text_vec, image_embeddings)[0]
            
            # Get Top Matches
            top_results = torch.topk(cos_scores, k=min(5, len(images)))
            
            st.divider()
            for i, idx in enumerate(top_results.indices):
                score = top_results.values[i].item()
                if score > 0.15: # Filter out bad matches
                    st.image(valid_files[idx], caption=f"Match: {int(score*100)}%", use_container_width=True)