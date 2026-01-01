import streamlit as st
import os
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Firstly Setting up the Page
st.set_page_config(page_title="Local AI Search", layout="wide")
st.title("🖥️ Local Disk Search")
st.markdown("Scan entire folders on your computer instantly without uploading.")

# 2. Here i am Loading the  Model
@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')

model = load_model()

# 3. About the Session State
if 'image_embeddings' not in st.session_state:
    st.session_state.image_embeddings = None
if 'image_files' not in st.session_state:
    st.session_state.image_files = []
if 'last_scanned_folder' not in st.session_state:
    st.session_state.last_scanned_folder = ""

# 4. Input for Folder Path (Replaces the crashing Tkinter button)
folder_path = st.text_input(
    "📂 Paste your folder path here:", 
    placeholder="/Users/ritikg/Desktop/Photos",
    help="On Mac, you can drag a folder from Finder into the terminal to see its path, then copy-paste it here."
)

# Logic to trigger scanning only when the path changes or hasn't been scanned
if folder_path and folder_path != st.session_state.last_scanned_folder:
    if os.path.isdir(folder_path):
        st.session_state.last_scanned_folder = folder_path
        
        # Finding images
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        found_files = []
        for root_dir, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    found_files.append(os.path.join(root_dir, file))
        
        st.session_state.image_files = found_files
        
        # Reset embeddings since we have a new folder
        st.session_state.image_embeddings = None
        st.toast(f"Found {len(found_files)} images! Processing...", icon="⏳")
    else:
        st.error("❌ That path does not exist. Please check your spelling.")

# 5. Indexing (Process the images)
if st.session_state.image_files and st.session_state.image_embeddings is None:
    with st.spinner(f"AI is reading {len(st.session_state.image_files)} images..."):
        loaded_images = []
        valid_paths = []
        
        # Process images
        for p in st.session_state.image_files:
            try:
                img = Image.open(p)
                loaded_images.append(img)
                valid_paths.append(p)
            except:
                continue
        
        if loaded_images:
            embeddings = model.encode(loaded_images, convert_to_tensor=True)
            st.session_state.image_embeddings = embeddings
            st.session_state.image_files = valid_paths
            st.success("Indexing Complete! You can now search.")

# 6. Search Interface
if st.session_state.image_embeddings is not None:
    st.divider()
    query = st.text_input("🔍 Search this folder:", placeholder="e.g. 'receipts' or 'cats'")
    
    if query:
        text_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(text_embedding, st.session_state.image_embeddings)[0]
        
        # Get Top results
        k = min(10, len(st.session_state.image_files))
        top_results = torch.topk(cos_scores, k=k)
        
        st.subheader("Results")
        cols = st.columns(5)
        
        for i, idx in enumerate(top_results.indices):
            score = top_results.values[i].item()
            filepath = st.session_state.image_files[idx]
            
            # Show image
            col_idx = i % 5
            with cols[col_idx]:
                st.image(filepath, use_container_width=True)
                st.caption(f"{os.path.basename(filepath)} ({int(score*100)}%)")