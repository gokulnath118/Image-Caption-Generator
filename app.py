import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os
import json
import numpy as np
import faiss

# Function to load the image captioning model and components
@st.cache_resource
def load_image_captioning_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device

# Function to load the CLIP model and processor
@st.cache_resource
def load_clip_model():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor, device

# Function to predict the caption of an image
def predict_caption(model, feature_extractor, tokenizer, device, image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    gen_kwargs = {"max_length": 16, "num_beams": 4}
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()

# Function to get text embeddings
def get_text_embedding(processor, model, device, text):
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.cpu().numpy()

# Function to get image embeddings
def get_image_embedding(processor, model, device, image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy()

# Function to search for similar images
def search_similar_images(index, images, processor, model, device, query, k=1):
    query_embedding = get_text_embedding(processor, model, device, query)
    distances, indices = index.search(query_embedding, k)
    return [(images[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

# Load models
model, feature_extractor, tokenizer, device = load_image_captioning_model()
clip_model, clip_processor, clip_device = load_clip_model()

# Define paths for storage
image_dir = 'stored_images'
caption_file = 'captions.json'

# Create storage directories if not exist
os.makedirs(image_dir, exist_ok=True)

# Load existing captions if any
if os.path.exists(caption_file):
    with open(caption_file, 'r') as f:
        img_caption = json.load(f)
else:
    img_caption = {}

# Streamlit UI
st.title("Image Captioning and Search")
uploaded_files = st.file_uploader("Upload a Folder of Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write("### Uploaded Images and Captions")
    for uploaded_file in uploaded_files:
        image_path = os.path.join(image_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict and store the caption
        caption = predict_caption(model, feature_extractor, tokenizer, device, image_path)
        img_caption[image_path] = caption

    # Save captions to file
    with open(caption_file, 'w') as f:
        json.dump(img_caption, f)

    # Display images and captions
    for image_path, caption in img_caption.items():
        st.image(image_path, caption=os.path.basename(image_path))
        st.write(f"**Caption:** {caption}")

    # Generate and store embeddings for captions
    captions = list(img_caption.values())
    images = list(img_caption.keys())
    caption_embeddings = np.vstack([get_text_embedding(clip_processor, clip_model, clip_device, caption) for caption in captions])

    # Prepare FAISS index
    dimension = caption_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(caption_embeddings)

    st.write("## Search for Similar Images")
    query = st.text_input("Enter a caption to search for similar images:")
    if query:
        similar_images = search_similar_images(index, images, clip_processor, clip_model, clip_device, query, k=3)
        st.write(f"### Results for '{query}'")
        for image_path, distance in similar_images:
            st.write(f"**Distance:** {distance:.4f}")
            st.image(image_path)

# Save the final captions dictionary to file
with open(caption_file, 'w') as f:
    json.dump(img_caption, f)
