
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="Flower Detection App",
    page_icon="🌸",
    layout="wide"
)

# Title and description
st.title("🌸 Flower Detection App")
st.markdown("Upload an image to detect and classify flowers from 5 different classes")

# Define flower classes (modify these according to your model)
FLOWER_CLASSES = [
    "Daisy",
    "Dandelion", 
    "Rose",
    "Sunflower",
    "Tulip"
]

@st.cache_resource
def load_model():
    """Load the trained flower detection model"""
    try:
        # Replace with your model path
        model = tf.keras.models.load_model('my_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize image
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalize pixel values
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_flower(model, image):
    """Make prediction on preprocessed image"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get class probabilities
        class_probabilities = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(class_probabilities)
        predicted_class = FLOWER_CLASSES[predicted_class_idx]
        confidence = float(class_probabilities[predicted_class_idx])
        
        return predicted_class, confidence, class_probabilities
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def display_results(predicted_class, confidence, class_probabilities):
    """Display prediction results"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Results")
        st.success(f"**Predicted Flower:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2%}")
        
        # Confidence threshold
        if confidence > 0.8:
            st.success("High confidence prediction! ✅")
        elif confidence > 0.6:
            st.warning("Medium confidence prediction ⚠️")
        else:
            st.error("Low confidence prediction ❌")
    
    with col2:
        st.subheader("Class Probabilities")
        
        # Create probability chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        bars = ax.barh(FLOWER_CLASSES, class_probabilities, color=colors)
        
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Probabilities for All Classes')
        ax.set_xlim(0, 1)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, class_probabilities):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)

def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Please ensure your flower detection model is saved as 'flower_model.h5' in the same directory")
        st.info("Model should be trained to classify these flower types: " + ", ".join(FLOWER_CLASSES))
        return
    
    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.info(f"**Classes:** {len(FLOWER_CLASSES)}")
    st.sidebar.info(f"**Flower Types:**")
    for i, flower in enumerate(FLOWER_CLASSES, 1):
        st.sidebar.write(f"{i}. {flower}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a flower image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image of a flower for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
        
        with col2:
            st.subheader("Processed Image")
            # Show processed version
            processed_for_display = preprocess_image(image)
            processed_img = (processed_for_display[0] * 255).astype(np.uint8)
            st.image(processed_img, caption="Processed Image (224x224)", use_column_width=True)
        
        # Prediction button
        if st.button("🔍 Detect Flower", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, class_probabilities = predict_flower(model, image)
                
                if predicted_class is not None:
                    st.success("Analysis complete!")
                    display_results(predicted_class, confidence, class_probabilities)
                    
                    # Additional information
                    st.subheader("About This Flower")
                    flower_info = {
                        "Daisy": "Daisies are simple, cheerful flowers with white petals and yellow centers.",
                        "Dandelion": "Dandelions are bright yellow flowers that later turn into fluffy seed heads.",
                        "Rose": "Roses are classic flowers known for their beauty and fragrance, available in many colors.",
                        "Sunflower": "Sunflowers are large, bright yellow flowers that follow the sun throughout the day.",
                        "Tulip": "Tulips are elegant spring flowers with cup-shaped blooms in various colors."
                    }
                    
                    if predicted_class in flower_info:
                        st.info(flower_info[predicted_class])
    
    else:
        # Instructions
        st.info("👆 Upload a flower image to get started!")
        
        # Example section
        st.subheader("How to use this app:")
        st.markdown("""
        1. **Upload an image** using the file uploader above
        2. **Click 'Detect Flower'** to analyze the image
        3. **View results** including the predicted flower type and confidence score
        4. **Check probabilities** for all flower classes in the chart
        """)
        
        st.subheader("Supported flower types:")
        cols = st.columns(5)
        for i, flower in enumerate(FLOWER_CLASSES):
            with cols[i]:
                st.write(f"🌸 **{flower}**")

if __name__ == "__main__":
    main()
