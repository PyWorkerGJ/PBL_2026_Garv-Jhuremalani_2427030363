import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import json
import os


DEVICE = "cpu"


auth_model_path = "models/auth_model.pth"
auth_model = models.resnet18(weights=None)
auth_model.fc = torch.nn.Linear(auth_model.fc.in_features, 1)

if os.path.exists(auth_model_path):
    auth_model.load_state_dict(torch.load(auth_model_path, map_location=DEVICE))
auth_model.eval()


disease_model_path = "models/best_disease_model.pth"
class_names_path = "models/class_names.json"

disease_model = models.resnet18(weights=None)

if os.path.exists(class_names_path):
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
else:
    class_names = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold"]

disease_model.fc = torch.nn.Linear(
    disease_model.fc.in_features,
    len(class_names)
)

if os.path.exists(disease_model_path):
    disease_model.load_state_dict(torch.load(disease_model_path, map_location=DEVICE))

disease_model.eval()


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(img):
    return transform(img).unsqueeze(0)

def predict_auth(img):
    x = preprocess(img)
    with torch.no_grad():
        out = auth_model(x).squeeze()
        prob = torch.sigmoid(out).item()
    label = "Real" if prob >= 0.5 else "Fake"
    return label, prob

def predict_disease(img):
    x = preprocess(img)
    with torch.no_grad():
        out = disease_model(x)
        idx = out.argmax(1).item()
        conf = torch.softmax(out, dim=1)[0][idx].item()
    return class_names[idx], conf


st.title("🌱 Plant Disease Detector with Fake Image Authentication")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        label, prob = predict_auth(img)
        st.write(f"**Authenticity:** {label} (Confidence: {prob:.3f})")

        if label == "Fake":
            st.error("❌ Fake/AI-generated image detected. Disease detection stopped.")
        else:
            disease, conf = predict_disease(img)
            st.success(f"🌿 Disease: {disease} (Confidence: {conf:.3f})")
