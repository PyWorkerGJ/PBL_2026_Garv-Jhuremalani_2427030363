import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import json
import os
from datetime import datetime

DEVICE = "cpu"


auth_model_path = "models/auth_model.pth"
auth_model = models.resnet18(weights=None)
auth_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 1)
)

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

disease_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, len(class_names))
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


TREATMENTS = {
    "Tomato___Early_blight": (
        "Early Blight (Alternaria solani): Remove infected lower leaves immediately. "
        "Apply copper-based fungicide or mancozeb every 7-10 days. "
        "Avoid overhead watering and ensure good air circulation around plants."
    ),
    "Tomato___Late_blight": (
        "Late Blight (Phytophthora infestans): This spreads rapidly — act immediately. "
        "Apply chlorothalonil or metalaxyl-based fungicide. "
        "Remove and destroy all infected plant material. Do not compost infected leaves."
    ),
    "Tomato___Leaf_Mold": (
        "Leaf Mold (Passalora fulva): Improve greenhouse ventilation and reduce humidity below 85%. "
        "Apply fungicides containing chlorothalonil or mancozeb. "
        "Water at the base of plants and avoid wetting foliage."
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Spider Mites: Spray plants with water to dislodge mites. "
        "Apply neem oil or insecticidal soap spray every 5-7 days. "
        "In severe cases, use miticides like abamectin. Avoid over-fertilizing with nitrogen."
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Tomato Mosaic Virus: No cure available — remove and destroy infected plants. "
        "Disinfect tools with bleach solution. Control aphid populations to prevent spread. "
        "Use virus-resistant seed varieties for future planting."
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Yellow Leaf Curl Virus: Transmitted by whiteflies — control whitefly population using yellow sticky traps and imidacloprid. "
        "Remove infected plants to prevent spread. Use reflective mulch to deter whiteflies."
    ),
    "Tomato___Bacterial_spot": (
        "Bacterial Spot: Apply copper-based bactericide at first sign of infection. "
        "Avoid working with plants when wet. Rotate crops and use certified disease-free seeds."
    ),
    "Tomato___Septoria_leaf_spot": (
        "Septoria Leaf Spot: Remove infected leaves and apply fungicide (mancozeb or chlorothalonil). "
        "Mulch around the base to prevent soil splash. Ensure proper plant spacing for air flow."
    ),
    "Tomato___healthy": (
        "Plant appears healthy! Continue regular care: water consistently at the base, "
        "fertilize every 2-3 weeks, and monitor regularly for early signs of disease."
    ),
}

def get_treatment(disease_name):
    if disease_name in TREATMENTS:
        return TREATMENTS[disease_name]
    for key in TREATMENTS:
        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
            return TREATMENTS[key]
    return "No specific treatment found. Consult a local agricultural expert for advice."


if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.header("📋 Prediction History")
    if st.session_state.history:
        for i, record in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{record['time']} — {record['result']}"):
                st.write(f"**Authenticity:** {record['auth']} ({record['auth_conf']:.2f})")
                if record['disease']:
                    st.write(f"**Disease:** {record['disease']}")
                    st.write(f"**Confidence:** {record['disease_conf']:.2%}")
                    st.write(f"**Result:** {record['result']}")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Analyze an image to get started.")


st.title("🌱 Plant Disease Detector with Fake Image Authentication")

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Capture"])

img = None

with tab1:
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

with tab2:
    camera_photo = st.camera_input("Take a photo of the leaf")
    if camera_photo:
        img = Image.open(camera_photo).convert("RGB")


if img:
    st.image(img, caption="Input Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):

            label, prob = predict_auth(img)
            st.write(f"**Authenticity:** {label} (Confidence: {prob:.3f})")

            record = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "auth": label,
                "auth_conf": prob,
                "disease": None,
                "disease_conf": None,
                "result": None,
            }

            if label == "Fake":
                st.error("❌ Fake/AI-generated image detected. Disease detection stopped.")
                record["result"] = "Fake Image — Stopped"

            else:
                disease, conf = predict_disease(img)

                CONFIDENCE_THRESHOLD = 0.60
                if conf < CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"⚠️ Low confidence prediction ({conf:.1%}). "
                        "Please retake the image in better lighting or at a closer distance."
                    )
                    record["result"] = f"Low Confidence ({conf:.1%})"
                else:
                    st.success(f"🌿 Disease Detected: **{disease}** (Confidence: {conf:.1%})")
                    record["result"] = disease

                record["disease"] = disease
                record["disease_conf"] = conf

                st.markdown("---")
                st.subheader("💊 Treatment Recommendation")
                treatment = get_treatment(disease)
                st.info(treatment)

            st.session_state.history.append(record)
