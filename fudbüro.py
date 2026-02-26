import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import datetime
import io

# -----------------------------------
# Seitenkonfiguration
# -----------------------------------
st.set_page_config(page_title="Bildklassifikation", layout="centered")
st.title("🧠 Bildklassifikation mit Datenbank")

# -----------------------------------
# Datenbank erstellen / verbinden
# -----------------------------------
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            image BLOB,
            predicted_class TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------------
# Speichern in DB
# -----------------------------------
def save_to_db(filename, image_bytes, predicted_class, confidence):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions 
        (filename, image, predicted_class, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (
        filename,
        image_bytes,
        predicted_class,
        confidence,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

# -----------------------------------
# Modell laden
# -----------------------------------
@st.cache_resource
def load_keras_model():
    model = load_model("keras_model.h5", compile=False)
    return model

model = load_keras_model()

# Labels laden
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -----------------------------------
# Datei Upload
# -----------------------------------
uploaded_file = st.file_uploader("📤 Lade ein Bild hoch...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Bild vorbereiten
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # Ergebnis anzeigen
    st.subheader("🔍 Ergebnis")
    st.success(f"**Klasse:** {class_name}")
    st.info(f"**Confidence:** {round(confidence_score * 100, 2)} %")

    # -----------------------------------
    # Bild als Bytes speichern
    # -----------------------------------
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # In DB speichern
    save_to_db(uploaded_file.name, img_bytes, class_name, confidence_score)

    st.success("✅ Bild und Vorhersage wurden in der Datenbank gespeichert.")
