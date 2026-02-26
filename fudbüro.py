import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import datetime
import io
import pandas as pd

# -------------------------------------------------
# Seitenkonfiguration
# -------------------------------------------------
st.set_page_config(page_title="KI Bildklassifikation", layout="wide")
st.title("🧠 KI Bildklassifikation mit Datenbank & Suche")

# -------------------------------------------------
# Datenbank initialisieren
# -------------------------------------------------
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

# -------------------------------------------------
# Daten speichern
# -------------------------------------------------
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

# -------------------------------------------------
# Suchfunktion
# -------------------------------------------------
def search_predictions(search_term=""):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    if search_term:
        query = """
            SELECT id, filename, predicted_class, confidence, timestamp 
            FROM predictions
            WHERE 
                id LIKE ? OR
                filename LIKE ? OR
                predicted_class LIKE ? OR
                timestamp LIKE ?
            ORDER BY id DESC
        """
        like_term = f"%{search_term}%"
        c.execute(query, (like_term, like_term, like_term, like_term))
    else:
        c.execute("""
            SELECT id, filename, predicted_class, confidence, timestamp 
            FROM predictions
            ORDER BY id DESC
        """)

    rows = c.fetchall()
    conn.close()
    return rows

# -------------------------------------------------
# Einzelnes Bild abrufen
# -------------------------------------------------
def get_image_by_id(record_id):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT image FROM predictions WHERE id=?", (record_id,))
    img = c.fetchone()
    conn.close()
    return img[0] if img else None

# -------------------------------------------------
# Löschen
# -------------------------------------------------
def delete_record(record_id):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE id=?", (record_id,))
    conn.commit()
    conn.close()

# -------------------------------------------------
# Modell laden
# -------------------------------------------------
@st.cache_resource
def load_keras_model():
    return load_model("keras_model.h5", compile=False)

model = load_keras_model()

with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Bild erkennen", "🗄 Datenbank anzeigen"])

# =================================================
# TAB 1 – Bildklassifikation
# =================================================
with tab1:

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

        # Bild als Bytes speichern
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        save_to_db(uploaded_file.name, img_bytes, class_name, confidence_score)
        st.success("✅ In Datenbank gespeichert")

    # =================================================
# PREMIUM TAB 2 – Galerie Dashboard
# =================================================

with tab2:

    st.subheader("🌌 Premium KI Galerie Dashboard")

    # -------------------------
    # Daten laden
    # -------------------------
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("""
        SELECT id, image, timestamp 
        FROM predictions 
        ORDER BY timestamp DESC
    """)

    records = c.fetchall()
    conn.close()

    if not records:
        st.info("Noch keine Bilder gespeichert.")
        st.stop()

    # -------------------------
    # DataFrame für Statistik
    # -------------------------
    import pandas as pd
    from collections import Counter

    df = pd.DataFrame(records, columns=["ID", "Image", "Timestamp"])

    df["Date"] = df["Timestamp"].apply(lambda x: x.split(" ")[0])

    # -------------------------
    # Sidebar Statistik
    # -------------------------
    st.sidebar.header("📊 Statistik")

    date_counts = Counter(df["Date"])

    if date_counts:
        stat_df = pd.DataFrame(
            list(date_counts.items()),
            columns=["Datum", "Anzahl Bilder"]
        )

        st.sidebar.bar_chart(
            stat_df.set_index("Datum")
        )

    # -------------------------
    # Suche
    # -------------------------
    search_term = st.text_input("🔎 Suche nach Datum (YYYY-MM-DD) oder ID")

    if search_term:
        df = df[
            df["Date"].str.contains(search_term) |
            df["ID"].astype(str).str.contains(search_term)
        ]

    # -------------------------
    # Pagination
    # -------------------------
    items_per_page = 12

    total_pages = max(1, len(df) // items_per_page + 1)

    page = st.number_input(
        "📄 Seite auswählen",
        min_value=1,
        max_value=total_pages,
        value=1
    )

    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    df_page = df.iloc[start_idx:end_idx]

    # -------------------------
    # Galerie Anzeige
    # -------------------------
    st.markdown("### 🖼 Galerie")

    cols = st.columns(4)

    for i, (_, row) in enumerate(df_page.iterrows()):

        with cols[i % 4]:

            st.image(row["Image"], use_container_width=True)

            st.caption(f"🆔 ID: {row['ID']}")
            st.caption(f"📅 {row['Timestamp']}")

            # Zoom Button
            if st.button(f"🔍 Bild {row['ID']}", key=f"zoom_{row['ID']}"):

                st.session_state["zoom_image"] = row["Image"]
                st.session_state["zoom_id"] = row["ID"]

    # -------------------------
    # Zoom View
    # -------------------------
    if "zoom_image" in st.session_state:

        st.markdown("---")
        st.subheader(f"🔍 Zoom Ansicht – ID {st.session_state['zoom_id']}")

        st.image(
            st.session_state["zoom_image"],
            use_container_width=True
        )
