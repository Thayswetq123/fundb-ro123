import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import datetime
import io
import pandas as pd
import collections

# ================================
# PAGE CONFIG (Dark Mode Style)
# ================================
st.set_page_config(
    page_title="Ultra KI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 ULTRA PRO KI Bild Dashboard")

# ================================
# DATABASE
# ================================

def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
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

# ================================
# DATABASE FUNCTIONS
# ================================

def save_to_db(filename, img_bytes, predicted_class, confidence):

    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO predictions
        (filename,image,predicted_class,confidence,timestamp)
        VALUES (?,?,?,?,?)
    """, (
        filename,
        img_bytes,
        predicted_class,
        confidence,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

def load_predictions(search=""):
    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()

    df["date"] = df["timestamp"].apply(lambda x: x.split(" ")[0])

    if search:
        df = df[
            df["filename"].str.contains(search, case=False, na=False) |
            df["predicted_class"].str.contains(search, case=False, na=False) |
            df["date"].str.contains(search)
        ]

    return df

# ================================
# MODEL LOAD
# ================================

@st.cache_resource
def load_model_cached():
    return load_model("keras_model.h5", compile=False)

model = load_model_cached()

with open("labels.txt") as f:
    labels = [x.strip() for x in f.readlines()]

# ================================
# SIDEBAR ANALYTICS
# ================================

st.sidebar.header("📊 Analytics Dashboard")

# ================================
# TABS
# ================================

tab1, tab2, tab3 = st.tabs([
    "🔍 KI Prediction",
    "🖼 Galerie Ultra Pro",
    "📷 Webcam AI"
])

# ======================================
# TAB 1 – Prediction
# ======================================

with tab1:

    uploaded_file = st.file_uploader(
        "📤 Bild hochladen",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        size = (224,224)
        img_resized = ImageOps.fit(image,size)

        arr = np.asarray(img_resized)
        arr = (arr.astype(np.float32)/127.5)-1

        data = np.expand_dims(arr,0)

        prediction = model.predict(data, verbose=0)

        index = np.argmax(prediction)
        class_name = labels[index]
        confidence = float(prediction[0][index])

        st.success(f"🎯 Klasse: {class_name}")
        st.info(f"🔥 Confidence: {round(confidence*100,2)}%")

        # Save DB
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")

        save_to_db(
            uploaded_file.name,
            img_bytes.getvalue(),
            class_name,
            confidence
        )

        st.success("✅ In Datenbank gespeichert")

# ======================================
# TAB 2 – ULTRA PRO GALLERY
# ======================================

with tab2:

    st.subheader("🌌 Ultra Galerie Dashboard")

    search = st.text_input("🔎 Suche")

    df = load_predictions(search)

    # Sidebar Statistik
    if len(df)>0:

        st.sidebar.subheader("📈 Bilder pro Tag")

        counter = collections.Counter(df["date"])

        stat_df = pd.DataFrame(
            list(counter.items()),
            columns=["Datum","Anzahl"]
        )

        st.sidebar.line_chart(
            stat_df.set_index("Datum")
        )

    # Pagination
    per_page = 12

    pages = max(1, len(df)//per_page + 1)

    page = st.number_input(
        "📄 Seite",
        min_value=1,
        max_value=pages,
        value=1
    )

    start = (page-1)*per_page
    end = start+per_page

    df_page = df.iloc[start:end]

    # Gallery
    cols = st.columns(4)

    for i, row in df_page.iterrows():

        with cols[i%4]:

            st.image(row["image"], use_container_width=True)

            st.caption(f"🆔 {row['id']}")
            st.caption(f"📅 {row['timestamp']}")

            if st.button(f"🔍 Zoom {row['id']}", key=f"z{row['id']}"):

                st.session_state["zoom"] = row["image"]
                st.session_state["zoom_id"] = row["id"]

    # Zoom Popup Simulation
    if "zoom" in st.session_state:

        st.markdown("---")
        st.subheader(f"🔍 Zoom View ID {st.session_state['zoom_id']}")

        st.image(
            st.session_state["zoom"],
            use_container_width=True
        )

# ======================================
# TAB 3 – Webcam AI
# ======================================

with tab3:

    st.subheader("📷 Live Webcam KI Prediction")

    cam = st.camera_input("Take picture")

    if cam:

        image = Image.open(cam).convert("RGB")

        st.image(image, use_container_width=True)

        size=(224,224)
        img = ImageOps.fit(image,size)

        arr = np.asarray(img)
        arr = (arr.astype(np.float32)/127.5)-1

        data=np.expand_dims(arr,0)

        pred=model.predict(data,verbose=0)

        index=np.argmax(pred)

        st.success(labels[index])
        st.info(f"{round(float(pred[0][index])*100,2)}%")
