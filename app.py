import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# 1. ตั้งค่าหน้าเว็บเป็นแบบ Wide
st.set_page_config(page_title="Iris Classifier App", page_icon="🌸", layout="wide")

# 2. โหลดข้อมูลและเทรนโมเดล (ใช้ Cache เพื่อให้โหลดเร็ว)
@st.cache_data
def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model, iris.target_names

model, target_names = train_model()

# ----------------------------------------------------
# SIDEBAR: ส่วนป้อนข้อมูล
# ----------------------------------------------------
with st.sidebar:
    st.title("🌸 scikit-learn")
    st.caption("Iris Classifier App")
    st.write("---")
    st.subheader("📊 INPUT FEATURES")
    st.caption("ปรับค่าสไลเดอร์เพื่อป้อนข้อมูลขนาดดอกไอริส:")

    sepal_length = st.slider("🌱 Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("🌿 Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
    petal_length = st.slider("🌷 Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("🌺 Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

# คำนวณผลลัพธ์จากโมเดล
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]
confidence = np.max(probabilities) * 100
predicted_species = target_names[prediction].title()

# คำอธิบายแต่ละสายพันธุ์
descriptions = {
    "Setosa": "ดอกไอริสเซโตซ่า มีลักษณะกลีบนอกขนาดเล็กและกลีบเลี้ยงกว้าง แยกแยะได้ชัดเจน",
    "Versicolor": "ดอกไอริสเวอร์ซิคัลเลอร์ มีขนาดปานกลาง สีม่วงหรือฟ้าอ่อน",
    "Virginica": "ดอกไอริสเวอร์จินิกา มีขนาดใหญ่ที่สุด กลีบดอกยาวและเรียว"
}

# ----------------------------------------------------
# MAIN CONTENT: ส่วนแสดงผล Dashboard
# ----------------------------------------------------
col_left, col_right = st.columns([2, 1])

# --- ฝั่งซ้าย: กราฟเปรียบเทียบค่า Feature ---
with col_left:
    st.subheader("Features Comparison")
    
    # วาดกราฟแท่งเปรียบเทียบ Input Features
    fig_features = go.Figure(data=[
        go.Bar(
            x=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            y=[sepal_length, sepal_width, petal_length, petal_width],
            marker_color=['#FF4B4B', '#1C83E1', '#00D4B1', '#808080']
        )
    ])
    fig_features.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_features, use_container_width=True)

# --- ฝั่งขวา: การ์ดแสดงผลคำทำนาย ---
with col_right:
    # แสดง % ความมั่นใจ
    st.metric(label="PREDICTION CONFIDENCE", value=f"{confidence:.1f}%")
    
    # แสดงชื่อสายพันธุ์ที่ทำนายได้
    st.success(f"### PREDICTED SPECIES\n## **Iris {predicted_species}**")
    st.caption(descriptions.get(predicted_species, ""))

# --- ส่วนล่าง: กราฟความน่าจะเป็น (Probability Distribution) ---
st.write("---")
st.subheader("Probability Distribution")
st.caption("ค่าความน่าจะเป็นจำแนกตาม 3 สายพันธุ์ (KNN Classifier)")

# วาดกราฟ Probability
fig_prob = go.Figure(data=[
    go.Bar(
        x=[name.title() for name in target_names],
        y=probabilities * 100,
        marker_color=['#808080' if i != prediction else '#1C83E1' for i in range(3)]
    )
])
fig_prob.update_layout(height=250, yaxis_range=[0, 100], margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_prob, use_container_width=True)