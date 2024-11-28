import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# โหลดโมเดล
model = load_model('mnist_cnn_with_augmentation.keras')

# ฟังก์ชันสำหรับการพยากรณ์
def predict_image(image):
    image = ImageOps.grayscale(image)  # แปลงภาพเป็นขาวดำ
    image = image.resize((28, 28))    # ปรับขนาดเป็น 28x28
    image = np.array(image).reshape(-1, 28, 28, 1) / 255.0  # Normalize
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return predicted_class, confidence

# ส่วนของ UI
st.title("MNIST Digit Classifier")
st.write("อัปโหลดภาพตัวเลข (0-9) เพื่อดูผลการพยากรณ์")

# อัปโหลดภาพ
uploaded_file = st.file_uploader("เลือกภาพที่ต้องการ", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)
    predicted_class, confidence = predict_image(image)
    st.write(f"ตัวเลขที่พยากรณ์: {predicted_class}")
    st.write(f"ความมั่นใจ: {confidence:.2%}")

# ส่วนสำหรับการดาวน์โหลดรูปภาพผ่าน Google Drive
st.write("---")
st.header("Download Test Images")
st.write("ดาวน์โหลดไฟล์รูปภาพตัวเลข 0-9 จาก MNIST Dataset ได้ที่ลิงก์ด้านล่าง:")

# แนบลิงก์ Google Drive
drive_link = "https://drive.google.com/drive/folders/17qQTu75wM5Pti_m9He3jBaLgMMGlc3ta?usp=sharing"
st.markdown(f"[Click here to download the test images]( {drive_link} )")
