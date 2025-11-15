import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
from torch import nn
from datetime import datetime, timedelta
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import smtplib
from email.mime.text import MIMEText

# === تعريف النموذج ===
class LSTMModel(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=3, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(128, 16)
        self.dense2 = nn.Linear(16, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = nn.ReLU()(self.dense1(out))
        out = self.dense2(out)
        return out

# === تحميل الموديل والمعاير ===
@st.cache_resource
def load_model():
    model = LSTMModel()
    model.load_state_dict(torch.load("model_v3.pt", map_location='cpu'))
    model.eval()
    with open("normalizers.pkl", 'rb') as f:
        normalizers = pickle.load(f)
    return model, normalizers

model, normalizers = load_model()

# === دالة تقدير الإشعاع ===
def estimate_rad(clouds, uv_max):
    base_rad = 6.5
    rad_reduction = (clouds / 100 * 4.0) + (uv_max / 10 * 0.5)
    return max(0, base_rad - rad_reduction)

# === تحميل البيانات ===
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv("pred_result_v3.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['city'] = df.filter(like='city_').idxmax(axis=1).str.replace('city_', '')
    return df

df = load_data()

# === دالة التنبؤ ===
def predict_tomorrow(city, last_6, temp_max, mean_rad):
    future_data = {"mean_rad": mean_rad, "temp_max": temp_max}
    last_7 = pd.concat([last_6, pd.Series(future_data).to_frame().T], ignore_index=True)
    last_7['rad_rolling'] = last_7['mean_rad'].rolling(3, min_periods=1).mean()
    last_7['temp_rolling'] = last_7['temp_max'].rolling(3, min_periods=1).mean()

    def norm(val, feat):
        mean, std = normalizers[city][feat]
        return (val - mean) / std

    features = []
    for _, row in last_7.iterrows():
        f = [
            norm(row['mean_rad'], 'mean_rad'),
            norm(row['temp_max'], 'temp_max'),
            norm(row['rad_rolling'], 'rad_rolling'),
            norm(row['temp_rolling'], 'temp_rolling')
        ]
        onehot = [1.0 if c == f"city_{city}" else 0.0 for c in normalizers.keys()]
        features.append(f + onehot)

    X = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()
    return prob

# === دالة الإشعارات ===
def send_alerts(results_df):
    high_risk = results_df[results_df['الاحتمال'].str.replace('%', '').astype(float) > 60]
    if not high_risk.empty:
        msg = MIMEText(f"تحذير: انقطاع محتمل في {high_risk['المدينة'].tolist()}")
        msg['Subject'] = 'تحذير انقطاع طاقة شمسية'
        msg['From'] = 'alert@yourdomain.com'
        msg['To'] = 'admin@yourdomain.com'
        
        # SMTP (غيّر الإعدادات)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('your_email@gmail.com', 'your_password')
        server.send_message(msg)
        server.quit()
        st.success(f"تم إرسال إشعار لـ {len(high_risk)} مدينة!")
    else:
        st.info("لا توجد مدن تحتاج إشعار.")

# === العنوان ===
st.title("نظام التنبؤ بانقطاع الطاقة الشمسية (مع خريطة وإشعارات)")
st.markdown(f"**آخر تحديث:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
st.subheader(f"احتمالات انقطاع الكهرباء غدًا: {tomorrow}")

# === التنبؤ لكل مدينة ===
cities = df['city'].unique()
results = []

for city in cities:
    last_6 = df[df['city'] == city].tail(6).copy()
    if len(last_6) >= 6:
        # بيانات الطقس الحقيقية
        if city == "Marrakech":
            temp_max = 18.3; clouds = 100; uv_max = 4; mean_rad = estimate_rad(clouds, uv_max)
        elif city == "Casablanca":
            temp_max = 20.0; clouds = 100; uv_max = 3; mean_rad = estimate_rad(clouds, uv_max)
        elif city == "Ouarzazate":
            temp_max = 20.6; clouds = 0; uv_max = 5; mean_rad = estimate_rad(clouds, uv_max)
        elif city == "Rabat":
            temp_max = 19.0; clouds = 100; uv_max = 3; mean_rad = estimate_rad(clouds, uv_max)
        elif city == "Tangier":
            temp_max = 16.7; clouds = 100; uv_max = 2; mean_rad = estimate_rad(clouds, uv_max)
        elif city == "Agadir":
            temp_max = 21.7; clouds = 20; uv_max = 5; mean_rad = estimate_rad(clouds, uv_max)
        
        prob = predict_tomorrow(city, last_6, temp_max, mean_rad)
        status = "مستقر" if prob < 0.4 else "مراقبة" if prob < 0.6 else "خطر"
        color = "green" if prob < 0.4 else "orange" if prob < 0.6 else "red"
        results.append({
            "المدينة": city, "الاحتمال": f"{prob:.1%}", "الحرارة غدًا": f"{temp_max:.1f}°C",
            "الإشعاع": f"{mean_rad:.1f} kWh/m²", "الحالة": status, "اللون": color
        })

results_df = pd.DataFrame(results)
st.table(results_df.style.apply(lambda row: [f"background-color: {row['اللون']}" for _ in row], axis=1))

# === إشعارات ===
if st.button("إرسال إشعارات للمدن المعرضة للخطر"):
    send_alerts(results_df)

# === خريطة تفاعلية ===
st.subheader("خريطة تفاعلية للاحتمالات")
m = folium.Map(location=[31.7917, -7.9921], zoom_start=6)  # مركز المغرب

coords = {
    "Marrakech": [31.63, -8.00],
    "Casablanca": [33.57, -7.59],
    "Ouarzazate": [30.93, -6.91],
    "Rabat": [34.02, -6.84],
    "Tangier": [35.76, -5.81],
    "Agadir": [30.41, -9.60]
}

for _, row in results_df.iterrows():
    city = row['المدينة']
    prob = float(row['الاحتمال'].replace('%', ''))
    color = row['اللون']
    folium.CircleMarker(
        location=coords[city],
        radius=prob / 2,
        popup=f"{city}: {row['الاحتمال']} - {row['الحالة']}",
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7
    ).add_to(m)

st_folium(m, width=700, height=500)

# === رسم بياني ===
st.subheader("تاريخ الاحتمالات (آخر 30 يومًا)")
selected_city = st.selectbox("اختر مدينة:", cities)
city_data = df[df['city'] == selected_city].tail(30).copy()
if not city_data.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_data['date'], y=city_data['pred_prob'], mode='lines+markers', name='الاحتمال', line=dict(color='blue')))
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="مراقبة")
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="خطر")
    fig.update_layout(title=f"احتمالات انقطاع الكهرباء في {selected_city}", xaxis_title="التاريخ", yaxis_title="الاحتمال")
    st.plotly_chart(fig, use_container_width=True)