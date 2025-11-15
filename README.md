# Solar Power Outage Prediction System in Morocco  
**Using LSTM + Real Weather Data + Interactive Dashboard**

---

## Overview
An intelligent system that predicts the **probability of solar power outage tomorrow** in 6 Moroccan cities based on:
- Last 7 days of solar radiation and temperature
- Real weather forecasts (clouds, UV, temperature)
- High-accuracy trained LSTM model (**PR-AUC = 0.968**)

---

## Required Files
| File | Description |
|-------|-----------|
| `pred_result_v3.csv` | Historical data + predictions |
| `model_v3.pt` | Trained model (state_dict) |
| `normalizers.pkl` | Normalization scalers per city |
| `app.py` | Interactive dashboard (Streamlit) |

---

## Installation
```bash
pip install streamlit pandas torch plotly folium streamlit-folium