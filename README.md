# thermal-plant-optimizer-streamlit
Streamlit dashboard for thermal plant generation optimization using real-time price &amp; schedule inputs.

# ⚡ Thermal Power Plant Optimization Dashboard

This Streamlit app provides a flexible interface for simulating and optimizing thermal power plant generation using price forecasts and discom schedules. 

---

## 📌 Features

- 🔧 Full control over technical & economic parameters (Pmin, Pmax, fuel cost, ramp constraints, etc.)
- 📅 Dynamic price forecasts from MongoDB, CSV, or simulation
- ⚙️ Discom schedule input from MongoDB, CSV, or simulation
- 📈 Interactive charts for generation, profit, and price
- 🧮 Optimization using PuLP (Linear Programming)
- 📥 CSV Export of optimized block-wise results

---

## 📂 File Structure

```bash
.
├── app.py                  # Streamlit app
├── requirements.txt        # Install dependencies
├── README.md               # This file
├── data/
│   ├── sample_price_forecast.csv
│   └── sample_schedule.csv

