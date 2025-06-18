import streamlit as st
import numpy as np
import pandas as pd
import datetime
from pymongo import MongoClient
from pulp import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide", page_title="Thermal Plant Optimizer")
st.title("⚡ Thermal Power Plant Optimization Dashboard")

# -----------------------------
# Sidebar - Inputs
# -----------------------------
st.sidebar.header("🔧 Model Parameters")

num_blocks = st.sidebar.slider("Number of Time Blocks", 4, 96, 96, step=4)
blocks = list(range(num_blocks))

fuel_cost = st.sidebar.number_input("Fuel Cost (₹/MWh)", value=1500)
variable_cost = st.sidebar.number_input("Variable Cost (₹/MWh)", value=50)
fixed_cost = st.sidebar.number_input("Fixed Cost per Block (₹)", value=200)

startup_cost = st.sidebar.number_input("Startup Cost (₹)", value=5000)
shutdown_cost = st.sidebar.number_input("Shutdown Cost (₹)", value=2000)

ramp_up_limit = st.sidebar.number_input("Ramp Up Limit (MW/block)", value=20)
ramp_down_limit = st.sidebar.number_input("Ramp Down Limit (MW/block)", value=25)
ramp_up_cost = st.sidebar.number_input("Ramp Up Cost (₹/MW)", value=100)
ramp_down_cost = st.sidebar.number_input("Ramp Down Cost (₹/MW)", value=80)

pmin = st.sidebar.number_input("Pmin (MW)", value=40)
pmax = st.sidebar.number_input("Pmax (MW)", value=100)

# -----------------------------
# Price Forecast Input
# -----------------------------
st.sidebar.markdown("### 📈 Price Forecast Source")
price_input_method = st.sidebar.radio("Select Price Input Method", ["Simulate", "Upload CSV", "From MongoDB"])

if price_input_method == "Simulate":
    base_prices = [1400, 1350, 1300, 1250, 1200, 1150, 1100, 1500, 1600, 1800, 2000, 2200,
                   2100, 1900, 1700, 1500]
    price_forecast = list(np.tile(base_prices, int(np.ceil(num_blocks / len(base_prices)))))[:num_blocks]
elif price_input_method == "Upload CSV":
    uploaded_price = st.sidebar.file_uploader("Upload Price Forecast CSV", type="csv")
    if uploaded_price:
        df_price = pd.read_csv(uploaded_price)
        price_forecast = df_price.iloc[:, 0].tolist()[:num_blocks]
    else:
        st.stop()
else:
    selected_date = st.sidebar.date_input("Select Date for Price Forecast", value=datetime.date.today())
    query_date = selected_date.strftime("%d-%m-%Y")
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["your_db"]
        collection = db["your_price_forecast_collection"]
        forecast_cursor = collection.find({"date": query_date}).sort("block", 1)
        price_forecast = [doc["price"] for doc in forecast_cursor]
        #query_date
        if len(price_forecast) < num_blocks:
            last_price = price_forecast[-1] if price_forecast else 1500
            price_forecast += [last_price] * (num_blocks - len(price_forecast))
    except:
        st.error("MongoDB connection failed.")
        st.stop()

# -----------------------------
# Discom Schedule Input
# -----------------------------
st.sidebar.markdown("### ⚙️ Discom Schedule Source")
schedule_input_method = st.sidebar.radio("Select Schedule Input", ["Simulate", "Upload CSV", "From MongoDB"])

if schedule_input_method == "Simulate":
    scheduled_gen = [60 if 24 <= t < 70 else 50 for t in blocks]
elif schedule_input_method == "Upload CSV":
    uploaded_sched = st.sidebar.file_uploader("Upload Discom Schedule CSV", type="csv")
    if uploaded_sched:
        df_sched = pd.read_csv(uploaded_sched)
        scheduled_gen = df_sched.iloc[:, 0].tolist()[:num_blocks]
    else:
        st.stop()
else:
    selected_date_sched = st.sidebar.date_input("Select Date for Discom Schedule", value=datetime.date.today())
    query_date_sched = selected_date_sched.strftime("%Y-%m-%d")
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["your_db"]
        coll_sched = db["your_schedule_collection"]
        sched_cursor = coll_sched.find({"date": query_date_sched}).sort("block", 1)
        scheduled_gen = [doc["schedule"] for doc in sched_cursor]
        if len(scheduled_gen) < num_blocks:
            last_val = scheduled_gen[-1] if scheduled_gen else 50
            scheduled_gen += [last_val] * (num_blocks - len(scheduled_gen))
    except:
        st.error("MongoDB schedule fetch failed.")
        st.stop()

# -----------------------------
# Run Optimization
# -----------------------------
model = LpProblem("Thermal_Plant_Optimization", LpMaximize)

P = LpVariable.dicts("Power", blocks, lowBound=0, upBound=pmax, cat='Continuous')
surplus = LpVariable.dicts("Surplus", blocks, lowBound=0, cat='Continuous')
ramp_up_amt = LpVariable.dicts("RampUp", blocks[1:], lowBound=0, cat='Continuous')
ramp_down_amt = LpVariable.dicts("RampDown", blocks[1:], lowBound=0, cat='Continuous')
status = LpVariable.dicts("Status", blocks, cat='Binary')
startup = LpVariable.dicts("Startup", blocks[1:], cat='Binary')
shutdown = LpVariable.dicts("Shutdown", blocks[1:], cat='Binary')

# Objective
model += lpSum(
    surplus[t] * (price_forecast[t] - fuel_cost - variable_cost) * 0.25
    - fixed_cost * status[t]
    - ramp_up_amt[t] * ramp_up_cost if t > 0 else 0
    - ramp_down_amt[t] * ramp_down_cost if t > 0 else 0
    - startup[t] * startup_cost if t > 0 else 0
    - shutdown[t] * shutdown_cost if t > 0 else 0
    for t in blocks
)

# Constraints
for t in blocks:
    model += P[t] >= scheduled_gen[t]
    model += surplus[t] == P[t] - scheduled_gen[t]
    model += P[t] <= pmax * status[t]
    model += P[t] >= pmin * status[t]

for t in blocks[1:]:
    model += P[t] - P[t-1] <= ramp_up_amt[t]
    model += P[t-1] - P[t] <= ramp_down_amt[t]
    model += P[t] - P[t-1] <= ramp_up_limit
    model += P[t-1] - P[t] <= ramp_down_limit
    model += startup[t] >= status[t] - status[t-1]
    model += shutdown[t] >= status[t-1] - status[t]

model.solve()

# -----------------------------
# Results
# -----------------------------
data = []
cumulative_profit = 0
net_profit_list = []
generation = []

for t in blocks:
    prev = P[t-1].varValue if t > 0 else 0
    curr = P[t].varValue
    surplus_val = surplus[t].varValue
    ru = ramp_up_amt[t].varValue if t > 0 else 0
    rd = ramp_down_amt[t].varValue if t > 0 else 0
    su = startup[t].varValue if t > 0 else 0
    sd = shutdown[t].varValue if t > 0 else 0
    stval = status[t].varValue

    gp = surplus_val * (price_forecast[t] - fuel_cost - variable_cost) * 0.25
    rc = ru * ramp_up_cost + rd * ramp_down_cost
    ssc = su * startup_cost + sd * shutdown_cost
    fcost = stval * fixed_cost
    net = gp - rc - ssc - fcost
    cumulative_profit += net
    net_profit_list.append(net)
    generation.append(curr)

    data.append({
        "Block": t+1,
        "Price": price_forecast[t],
        "P[t-1]": prev,
        "P[t]": curr,
        "Surplus": surplus_val,
        "Ramp Up": ru,
        "Ramp Down": rd,
        "Startup": su,
        "Shutdown": sd,
        "Net Profit": net,
        "Cumulative Profit": cumulative_profit
    })

results_df = pd.DataFrame(data)

# -----------------------------
# Plotly Chart
# -----------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                    subplot_titles=("Price & Generation", "Net Profit & Cumulative Profit"))

fig.add_trace(go.Scatter(x=blocks, y=price_forecast, name="Price (₹/MWh)", line=dict(color='red')), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=blocks, y=generation, name="Generation (MW)", line=dict(color='blue')), row=1, col=1, secondary_y=True)
fig.add_trace(go.Bar(x=blocks, y=net_profit_list, name="Net Profit", marker_color=['green' if x >= 0 else 'crimson' for x in net_profit_list]), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=blocks, y=np.cumsum(net_profit_list), name="Cumulative Profit", line=dict(color='black', dash='dash')), row=2, col=1, secondary_y=True)

fig.update_layout(height=700, title_text="Thermal Plant Operation Results", template="plotly_white", showlegend=True)
fig.update_yaxes(title_text="Price (₹/MWh)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Generation (MW)", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Net Profit (₹)", row=2, col=1, secondary_y=False)
fig.update_yaxes(title_text="Cumulative Profit (₹)", row=2, col=1, secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Export
# -----------------------------
st.download_button("📥 Download Results as CSV", data=results_df.to_csv(index=False), file_name="thermal_results.csv")
