import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- App Configuration ---
st.set_page_config(page_title="KELAB Scalability, Scenario & Monte Carlo Model", layout="wide")
st.title("💧 KELAB Scalability, Scenario Comparison & Monte Carlo Simulation")

# --- Sidebar: Base Input Parameters ---
st.sidebar.header("🔧 Base Input Parameters")

# Timeline Settings
start_date = "2025-04-01"    # Lab operations begin in April 2025
certification_date = "2025-12-01"  # Certification target
end_date = "2027-12-01"      # Projection end date

# Staffing & Capacity Inputs
st.sidebar.subheader("Staffing & Capacity")
analysts_phase1 = st.sidebar.number_input("👩‍🔬 Analysts in Phase 1 (2025)", min_value=1, value=3, step=1)
analysts_phase2 = st.sidebar.number_input("📈 Analysts in Phase 2 (2026)", min_value=1, value=6, step=1)
analysts_phase3 = st.sidebar.number_input("🤖 Analysts in Phase 3 (2027)", min_value=1, value=8, step=1)
capacity_per_analyst = st.sidebar.number_input("📊 Capacity per Analyst (Samples/Month)", min_value=10, value=100, step=10)
shift_expansion_factor = st.sidebar.slider("⚡ Shift Expansion Factor (Phase 2)", 1.0, 3.0, 1.5, 0.1)
automation_factor = st.sidebar.slider("🤖 AI Automation Factor (Phase 3)", 1.0, 5.0, 2.0, 0.1)

# Pricing Inputs (from KELAB document)
st.sidebar.subheader("Pricing Model")
basic_metals_price = st.sidebar.number_input("💵 Basic Heavy Metal Test ($)", min_value=100, value=300, step=50)
advanced_metals_price = st.sidebar.number_input("💵 Advanced Heavy Metal Test ($)", min_value=200, value=400, step=50)
anion_test_price = st.sidebar.number_input("💵 Anion Test ($)", min_value=50, value=150, step=50)
# Test mix ratios (assumed constant)
mix_basic = 0.5    # 50% of samples
mix_advanced = 0.3 # 30% of samples
mix_anion = 0.2    # 20% of samples

# Fixed Cost Structure (monthly)
st.sidebar.subheader("Cost Structure")
equipment_lease = 10378.19  # Equipment lease cost
instrument_running = 2775    # Instrument running costs
labor_cost = 16667           # Labor costs
software_cost = 2000         # Software/licensing costs
qa_qc_percentage = 0.08       # QA/QC cost is 8% of revenue

# --- Base Model: Timeline & Capacity ---
months = pd.date_range(start=start_date, end=end_date, freq='MS')
df = pd.DataFrame({"Month": months})

# Define phases:
# Phase 1: 2025, Phase 2: 2026, Phase 3: 2027 and beyond.
df["Phase"] = np.where(df["Month"].dt.year == 2025, "Phase 1",
                np.where(df["Month"].dt.year == 2026, "Phase 2", "Phase 3"))

# Staffing assignment based on phase
df["Analysts"] = np.where(df["Phase"] == "Phase 1", analysts_phase1,
                   np.where(df["Phase"] == "Phase 2", analysts_phase2, analysts_phase3))

# Capacity adjustment based on phase
df["Capacity per Analyst"] = np.where(df["Phase"] == "Phase 1", capacity_per_analyst,
                               np.where(df["Phase"] == "Phase 2", capacity_per_analyst * shift_expansion_factor,
                                        capacity_per_analyst * automation_factor))
# Maximum possible capacity
df["Total Capacity"] = df["Analysts"] * df["Capacity per Analyst"]

# Base Revenue (assuming full capacity utilization)
df["Base Revenue"] = (
    df["Total Capacity"] * mix_basic * basic_metals_price +
    df["Total Capacity"] * mix_advanced * advanced_metals_price +
    df["Total Capacity"] * mix_anion * anion_test_price
)
# QA/QC cost and Fixed Costs
df["QA/QC Cost"] = df["Base Revenue"] * qa_qc_percentage
fixed_costs = equipment_lease + instrument_running + labor_cost + software_cost
df["Total Costs"] = fixed_costs + df["QA/QC Cost"]
df["Base Profit"] = df["Base Revenue"] - df["Total Costs"]

# Goal Tracking for 1,000 samples/month
df["Goal Status"] = np.where(df["Total Capacity"] >= 1000, "✅ Goal Met", "⚠️ Under Target")

st.markdown("### 📅 Base Model: Monthly Capacity & Financial Projection")
st.dataframe(df.style.applymap(lambda x: "background-color: #90EE90" if x == "✅ Goal Met" else "", subset=["Goal Status"]))

fig_capacity = px.line(df, x="Month", y="Total Capacity", color="Phase",
                       title="📈 Lab Capacity Growth Over Time",
                       labels={"Total Capacity": "Samples Processed/Month"},
                       markers=True)
fig_capacity.add_hline(y=1000, line_dash="dot", line_color="red", annotation_text="Target: 1,000 Samples/Month")
st.plotly_chart(fig_capacity, use_container_width=True)

fig_financial = px.line(df, x="Month", y=["Base Revenue", "Base Profit"],
                        title="💰 Base Revenue & Profit Projection",
                        labels={"value": "Amount ($)", "variable": "Metric"},
                        markers=True)
st.plotly_chart(fig_financial, use_container_width=True)

# --- Scenario Comparison Section ---
st.markdown("## 🔍 Scenario Comparison")
st.markdown("""
We now compare three scenarios by applying different multipliers to the computed capacity and pricing.  
- **Conservative:** Lower realized capacity and discounted pricing.  
- **Moderate:** Base case.  
- **Optimistic:** Higher realized capacity and premium pricing.
""")

# Define scenario multipliers
scenario_params = {
    "Conservative": {"capacity_factor": 0.5, "price_multiplier": 0.8333},
    "Moderate": {"capacity_factor": 0.75, "price_multiplier": 1.0},
    "Optimistic": {"capacity_factor": 1.0, "price_multiplier": 1.1667},
}

# Create a copy of base model for scenario comparison
df_scenarios = df.copy()
for scenario, params in scenario_params.items():
    cap_factor = params["capacity_factor"]
    price_mult = params["price_multiplier"]
    # Realized capacity adjusted by factor
    df_scenarios[f"{scenario} Realized Capacity"] = df_scenarios["Total Capacity"] * cap_factor
    # Scenario revenue: apply price multiplier to all test types
    df_scenarios[f"{scenario} Revenue"] = (
        df_scenarios[f"{scenario} Realized Capacity"] * mix_basic * (basic_metals_price * price_mult) +
        df_scenarios[f"{scenario} Realized Capacity"] * mix_advanced * (advanced_metals_price * price_mult) +
        df_scenarios[f"{scenario} Realized Capacity"] * mix_anion * (anion_test_price * price_mult)
    )
    # QA/QC cost and profit for scenario
    df_scenarios[f"{scenario} QA/QC Cost"] = df_scenarios[f"{scenario} Revenue"] * qa_qc_percentage
    df_scenarios[f"{scenario} Total Costs"] = fixed_costs + df_scenarios[f"{scenario} QA/QC Cost"]
    df_scenarios[f"{scenario} Profit"] = df_scenarios[f"{scenario} Revenue"] - df_scenarios[f"{scenario} Total Costs"]

# Show scenario comparison for the final projection month
final_month = df_scenarios["Month"].iloc[-1]
st.markdown(f"### Scenario Comparison for {final_month.strftime('%B %Y')}")
scenario_summary = pd.DataFrame({
    "Scenario": list(scenario_params.keys()),
    "Realized Capacity": [
        df_scenarios[f"{s} Realized Capacity"].iloc[-1] for s in scenario_params
    ],
    "Revenue ($)": [
        df_scenarios[f"{s} Revenue"].iloc[-1] for s in scenario_params
    ],
    "Profit ($)": [
        df_scenarios[f"{s} Profit"].iloc[-1] for s in scenario_params
    ]
})
st.table(scenario_summary)

# --- Monte Carlo Simulation Section ---
st.markdown("## 🎲 Monte Carlo Simulation")
st.markdown("""
This simulation assesses the impact of uncertainty in the realized capacity and pricing.  
For the final projection month, we sample:
- **Realized Capacity Factor:** Mean = 0.75 (Moderate scenario)  
- **Price Multiplier:** Mean = 1.0 (Moderate scenario)  
Use the sliders to adjust the uncertainty (standard deviation) and number of simulation runs.
""")

# Monte Carlo simulation parameters
mc_runs = st.number_input("Number of Simulation Runs", min_value=100, value=1000, step=100)
cap_std = st.slider("Capacity Factor Std Dev", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
price_std = st.slider("Price Multiplier Std Dev", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

# For simulation, use the computed maximum capacity from the final month
final_capacity = df["Total Capacity"].iloc[-1]
# Base mix: use the weighted average price per sample (weighted by mix ratios)
base_avg_price = (mix_basic * basic_metals_price +
                  mix_advanced * advanced_metals_price +
                  mix_anion * anion_test_price)

# Arrays to store simulation outputs
sim_revenues = []
sim_profits = []

# Fixed cost remains the same for final month (QA/QC will be computed per simulation)
for _ in range(mc_runs):
    # Sample realized capacity factor and price multiplier from normal distributions
    cap_factor_sample = np.random.normal(loc=0.75, scale=cap_std)
    price_mult_sample = np.random.normal(loc=1.0, scale=price_std)
    # Ensure factors remain non-negative
    cap_factor_sample = max(cap_factor_sample, 0)
    price_mult_sample = max(price_mult_sample, 0)
    
    realized_capacity = final_capacity * cap_factor_sample
    # Compute revenue using the realized capacity and sampled price multiplier for all tests
    revenue = realized_capacity * (mix_basic * (basic_metals_price * price_mult_sample) +
                                   mix_advanced * (advanced_metals_price * price_mult_sample) +
                                   mix_anion * (anion_test_price * price_mult_sample))
    qa_cost = revenue * qa_qc_percentage
    total_costs = fixed_costs + qa_cost
    profit = revenue - total_costs
    
    sim_revenues.append(revenue)
    sim_profits.append(profit)

# Create DataFrame for simulation results
mc_df = pd.DataFrame({
    "Revenue": sim_revenues,
    "Profit": sim_profits
})

st.markdown("### Monte Carlo Simulation Results for Final Month")
st.write(mc_df.describe())

# Plot histograms for Revenue and Profit
fig_mc_rev = px.histogram(mc_df, x="Revenue", nbins=30,
                          title="Revenue Distribution (Monte Carlo Simulation)",
                          labels={"Revenue": "Revenue ($)"})
st.plotly_chart(fig_mc_rev, use_container_width=True)

fig_mc_profit = px.histogram(mc_df, x="Profit", nbins=30,
                             title="Profit Distribution (Monte Carlo Simulation)",
                             labels={"Profit": "Profit ($)"})
st.plotly_chart(fig_mc_profit, use_container_width=True)

st.markdown("""
#### Insights from Monte Carlo Simulation:
- The simulation provides a probability distribution for revenue and profit in the final projection month.  
- Adjust the uncertainty parameters to see how variability in operational efficiency and pricing impacts your financial outcomes.
""")
