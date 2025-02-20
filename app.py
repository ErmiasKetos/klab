import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Environmental Lab Growth & Resilience Simulator", layout="wide")
st.title("🌱 Environmental Lab Growth & Resilience Simulator")
st.markdown("""
This app provides a robust projection framework for an environmental testing lab. 
It compares different growth scenarios and runs Monte Carlo simulations to capture uncertainty in lab performance and pricing.
Adjust the inputs in the sidebar to see how operational or financial outcomes vary.
""")

# -----------------------------
# Sidebar: Input Parameters
# -----------------------------

st.sidebar.header("🔧 Base Input Parameters")

# Timeline Settings
start_date = "2025-04-01"    # Lab operations begin
end_date = "2027-12-01"      # Projection end date

# Staffing & Capacity Inputs
st.sidebar.subheader("Staffing & Capacity")
analysts_phase1 = st.sidebar.number_input("Analysts (Phase 1 - 2025)", min_value=1, value=3, step=1)
analysts_phase2 = st.sidebar.number_input("Analysts (Phase 2 - 2026)", min_value=1, value=6, step=1)
analysts_phase3 = st.sidebar.number_input("Analysts (Phase 3 - 2027)", min_value=1, value=8, step=1)
capacity_per_analyst = st.sidebar.number_input("Capacity per Analyst (Samples/Month)", min_value=10, value=100, step=10)
shift_factor = st.sidebar.slider("Shift Expansion Factor (Phase 2)", 1.0, 3.0, 1.5, 0.1)
automation_factor = st.sidebar.slider("AI Automation Factor (Phase 3)", 1.0, 5.0, 2.0, 0.1)

# Fixed Cost Structure (monthly)
st.sidebar.subheader("Fixed Costs")
fixed_cost = st.sidebar.number_input("Monthly Fixed Costs ($)", value=34232.45, step=100.0)
# Extra cost multiplier to account for unforeseen expenses
cost_overrun_mult = st.sidebar.slider("Cost Overrun Multiplier", min_value=1.0, max_value=2.0, value=1.0, step=0.05)

# Pricing Inputs for Tests & Turnaround Times
st.sidebar.subheader("Test Pricing & Turnaround")
# Define base tests: name, default base price (in $) and required hours per sample
test_info = {
    "Basic Heavy Metal Test": {"base_price": 300.0, "hours": 1.0},
    "Advanced Heavy Metal Test": {"base_price": 400.0, "hours": 1.5},
    "Anion Test": {"base_price": 150.0, "hours": 0.5}
}
# Turnaround Time Options (multiplier)
tat_options = {"Standard (5-7 days)": 1.0, "4-day TAT (1.5x)": 1.5, "2-day TAT (2x)": 2.0}
test_data = {}
for test, info in test_info.items():
    st.markdown(f"**{test}**")
    base_price = st.number_input(f"{test} - Base Price ($)", value=info["base_price"], step=50.0, key=f"{test}_price")
    tat_choice = st.selectbox(f"{test} - Turnaround Option", list(tat_options.keys()), key=f"{test}_tat")
    test_data[test] = {
        "base_price": base_price,
        "hours_per_sample": info["hours"],
        "tat_multiplier": tat_options[tat_choice]
    }
    st.divider()

# Premium Services
st.sidebar.subheader("Premium Services")
comp_reporting = st.checkbox("Comprehensive Reporting (+15%)", value=False)
reg_consult_hrs = st.number_input("Regulatory Consultation (hrs @ $200/hr)", min_value=0, value=0, step=1)
onsite_miles = st.number_input("On-site Sampling (miles @ $2/mile + $150 base)", min_value=0, value=0, step=5)

# Monthly Test Volumes
st.sidebar.subheader("Monthly Test Volumes")
monthly_volumes = {}
for test in test_data.keys():
    monthly_volumes[test] = st.number_input(f"{test} - Samples/Month", min_value=0, value=0, step=10, key=f"{test}_vol")

# Monte Carlo Simulation Parameters
st.sidebar.header("Monte Carlo Simulation")
mc_runs = st.sidebar.number_input("Number of Simulation Runs", min_value=100, value=1000, step=100)
cap_std = st.sidebar.slider("Capacity Factor Std Dev", 0.0, 0.5, 0.1, 0.01)
price_std = st.sidebar.slider("Price Multiplier Std Dev", 0.0, 0.5, 0.1, 0.01)

# -----------------------------
# Base Model: Timeline & Capacity Projection
# -----------------------------
months = pd.date_range(start=start_date, end=end_date, freq='MS')
df = pd.DataFrame({"Month": months})
df["Phase"] = np.where(df["Month"].dt.year == 2025, "Phase 1",
                np.where(df["Month"].dt.year == 2026, "Phase 2", "Phase 3"))

# Staffing assignment by phase
df["Analysts"] = np.where(df["Phase"] == "Phase 1", analysts_phase1,
                   np.where(df["Phase"] == "Phase 2", analysts_phase2, analysts_phase3))
# Adjust capacity per analyst per phase
df["Capacity per Analyst"] = np.where(df["Phase"] == "Phase 1", capacity_per_analyst,
                               np.where(df["Phase"] == "Phase 2", capacity_per_analyst * shift_factor,
                                        capacity_per_analyst * automation_factor))
df["Total Capacity"] = df["Analysts"] * df["Capacity per Analyst"]

# -----------------------------
# Base Revenue & Profit Calculations
# -----------------------------
# Calculate revenue per test type based on sample volumes and TAT pricing
test_revenue = 0.0
total_required_hours = 0.0

for test, config in test_data.items():
    volume = monthly_volumes[test]
    # Hours required
    hours_needed = config["hours_per_sample"] * volume
    total_required_hours += hours_needed
    
    # Adjust price by turnaround multiplier and optional reporting fee
    price = config["base_price"] * config["tat_multiplier"]
    if comp_reporting:
        price *= 1.15
    test_revenue += price * volume

# Premium services revenue
premium_revenue = reg_consult_hrs * 200.0
if onsite_miles > 0:
    premium_revenue += 150.0 + 2.0 * onsite_miles

total_revenue = test_revenue + premium_revenue

# Apply cost overrun multiplier
total_fixed_costs = fixed_cost * cost_overrun_mult

# Assume profit = revenue - fixed costs
base_profit = total_revenue - total_fixed_costs

# -----------------------------
# Capacity Check (Assuming monthly available hours from staffing)
# -----------------------------
# Assume each staff works ~40 hours/week; total monthly available = sum of staff weekly hours * 4
staff_weekly_hours = st.sidebar.number_input("Total Staff Hours/Week (if different from above)", min_value=0, value=100, step=1)
monthly_available_hours = staff_weekly_hours * 4
capacity_status = "OK" if total_required_hours <= monthly_available_hours else "Over Capacity"

# -----------------------------
# Display Base Results
# -----------------------------
st.header("📊 Base Monthly Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Required Hours", f"{int(total_required_hours)} hrs")
col2.metric("Available Hours", f"{int(monthly_available_hours)} hrs")
col3.metric("Capacity Status", capacity_status)
col4.metric("Net Profit", f"${base_profit:,.2f}")

st.subheader("Detailed Financial Breakdown")
df_financial = pd.DataFrame({
    "Item": ["Test Revenue", "Premium Services", "Total Revenue", "Fixed Costs", "Net Profit"],
    "Amount ($)": [round(test_revenue,2), round(premium_revenue,2), round(total_revenue,2), round(total_fixed_costs,2), round(base_profit,2)]
})
st.table(df_financial)

# -----------------------------
# Scenario Comparison
# -----------------------------
st.header("🔍 Scenario Comparison")
st.markdown("""
The following scenarios adjust key parameters to simulate different growth conditions:
- **Conservative:** Lower capacity and discounted pricing.
- **Moderate:** Base case.
- **Optimistic:** Higher capacity and premium pricing.
- **Contingency:** Unforeseen challenges (reduced capacity, lower pricing, higher costs).
""")

scenario_params = {
    "Conservative": {"capacity_factor": 0.5, "price_multiplier": 0.9, "cost_multiplier": 1.0},
    "Moderate": {"capacity_factor": 0.75, "price_multiplier": 1.0, "cost_multiplier": 1.0},
    "Optimistic": {"capacity_factor": 1.0, "price_multiplier": 1.1, "cost_multiplier": 1.0},
    "Contingency": {"capacity_factor": 0.6, "price_multiplier": 0.9, "cost_multiplier": 1.2},
}

scenario_summary = []
for scenario, params in scenario_params.items():
    # Adjust revenue based on capacity and pricing factors
    scenario_revenue = test_revenue * params["capacity_factor"] * params["price_multiplier"]
    scenario_fixed = total_fixed_costs * params["cost_multiplier"]
    scenario_profit = scenario_revenue + premium_revenue - scenario_fixed
    scenario_summary.append({
        "Scenario": scenario,
        "Adjusted Revenue ($)": round(scenario_revenue + premium_revenue, 2),
        "Adjusted Profit ($)": round(scenario_profit, 2)
    })

df_scenarios = pd.DataFrame(scenario_summary)
st.table(df_scenarios)

# -----------------------------
# Monte Carlo Simulation
# -----------------------------
st.header("🎲 Monte Carlo Simulation")
st.markdown("""
This simulation models uncertainty in realized capacity and pricing for the final projection month.
For each simulation run, the following parameters are randomly sampled:
- **Capacity Factor:** (mean = 0.75, adjustable std dev)
- **Price Multiplier:** (mean = 1.0, adjustable std dev)
""")

final_capacity = df["Total Capacity"].iloc[-1]
sim_revenues = []
sim_profits = []
for _ in range(mc_runs):
    # Sample random factors (ensuring non-negative)
    cap_factor_sample = max(np.random.normal(0.75, cap_std), 0)
    price_mult_sample = max(np.random.normal(1.0, price_std), 0)
    
    # Adjust revenue for tests (premium services remain constant)
    sim_test_rev = test_revenue * cap_factor_sample * price_mult_sample
    sim_total_rev = sim_test_rev + premium_revenue
    sim_profit = sim_total_rev - total_fixed_costs
    sim_revenues.append(sim_total_rev)
    sim_profits.append(sim_profit)

mc_df = pd.DataFrame({"Revenue": sim_revenues, "Profit": sim_profits})
st.markdown("### Monte Carlo Simulation Results (Final Month)")
st.write(mc_df.describe())

fig_mc_rev = px.histogram(mc_df, x="Revenue", nbins=30, title="Revenue Distribution (Monte Carlo)")
st.plotly_chart(fig_mc_rev, use_container_width=True)

fig_mc_profit = px.histogram(mc_df, x="Profit", nbins=30, title="Profit Distribution (Monte Carlo)")
st.plotly_chart(fig_mc_profit, use_container_width=True)

st.markdown("""
#### Key Insights from Monte Carlo Simulation:
- The simulation shows the range and probability distribution of potential revenue and profit outcomes.
- Adjust the uncertainty parameters to see how sensitive your financials are to fluctuations in capacity and pricing.
- This tool helps you plan for unforeseen circumstances and ensures your lab is resilient in varying market conditions.
""")
