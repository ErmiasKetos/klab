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
This app forecasts monthly lab capacity, revenue, and profit. It compares different growth scenarios and runs Monte Carlo simulations to capture uncertainty in lab performance and pricing.
""")

# -----------------------------
# Helper Functions
# -----------------------------
def compute_test_metrics(test_data, volumes, comp_reporting):
    """Compute total test revenue and total required testing hours."""
    total_rev = 0.0
    total_hours = 0.0
    breakdown = []
    for test, config in test_data.items():
        vol = volumes.get(test, 0)
        hours_needed = config["hours_per_sample"] * vol
        total_hours += hours_needed
        price = config["base_price"] * config["tat_multiplier"]
        if comp_reporting:
            price *= 1.15
        rev = price * vol
        total_rev += rev
        breakdown.append({
            "Test": test,
            "Samples": vol,
            "Hours per Sample": config["hours_per_sample"],
            "Total Hours": hours_needed,
            "Price per Sample ($)": round(price, 2),
            "Revenue ($)": round(rev, 2)
        })
    return total_rev, total_hours, pd.DataFrame(breakdown)

def compute_premium_revenue(reg_consult, onsite_miles):
    """Compute revenue from premium services."""
    premium_rev = reg_consult * 200.0
    if onsite_miles > 0:
        premium_rev += 150.0 + 2.0 * onsite_miles
    return premium_rev

def scenario_results(base_test_rev, premium_rev, fixed_cost, cost_mult, cap_factor, price_mult):
    """Compute scenario-adjusted revenue and profit."""
    scen_rev = (base_test_rev * cap_factor * price_mult) + premium_rev
    scen_fixed = fixed_cost * cost_mult
    scen_profit = scen_rev - scen_fixed
    return scen_rev, scen_profit

# -----------------------------
# Sidebar: Input Parameters
# -----------------------------

st.sidebar.header("🔧 Input Parameters")

# Staffing & Capacity (Assuming 40 hrs/week per analyst)
with st.sidebar.expander("Staffing & Capacity"):
    st.markdown("Assuming each analyst works **40 hrs/week**.")
    analysts_phase1 = st.number_input("Analysts (Phase 1 - 2025)", min_value=1, value=3, step=1)
    analysts_phase2 = st.number_input("Analysts (Phase 2 - 2026)", min_value=1, value=6, step=1)
    analysts_phase3 = st.number_input("Analysts (Phase 3 - 2027)", min_value=1, value=8, step=1)
    # Automatically compute available hours for Phase 1 (base case)
    base_weekly_hours = analysts_phase1 * 40
    monthly_available_hours = base_weekly_hours * 4
    st.markdown(f"**Base Weekly Hours:** {base_weekly_hours} hrs")
    st.markdown(f"**Monthly Available Hours (Phase 1):** {monthly_available_hours} hrs")
    capacity_per_analyst = st.number_input("Capacity per Analyst (Samples/Month)", min_value=10, value=100, step=10)
    shift_factor = st.slider("Shift Expansion Factor (Phase 2)", 1.0, 3.0, 1.5, 0.1)
    automation_factor = st.slider("AI Automation Factor (Phase 3)", 1.0, 5.0, 2.0, 0.1)

# Fixed Costs
with st.sidebar.expander("Fixed Costs"):
    fixed_cost = st.number_input("Monthly Fixed Costs ($)", value=34232.45, step=100.0)
    cost_overrun_mult = st.slider("Cost Overrun Multiplier", 1.0, 2.0, 1.0, 0.05)

# Test Pricing & Turnaround
with st.sidebar.expander("Test Pricing & Turnaround"):
    st.markdown("Define tests with base prices and turnaround time multipliers. The 'hours per sample' reflect the time needed to run a test (from your attached file).")
    tat_options = {"Standard (5-7 days)": 1.0, "4-day TAT (1.5x)": 1.5, "2-day TAT (2x)": 2.0}
    # Define test details (base price in $, hours per sample)
    default_tests = {
        "Basic Heavy Metal Test": {"base_price": 300.0, "hours": 1.0},
        "Advanced Heavy Metal Test": {"base_price": 400.0, "hours": 1.5},
        "Anion Test": {"base_price": 150.0, "hours": 0.5}
    }
    test_data = {}
    for test, info in default_tests.items():
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
with st.sidebar.expander("Premium Services"):
    comp_reporting = st.checkbox("Comprehensive Reporting (+15%)", value=False)
    reg_consult = st.number_input("Regulatory Consultation (hrs @ $200/hr)", min_value=0, value=0, step=1)
    onsite_miles = st.number_input("On-site Sampling (miles @ $2/mile + $150 base)", min_value=0, value=0, step=5)

# Monthly Test Volumes
with st.sidebar.expander("Monthly Test Volumes"):
    monthly_volumes = {}
    for test in test_data.keys():
        monthly_volumes[test] = st.number_input(f"{test} - Samples/Month", min_value=0, value=0, step=10, key=f"{test}_vol")

# Monte Carlo Simulation Parameters
with st.sidebar.expander("Monte Carlo Simulation"):
    mc_runs = st.number_input("Simulation Runs", min_value=100, value=1000, step=100)
    cap_std = st.slider("Capacity Factor Std Dev", 0.0, 0.5, 0.1, 0.01)
    price_std = st.slider("Price Multiplier Std Dev", 0.0, 0.5, 0.1, 0.01)

# -----------------------------
# Base Model Computation
# -----------------------------
# Compute base test revenue, required testing hours, and get detailed breakdown
base_test_revenue, total_required_hours, df_test_breakdown = compute_test_metrics(test_data, monthly_volumes, comp_reporting)
premium_rev = compute_premium_revenue(reg_consult, onsite_miles)
total_revenue = base_test_revenue + premium_rev
total_fixed = fixed_cost * cost_overrun_mult
base_profit = total_revenue - total_fixed

# Use computed monthly available hours (from Phase 1 staffing)
capacity_status = "OK" if total_required_hours <= monthly_available_hours else "Over Capacity"

# -----------------------------
# Display Base Results
# -----------------------------
st.header("📊 Base Monthly Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Required Hours", f"{int(total_required_hours)} hrs")
col2.metric("Monthly Available Hours", f"{int(monthly_available_hours)} hrs")
col3.metric("Capacity Status", capacity_status)
col4.metric("Net Profit", f"${base_profit:,.2f}")

st.subheader("Financial Breakdown")
df_financial = pd.DataFrame({
    "Item": ["Test Revenue", "Premium Revenue", "Total Revenue", "Fixed Costs", "Net Profit"],
    "Amount ($)": [round(base_test_revenue,2), round(premium_rev,2), round(total_revenue,2),
                   round(total_fixed,2), round(base_profit,2)]
})
st.table(df_financial)

st.subheader("Detailed Test Breakdown")
st.table(df_test_breakdown)

# -----------------------------
# Scenario Comparison
# -----------------------------
st.header("🔍 Scenario Comparison")
st.markdown("""
The following scenarios adjust key parameters to simulate different growth conditions:
- **Conservative:** Lower capacity & discounted pricing.
- **Moderate:** Base case.
- **Optimistic:** Higher capacity & premium pricing.
- **Contingency:** Unforeseen challenges (reduced capacity, lower pricing, higher costs).
""")
scenario_params = {
    "Conservative": {"cap_factor": 0.5, "price_mult": 0.9, "cost_mult": 1.0},
    "Moderate": {"cap_factor": 0.75, "price_mult": 1.0, "cost_mult": 1.0},
    "Optimistic": {"cap_factor": 1.0, "price_mult": 1.1, "cost_mult": 1.0},
    "Contingency": {"cap_factor": 0.6, "price_mult": 0.9, "cost_mult": 1.2},
}
scenarios = []
for name, params in scenario_params.items():
    rev, profit = scenario_results(base_test_revenue, premium_rev, total_fixed,
                                   params["cost_mult"], params["cap_factor"], params["price_mult"])
    scenarios.append({
        "Scenario": name,
        "Adjusted Revenue ($)": round(rev,2),
        "Adjusted Profit ($)": round(profit,2)
    })
df_scenarios = pd.DataFrame(scenarios)
st.table(df_scenarios)

# -----------------------------
# Monte Carlo Simulation
# -----------------------------
st.header("🎲 Monte Carlo Simulation")
st.markdown("""
This simulation captures uncertainty in realized capacity and pricing for the final month.
For each simulation run, the following parameters are sampled:
- **Capacity Factor:** (mean = 0.75, adjustable std dev)
- **Price Multiplier:** (mean = 1.0, adjustable std dev)
""")
sim_revenues = []
sim_profits = []
for _ in range(mc_runs):
    cap_factor_sample = max(np.random.normal(0.75, cap_std), 0)
    price_mult_sample = max(np.random.normal(1.0, price_std), 0)
    sim_test_rev = base_test_revenue * cap_factor_sample * price_mult_sample
    sim_total_rev = sim_test_rev + premium_rev
    sim_profit = sim_total_rev - total_fixed
    sim_revenues.append(sim_total_rev)
    sim_profits.append(sim_profit)

mc_df = pd.DataFrame({"Revenue": sim_revenues, "Profit": sim_profits})
st.subheader("Monte Carlo Simulation Results")
st.write(mc_df.describe())

fig_rev = px.histogram(mc_df, x="Revenue", nbins=30, title="Revenue Distribution")
st.plotly_chart(fig_rev, use_container_width=True)
fig_profit = px.histogram(mc_df, x="Profit", nbins=30, title="Profit Distribution")
st.plotly_chart(fig_profit, use_container_width=True)

st.markdown("""
**Insights:**  
- The simulation provides a range of potential revenue and profit outcomes under uncertainty.  
- Adjust the standard deviation parameters to evaluate risk and operational resilience.
""")
