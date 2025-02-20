import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. SCENARIO DEFINITIONS
#    Here we define multipliers for three scenarios: Conservative, Moderate, Optimistic.
#    - staff_multiplier: scales the number of analysts
#    - capacity_multiplier: scales capacity per analyst
#    - price_multiplier: scales test pricing
# -----------------------------------------------------------------------------
SCENARIOS = {
    "Conservative": {
        "staff_multiplier": 0.8,
        "capacity_multiplier": 0.9,
        "price_multiplier": 0.9
    },
    "Moderate": {
        "staff_multiplier": 1.0,
        "capacity_multiplier": 1.0,
        "price_multiplier": 1.0
    },
    "Optimistic": {
        "staff_multiplier": 1.2,
        "capacity_multiplier": 1.1,
        "price_multiplier": 1.1
    },
}

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def phase_label(date):
    """Return the phase label based on date."""
    # Phase 1: Apr 2025 - Dec 2025
    # Phase 2: Jan 2026 - Dec 2026
    # Phase 3: Jan 2027 - Dec 2027
    if date.year == 2025 and date.month >= 4:
        return "Phase 1"
    elif date.year == 2026:
        return "Phase 2"
    else:
        return "Phase 3"

def generate_timeline(assumptions, scenario_name="Moderate"):
    """
    Create a monthly timeline from Apr 2025 to Dec 2027.
    Calculate capacity, revenue, costs, and profit for each month.
    """
    scenario = SCENARIOS[scenario_name]
    
    # Generate monthly dates
    dates = pd.date_range(start="2025-04-01", end="2027-12-31", freq="MS")
    timeline = pd.DataFrame(index=dates)
    timeline.index.name = "Month"
    
    # Determine Phase
    timeline["Phase"] = timeline.index.map(phase_label)
    
    # Calculate capacity
    #   capacity_per_analyst * number_of_analysts
    #   multiplied by scenario capacity factor
    capacity_per_analyst = assumptions["base_capacity"] * scenario["capacity_multiplier"]
    
    # For each Phase, we might have different staff levels
    def staff_for_phase(phase):
        if phase == "Phase 1":
            return int(assumptions["phase1_staff"] * scenario["staff_multiplier"])
        elif phase == "Phase 2":
            return int(assumptions["phase2_staff"] * scenario["staff_multiplier"])
        else:  # Phase 3
            return int(assumptions["phase3_staff"] * scenario["staff_multiplier"])
    
    timeline["Staff"] = timeline["Phase"].apply(staff_for_phase)
    timeline["Monthly Capacity"] = timeline["Staff"] * capacity_per_analyst
    
    # Goal Tracking
    timeline["Goal Met?"] = timeline["Monthly Capacity"].apply(
        lambda x: "Goal Met" if x >= 1000 else "Under Target"
    )
    
    # Calculate Revenue
    # Weighted test price * monthly capacity
    # We'll apply a scenario multiplier on top of the base price.
    base_price = assumptions["avg_test_price"]  # e.g. $300
    scenario_price = base_price * scenario["price_multiplier"]
    timeline["Revenue"] = timeline["Monthly Capacity"] * scenario_price
    
    # Calculate Costs
    # Summation of fixed monthly costs + labor costs + QA/QC
    # You can expand or modify the logic to incorporate more detail.
    
    # Fixed costs (from your slides)
    # - Equipment Lease
    # - Instrument Running
    # - Labor
    # - Software/Licenses
    # - QA/QC is 8% of operational budget (example assumption)
    timeline["Equipment Lease"] = assumptions["equip_lease"]
    timeline["Instrument Running"] = assumptions["instrument_run"]
    
    # Labor cost scales with the number of staff
    # e.g., assumptions["base_labor"] is the cost for the nominal staff count
    # We scale it by the ratio of actual staff to the "base staff" used in the assumption
    base_staff_count = assumptions["phase1_staff"]  # Just an anchor
    timeline["Labor"] = assumptions["base_labor"] * (timeline["Staff"] / base_staff_count)
    
    timeline["Software/Licenses"] = assumptions["software"]
    
    # Sum up
    timeline["Total Operational"] = (
        timeline["Equipment Lease"] +
        timeline["Instrument Running"] +
        timeline["Labor"] +
        timeline["Software/Licenses"]
    )
    # QA/QC cost (example: 8% of operational budget)
    timeline["QA/QC"] = timeline["Total Operational"] * assumptions["qaqc_rate"]
    
    timeline["Total Cost"] = timeline["Total Operational"] + timeline["QA/QC"]
    # Profit
    timeline["Profit"] = timeline["Revenue"] - timeline["Total Cost"]
    
    return timeline

def render_base_visualizations(df, scenario_name):
    """Render line charts for Capacity, Revenue, and Profit."""
    st.subheader(f"Key Metrics - {scenario_name} Scenario")
    
    # Capacity Chart
    fig_cap = px.line(
        df.reset_index(),
        x="Month",
        y="Monthly Capacity",
        title="Monthly Capacity"
    )
    # Show 1,000-sample goal
    fig_cap.add_hline(y=1000, line_dash="dot", line_color="red")
    st.plotly_chart(fig_cap, use_container_width=True)
    
    # Revenue Chart
    fig_rev = px.line(
        df.reset_index(),
        x="Month",
        y="Revenue",
        title="Monthly Revenue ($)"
    )
    st.plotly_chart(fig_rev, use_container_width=True)
    
    # Profit Chart
    fig_profit = px.line(
        df.reset_index(),
        x="Month",
        y="Profit",
        title="Monthly Profit ($)"
    )
    st.plotly_chart(fig_profit, use_container_width=True)
    
    # Show table with "Goal Met?" for reference
    st.dataframe(
        df[["Phase", "Staff", "Monthly Capacity", "Goal Met?", "Revenue", "Total Cost", "Profit"]]
    )

def compare_scenarios(assumptions):
    """Compare the three scenarios side-by-side."""
    st.subheader("Scenario Comparison")
    
    combined = []
    for scenario_name in SCENARIOS.keys():
        timeline = generate_timeline(assumptions, scenario_name)
        timeline["Scenario"] = scenario_name
        combined.append(timeline)
    
    df_combined = pd.concat(combined)
    
    # Capacity Comparison
    fig_cap = px.line(
        df_combined.reset_index(),
        x="Month",
        y="Monthly Capacity",
        color="Scenario",
        title="Monthly Capacity Comparison"
    )
    fig_cap.add_hline(y=1000, line_dash="dot", line_color="red")
    st.plotly_chart(fig_cap, use_container_width=True)
    
    # Profit Comparison
    fig_profit = px.line(
        df_combined.reset_index(),
        x="Month",
        y="Profit",
        color="Scenario",
        title="Monthly Profit Comparison"
    )
    st.plotly_chart(fig_profit, use_container_width=True)
    
    # Show final profit by scenario
    summary = df_combined.groupby("Scenario").agg(
        Total_Revenue=("Revenue", "sum"),
        Total_Cost=("Total Cost", "sum"),
        Total_Profit=("Profit", "sum"),
        Peak_Capacity=("Monthly Capacity", "max")
    )
    st.dataframe(summary)

# -----------------------------------------------------------------------------
# 3. MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.title("Lab Scalability & Profit Modeling Tool")
    st.markdown(
        """
        This tool models lab capacity, revenue, and profit from **April 2025** 
        to **December 2027**, divided into three phases:
        - **Phase 1:** Apr 2025 – Dec 2025  
        - **Phase 2:** Jan 2026 – Dec 2026  
        - **Phase 3:** Jan 2027 – Dec 2027  

        It calculates monthly capacity based on the number of staff and per-analyst 
        capacity, then derives revenue using a weighted test price. Fixed costs 
        and QA/QC expenses are subtracted to obtain monthly profit. A 1,000 
        samples/month goal is tracked automatically.
        """
    )

    # --- Dynamic Input Controls ---
    st.sidebar.header("Model Assumptions")
    base_capacity = st.sidebar.number_input("Base Capacity (tests per analyst/month)", 100, 2000, 440)
    
    # Staff in each phase
    phase1_staff = st.sidebar.number_input("Staff - Phase 1 (2025)", 1, 10, 3)
    phase2_staff = st.sidebar.number_input("Staff - Phase 2 (2026)", 1, 15, 5)
    phase3_staff = st.sidebar.number_input("Staff - Phase 3 (2027+)", 1, 20, 5)
    
    # Pricing
    avg_test_price = st.sidebar.number_input("Average Test Price ($)", 50, 1000, 300)
    
    # Costs (monthly)
    equip_lease = st.sidebar.number_input("Equipment Lease ($/month)", 0, 50000, 10378)
    instrument_run = st.sidebar.number_input("Instrument Running ($/month)", 0, 10000, 2775)
    base_labor = st.sidebar.number_input("Base Labor Cost ($/month)", 0, 50000, 16667)
    software = st.sidebar.number_input("Software/Licenses ($/month)", 0, 20000, 2000)
    qaqc_rate = st.sidebar.slider("QA/QC Rate (% of operational)", 0.0, 0.2, 0.08)
    
    # Package assumptions
    assumptions = {
        "base_capacity": base_capacity,
        "phase1_staff": phase1_staff,
        "phase2_staff": phase2_staff,
        "phase3_staff": phase3_staff,
        
        "avg_test_price": avg_test_price,
        
        "equip_lease": equip_lease,
        "instrument_run": instrument_run,
        "base_labor": base_labor,
        "software": software,
        "qaqc_rate": qaqc_rate,
    }

    # --- Tabs ---
    tab1, tab2 = st.tabs(["Single Scenario View", "Compare Scenarios"])
    
    with tab1:
        # Let user pick which scenario to view in single-scenario mode
        scenario_choice = st.selectbox("Select Scenario", list(SCENARIOS.keys()))
        timeline = generate_timeline(assumptions, scenario_choice)
        render_base_visualizations(timeline, scenario_choice)
    
    with tab2:
        compare_scenarios(assumptions)

if __name__ == "__main__":
    main()
