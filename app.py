import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Environmental Lab Growth & Resilience Simulator (Enhanced)",
    layout="wide"
)

# -----------------------------
# Page Title
# -----------------------------
st.title("🌱 Environmental Lab Growth & Resilience Simulator (Enhanced)")
st.markdown("""
This model helps you forecast **monthly capacity, costs, and profitability** for an environmental testing lab. 
Adjust the inputs to see real-time updates on **lab throughput** and **financial performance**.
""")

# -----------------------------
# Sidebar / Input Configuration
# -----------------------------

# 1) Staff & Capacity
with st.sidebar.expander("1) Staffing & Capacity"):
    st.markdown("#### Staff Hours/Week")
    tech_mgr_hours = st.number_input("Technical Manager (Testing hrs/week)", min_value=0, value=10, step=1)
    analyst1_hours = st.number_input("Analyst #1 (hrs/week)", min_value=0, value=40, step=1)
    analyst2_hours = st.number_input("Analyst #2 (hrs/week)", min_value=0, value=40, step=1)
    parttime_hours = st.number_input("Part-time Scientist (hrs/week)", min_value=0, value=20, step=1)
    
    # Convert weekly hours to monthly (approx 4x)
    total_weekly_hours = tech_mgr_hours + analyst1_hours + analyst2_hours + parttime_hours
    monthly_hours_available = total_weekly_hours * 4
    
    st.markdown(f"**Total Testing Hours/Month:** {monthly_hours_available}")

# 2) Cost Structure
with st.sidebar.expander("2) Monthly Fixed Costs"):
    st.markdown("From the attached cost table, a typical monthly total is **$34,232.45**, which includes:")
    st.markdown("- Equipment Lease: $10,378.19\n- Instrument Running: $2,775.00\n- Labor: $16,667.00\n- Software/Licenses: $2,000.00\n- QA/QC portion: $2,412.26")
    fixed_monthly_cost = st.number_input("Fixed Monthly Cost (Total)", value=34232.45, step=100.0)
    st.markdown("Adjust this if your overhead changes (e.g., lease terms, consumables, staffing).")

# 3) Pricing & Turnaround Times
with st.sidebar.expander("3) Test Pricing & TAT Multipliers"):
    st.markdown("**Base Prices** (per sample) & TAT multipliers (2-day or 4-day) from your pricing table:")
    
    # For TAT, we can store multipliers in a dict
    tat_options = {"Standard (5-7 days)": 1.0, "4-day TAT (1.5x)": 1.5, "2-day TAT (2x)": 2.0}
    
    # We'll gather test info in a dictionary
    # Key = test name, value = (default_base_price, default_hours_per_sample)
    test_info = {
        "Basic Heavy Metal Test": {"base_price": 300, "hours": 1.0},
        "Advanced Heavy Metal Test": {"base_price": 400, "hours": 1.5},
        "Anion Test": {"base_price": 150, "hours": 0.5},
        "Dissolved Silica": {"base_price": 100, "hours": 0.5},
        "Calcium & Iron": {"base_price": 100, "hours": 0.5},
        "Hardness": {"base_price": 100, "hours": 0.5},
        "Specific Conductance": {"base_price": 50, "hours": 0.5},
        "pH": {"base_price": 50, "hours": 0.5},
        "TDS": {"base_price": 50, "hours": 0.5},
        "Boron": {"base_price": 100, "hours": 1.0},
    }
    
    # Let user pick TAT multiplier for each test
    test_data = {}
    for test_name, info in test_info.items():
        st.markdown(f"**{test_name}**")
        # Convert default base price to float so that value and step are the same type
        base_price = st.number_input(
            f"{test_name} - Base Price",
            value=float(info["base_price"]),
            step=50.0,
            key=f"{test_name}_price"
        )
        tat_choice = st.selectbox(
            f"{test_name} - Turnaround Time",
            list(tat_options.keys()),
            key=f"{test_name}_tat"
        )
        test_data[test_name] = {
            "base_price": base_price,
            "hours_per_sample": info["hours"],
            "tat_multiplier": tat_options[tat_choice]
        }
        st.divider()

# 4) Premium Services
with st.sidebar.expander("4) Premium Services"):
    comp_reporting = st.checkbox("Comprehensive Reporting (+15%)", value=False)
    reg_consult_hours = st.number_input("Regulatory Consultation (hrs @ $200/hr)", min_value=0, value=0, step=1)
    onsite_miles = st.number_input("On-site Sampling (miles @ $2/mile + $150 base)", min_value=0, value=0, step=5)

# 5) Monthly Test Volumes
with st.sidebar.expander("5) Monthly Test Volumes"):
    st.markdown("Enter how many samples you expect **per test type** each month:")
    monthly_volumes = {}
    for test_name in test_data.keys():
        monthly_volumes[test_name] = st.number_input(
            f"{test_name} - # of Samples/Month",
            min_value=0,
            value=0,
            step=10,
            key=f"{test_name}_volume"
        )

# -----------------------------
# Main Calculation & Results
# -----------------------------
st.header("📊 Monthly Results")

# 1) Calculate total required hours and revenue
total_required_hours = 0.0
test_revenue = 0.0

for test_name, config in test_data.items():
    volume = monthly_volumes[test_name]
    # Hours needed
    hours_for_test = config["hours_per_sample"] * volume
    total_required_hours += hours_for_test
    
    # Price (base * TAT)
    tat_price = config["base_price"] * config["tat_multiplier"]
    
    # Add 15% if "Comprehensive Reporting" is checked
    if comp_reporting:
        tat_price *= 1.15
    
    # Revenue from this test
    test_revenue += tat_price * volume

# 2) Add Premium Services revenue
#    - Regulatory Consultation: $200/hr
#    - On-site sampling: $150 + $2/mile
premium_services_revenue = 0.0
premium_services_revenue += reg_consult_hours * 200.0
if onsite_miles > 0:
    premium_services_revenue += 150.0 + 2.0 * onsite_miles

# 3) Total Revenue
total_revenue = test_revenue + premium_services_revenue

# 4) Check capacity
capacity_status = "OK"
if total_required_hours > monthly_hours_available:
    capacity_status = "Over Capacity"

# 5) Compute Profit
profit = total_revenue - fixed_monthly_cost

# Display Key Results
col1, col2, col3, col4 = st.columns(4)
col1.metric("Monthly Hours Required", f"{int(total_required_hours)} hrs")
col2.metric("Available Hours", f"{int(monthly_hours_available)} hrs")
col3.metric("Capacity Status", capacity_status)
col4.metric("Profit (Monthly)", f"${profit:,.2f}")

# Detailed financial breakdown
st.subheader("Detailed Financial Breakdown")
df_financial = pd.DataFrame({
    "Description": [
        "Test Revenue (All Tests)",
        "Premium Services Revenue",
        "Total Revenue",
        "Fixed Monthly Costs",
        "Net Profit"
    ],
    "Amount ($)": [
        round(test_revenue, 2),
        round(premium_services_revenue, 2),
        round(total_revenue, 2),
        round(fixed_monthly_cost, 2),
        round(profit, 2)
    ]
})
st.table(df_financial)

# If capacity is exceeded, show a warning
if capacity_status == "Over Capacity":
    st.warning(
        f"You require **{int(total_required_hours)} hours** but only have **{int(monthly_hours_available)} hours** available. "
        "Consider adding staff, reducing test volume, or extending turnaround times."
    )

# Plot a simple bar chart for test volume vs. revenue contribution
chart_data = []
for test_name, config in test_data.items():
    volume = monthly_volumes[test_name]
    tat_price = config["base_price"] * config["tat_multiplier"]
    if comp_reporting:
        tat_price *= 1.15
    revenue_for_test = tat_price * volume
    
    chart_data.append({
        "Test Type": test_name,
        "Monthly Samples": volume,
        "Revenue ($)": revenue_for_test
    })

df_chart = pd.DataFrame(chart_data)

st.subheader("Revenue by Test Type")
fig = px.bar(df_chart, x="Test Type", y="Revenue ($)", color="Test Type",
             title="Monthly Revenue by Test Type",
             text="Revenue ($)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Note**: This model calculates revenue based on **sample volume, TAT pricing multipliers, and optional premium services**. 
It compares total required hours to **staff capacity**.  
You can adjust all inputs in the sidebar to explore different operational or financial outcomes.
""")
