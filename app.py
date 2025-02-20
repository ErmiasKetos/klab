import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set Streamlit app title
st.set_page_config(page_title="Lab Scalability & Growth Model", layout="wide")

# Sidebar: User Inputs
st.sidebar.header("Scalability Model Inputs")

start_year = 2025
end_year = 2027
months = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq='MS')

# User-defined parameters
analysts_phase1 = st.sidebar.number_input("Analysts in Phase 1 (2025)", min_value=1, value=3, step=1)
analysts_phase2 = st.sidebar.number_input("Analysts in Phase 2 (2026)", min_value=1, value=6, step=1)
analysts_phase3 = st.sidebar.number_input("Analysts in Phase 3 (2027)", min_value=1, value=8, step=1)

capacity_per_analyst = st.sidebar.number_input("Capacity per Analyst per Month", min_value=10, value=100, step=10)
shift_expansion_factor = st.sidebar.slider("Shift Expansion Factor (Phase 2)", 1.0, 3.0, 1.5, 0.1)
automation_factor = st.sidebar.slider("AI Automation Factor (Phase 3)", 1.0, 5.0, 2.0, 0.1)

# Initialize dataframe
df = pd.DataFrame({"Month": months})

# Assign phases based on timeline
df["Phase"] = np.where(df["Month"].dt.year < 2026, "Phase 1",
                np.where(df["Month"].dt.year < 2027, "Phase 2", "Phase 3"))

# Assign number of analysts based on phase
df["Analysts"] = np.where(df["Phase"] == "Phase 1", analysts_phase1,
                  np.where(df["Phase"] == "Phase 2", analysts_phase2, analysts_phase3))

# Assign capacity per analyst based on phase
df["Capacity per Analyst"] = np.where(df["Phase"] == "Phase 1", capacity_per_analyst,
                             np.where(df["Phase"] == "Phase 2", capacity_per_analyst * shift_expansion_factor,
                                      capacity_per_analyst * automation_factor))

# Compute total capacity
df["Total Capacity"] = df["Analysts"] * df["Capacity per Analyst"]

# Check if the target of 1,000 samples/month is met
df["Goal Status"] = np.where(df["Total Capacity"] >= 1000, "Goal Met", "Under Target")

# Display results
st.title("Lab Scalability & Growth Projection")
st.markdown("📊 **Dynamic Model for Lab Expansion (2025-2027)**")

# Show key summary metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Analysts in 2025", analysts_phase1)
col2.metric("Total Analysts in 2026", analysts_phase2)
col3.metric("Total Analysts in 2027", analysts_phase3)

st.write("### 📅 Monthly Capacity Projection Table")
st.dataframe(df.style.applymap(lambda x: "background-color: #90EE90" if x == "Goal Met" else "", subset=["Goal Status"]))

# 📈 Visualization: Capacity Growth Over Time
fig = px.line(df, x="Month", y="Total Capacity", color="Phase",
              title="📈 Lab Capacity Growth Over Time",
              labels={"Total Capacity": "Samples Processed Per Month"},
              markers=True)

fig.add_hline(y=1000, line_dash="dot", line_color="red", annotation_text="Target: 1,000 Samples/Month")

st.plotly_chart(fig, use_container_width=True)

# 📊 Bar Chart: Capacity by Year
df["Year"] = df["Month"].dt.year
yearly_capacity = df.groupby("Year")["Total Capacity"].sum().reset_index()

fig2 = px.bar(yearly_capacity, x="Year", y="Total Capacity", text_auto=True,
              title="📊 Annual Sample Capacity Projection",
              labels={"Total Capacity": "Total Samples Processed Per Year"})

st.plotly_chart(fig2, use_container_width=True)

# 🎯 Highlight Goal Achievements
st.write("### 🎯 Goal Tracking: 1,000 Samples/Month")
goal_met_month = df[df["Goal Status"] == "Goal Met"].iloc[0]["Month"] if "Goal Met" in df["Goal Status"].values else "Not Achieved"
st.success(f"✅ The lab is projected to reach 1,000 samples/month in **{goal_met_month.strftime('%B %Y') if isinstance(goal_met_month, pd.Timestamp) else goal_met_month}**.")

# 🚀 Final Notes
st.markdown("""
💡 **How this model helps?**  
This tool dynamically models **staffing, shift expansion, and automation** to scale lab capacity.  
You can adjust inputs to simulate different growth strategies for optimizing your environmental testing lab.  
""")

st.sidebar.markdown("📌 **Adjust inputs to see real-time impact on lab scalability!**")
