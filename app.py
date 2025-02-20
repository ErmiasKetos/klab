import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- App Config ---
st.set_page_config(page_title="Lab Scalability Model", layout="wide")

# --- Helper Functions ---
def calculate_capacity(row, assumptions):
    if row['Phase'] == 'Phase 1':
        return assumptions['base_capacity'] * assumptions['phase1_staff']
    elif row['Phase'] == 'Phase 2':
        return (assumptions['base_capacity'] * assumptions['phase2_staff'] 
                * assumptions['shift_multiplier'])
    elif row['Phase'] == 'Phase 3':
        return (assumptions['base_capacity'] * assumptions['phase3_staff']
                * assumptions['shift_multiplier'] 
                * (1 + assumptions['ai_efficiency']))

# --- Model Logic ---
def generate_timeline(assumptions):
    dates = pd.date_range(
        start=assumptions['phase1_start'],
        end=assumptions['phase3_end'],
        freq='MS'
    )
    
    timeline = pd.DataFrame(index=dates)
    timeline.index.name = 'Month'
    
    # Assign Phases
    timeline['Phase'] = 'Phase 1'
    timeline.loc[timeline.index >= assumptions['phase2_start'], 'Phase'] = 'Phase 2'
    timeline.loc[timeline.index >= assumptions['phase3_start'], 'Phase'] = 'Phase 3'
    
    # Calculate Metrics
    timeline['Monthly Capacity'] = timeline.apply(
        lambda x: calculate_capacity(x, assumptions), axis=1)
    timeline['Cumulative Samples'] = timeline['Monthly Capacity'].cumsum()
    
    # Calculate Costs
    timeline['Staff Costs'] = timeline['Phase'].map({
        'Phase 1': assumptions['phase1_staff'] * assumptions['salary'],
        'Phase 2': assumptions['phase2_staff'] * assumptions['salary'],
        'Phase 3': assumptions['phase3_staff'] * assumptions['salary']
    })
    
    timeline['AI Costs'] = (timeline['Phase'] == 'Phase 3') * assumptions['ai_cost']
    
    return timeline

# --- UI Components ---
def input_assumptions():
    with st.sidebar:
        st.header("Model Assumptions")
        
        assumptions = {
            'base_capacity': st.number_input("Base Capacity/Analyst/Month", 100),
            'phase1_staff': st.number_input("Phase 1 Staff", 3),
            'phase2_staff': st.number_input("Phase 2 Staff", 5),
            'phase3_staff': st.number_input("Phase 3 Staff", 5),
            'shift_multiplier': st.slider("Shift Multiplier (Phase 2+)", 1.0, 3.0, 2.0),
            'ai_efficiency': st.slider("AI Efficiency Boost", 0.0, 1.0, 0.3),
            'salary': st.number_input("Monthly Salary/Analyst ($)", 5000),
            'ai_cost': st.number_input("AI Setup Cost ($)", 50000)
        }
        
        st.subheader("Phase Dates")
        assumptions['phase1_start'] = st.date_input("Phase 1 Start", datetime(2025,1,1))
        assumptions['phase2_start'] = st.date_input("Phase 2 Start", datetime(2026,1,1))
        assumptions['phase3_start'] = st.date_input("Phase 3 Start", datetime(2027,1,1))
        assumptions['phase3_end'] = st.date_input("Model End Date", datetime(2027,12,31))
        
        return assumptions

# --- Main App ---
def main():
    st.title("Lab Scalability Modeling Tool")
    
    # Get Inputs
    assumptions = input_assumptions()
    
    # Generate Timeline
    timeline = generate_timeline(assumptions)
    
    # --- Visualizations ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Capacity Planning")
        fig = px.line(timeline, y='Monthly Capacity',
                     title="Monthly Testing Capacity")
        fig.add_hline(y=1000, line_dash="dot", line_color="red",
                     annotation_text="1,000 Sample Goal")
        st.plotly_chart(fig, use_container_width=True)
        
        # Phase Timeline
        phase_changes = timeline[timeline['Phase'] != timeline['Phase'].shift(1)]
        fig2 = px.timeline(phase_changes, x_start="Month", x_end="Month",
                          y="Phase", color="Phase")
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        st.subheader("Financials")
        
        # Cost Breakdown
        total_costs = timeline[['Staff Costs', 'AI Costs']].sum()
        fig3 = px.pie(values=total_costs, names=total_costs.index,
                     title="Total Cost Breakdown")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Key Metrics
        current_capacity = timeline['Monthly Capacity'].iloc[-1]
        months_to_goal = ((1000 - current_capacity) / 
                        timeline['Monthly Capacity'].mean())
        
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Current Monthly Capacity", f"{current_capacity:,.0f}")
        metric_col2.metric("Months to Reach Goal", 
                          f"{max(0, months_to_goal):.1f}" if current_capacity < 1000 else "Achieved")
        
    # Raw Data
    with st.expander("View Detailed Timeline Data"):
        st.dataframe(timeline.style.format("{:,.0f}"))

if __name__ == "__main__":
    main()
