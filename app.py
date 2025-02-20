import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Initialize session state for scenario management
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}

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
        assumptions['phase1_start'] = pd.to_datetime(st.date_input("Phase 1 Start", datetime(2025,1,1)))
        assumptions['phase2_start'] = pd.to_datetime(st.date_input("Phase 2 Start", datetime(2026,1,1)))
        assumptions['phase3_start'] = pd.to_datetime(st.date_input("Phase 3 Start", datetime(2027,1,1)))
        assumptions['phase3_end'] = pd.to_datetime(st.date_input("Model End Date", datetime(2027,12,31)))
        
        return assumptions

def scenario_management(assumptions):
    with st.sidebar:
        st.subheader("Scenario Management")
        
        # Save/Load Interface
        scenario_name = st.text_input("Scenario Name")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Current Settings"):
                if scenario_name:
                    save_scenario(assumptions.copy(), scenario_name)
        
        # Scenario Selection
        selected = st.multiselect(
            "Compare Scenarios",
            options=list(st.session_state.scenarios.keys()),
            default=list(st.session_state.scenarios.keys())[-1] if st.session_state.scenarios else []
        )
        
        # Delete Scenarios
        with col2:
            if st.button("🗑️ Delete Selected"):
                for name in selected:
                    delete_scenario(name)
        
        return selected

def save_scenario(assumptions, name):
    st.session_state.scenarios[name] = assumptions
    st.success(f"Scenario '{name}' saved!")

def delete_scenario(name):
    del st.session_state.scenarios[name]

def render_comparison(selected_scenarios):
    if len(selected_scenarios) < 1:
        return
    
    # Generate timelines for all selected scenarios
    all_data = []
    for name in selected_scenarios:
        timeline = generate_timeline(st.session_state.scenarios[name])
        timeline['Scenario'] = name
        all_data.append(timeline)
    
    combined = pd.concat(all_data)
    
    # Comparison Chart
    st.subheader("Scenario Comparison")
    fig = px.line(combined.reset_index(), x='Month', y='Monthly Capacity',
                 color='Scenario', line_dash='Scenario',
                 title="Capacity Across Scenarios")
    fig.add_hline(y=1000, line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics Table
    metrics = []
    for name in selected_scenarios:
        tl = generate_timeline(st.session_state.scenarios[name])
        metrics.append({
            'Scenario': name,
            'Peak Capacity': tl['Monthly Capacity'].max(),
            'Total Cost ($M)': (tl['Staff Costs'].sum() + tl['AI Costs'].sum()) / 1e6,
            'Goal Achieved Date': tl[tl['Monthly Capacity'] >= 1000].index[0] if any(tl['Monthly Capacity'] >= 1000) else "Not Reached"
        })
    
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

def run_monte_carlo(base_assumptions, n_simulations=500):
    results = []
    progress_bar = st.progress(0)
    
    # Define distributions for key variables
    for i in range(n_simulations):
        # Perturb assumptions
        perturbed = base_assumptions.copy()
        perturbed['base_capacity'] = np.random.normal(
            base_assumptions['base_capacity'],
            base_assumptions['base_capacity'] * 0.1  # ±10% variability
        )
        perturbed['ai_efficiency'] = np.random.triangular(
            left=0.1, mode=base_assumptions['ai_efficiency'], right=0.5
        )
        perturbed['phase2_staff'] = int(np.random.choice(
            [base_assumptions['phase2_staff']-1, 
             base_assumptions['phase2_staff'], 
             base_assumptions['phase2_staff']+1]
        ))
        
        # Run simulation
        timeline = generate_timeline(perturbed)
        goal_met = timeline[timeline['Monthly Capacity'] >= 1000]
        
        results.append({
            'peak_capacity': timeline['Monthly Capacity'].max(),
            'months_to_goal': (goal_met.index[0] - timeline.index[0]).days // 30 if len(goal_met) > 0 else None,
            'total_cost': (timeline['Staff Costs'].sum() + timeline['AI Costs'].sum()) / 1e6
        })
        progress_bar.progress((i+1)/n_simulations)
    
    return pd.DataFrame(results)

def render_monte_carlo(base_assumptions):
    st.subheader("Risk Analysis (Monte Carlo Simulation)")
    
    with st.expander("⚙️ Simulation Settings"):
        n_simulations = st.number_input("Number of Simulations", 100, 5000, 500)
        run_sim = st.button("Run Simulation")
    
    if run_sim:
        with st.spinner(f"Running {n_simulations} simulations..."):
            results = run_monte_carlo(base_assumptions, n_simulations)
        
        # Show distributions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1 = px.histogram(results, x='peak_capacity', 
                               title="Peak Capacity Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.histogram(results[~results['months_to_goal'].isna()], 
                               x='months_to_goal',
                               title="Time to Reach Goal (Months)")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            fig3 = px.scatter(results, x='total_cost', y='peak_capacity',
                            title="Cost vs Capacity Tradeoff")
            st.plotly_chart(fig3, use_container_width=True)
        
        # Risk Metrics
        success_rate = (1 - results['months_to_goal'].isna().mean()) * 100
        st.metric("Probability of Achieving Goal", f"{success_rate:.1f}%")

def render_base_visualizations(timeline):
    st.subheader("Timeline Visualization")
    # Example: Plot monthly capacity over time
    fig = px.line(
        timeline.reset_index(),
        x="Month",
        y="Monthly Capacity",
        title="Monthly Capacity Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Optionally, display the data table
    st.dataframe(timeline)

def main():

    st.title("Lab Scalability Modeling Tool")
    
    # Get Inputs and Manage Scenarios
    assumptions = input_assumptions()
    selected_scenarios = scenario_management(assumptions)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["Current Scenario", "Scenario Comparison", "Risk Analysis"])
    
    with tab1:
        # Original single-scenario view
        timeline = generate_timeline(assumptions)
        render_base_visualizations(timeline)
    
    with tab2:
        render_comparison(selected_scenarios)
    
    with tab3:
        render_monte_carlo(assumptions)

if __name__ == "__main__":
    main()
