import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = {}

def calculate_capacity(row, assumptions):
    base = assumptions['base_capacity']
    if row['Phase'] == 'Phase 1':
        return base * assumptions['phase1_staff']
    elif row['Phase'] == 'Phase 2':
        return base * assumptions['phase2_staff'] * assumptions['shift_multiplier']
    else:  # Phase 3
        return (base * assumptions['phase3_staff'] *
                assumptions['shift_multiplier'] *
                (1 + assumptions['ai_efficiency']))

def generate_timeline(assumptions):
    # Create monthly date range
    dates = pd.date_range(
        start=assumptions['phase1_start'],
        end=assumptions['phase3_end'],
        freq='MS'
    )
    
    timeline = pd.DataFrame(index=dates)
    timeline.index.name = 'Month'
    
    # Assign phases
    timeline['Phase'] = 'Phase 1'
    timeline.loc[timeline.index >= assumptions['phase2_start'], 'Phase'] = 'Phase 2'
    timeline.loc[timeline.index >= assumptions['phase3_start'], 'Phase'] = 'Phase 3'
    
    # Calculate Monthly Capacity & Cumulative Samples
    timeline['Monthly Capacity'] = timeline.apply(
        lambda x: calculate_capacity(x, assumptions), axis=1
    )
    timeline['Cumulative Samples'] = timeline['Monthly Capacity'].cumsum()
    
    # Staff Costs
    timeline['Staff Costs'] = timeline['Phase'].map({
        'Phase 1': assumptions['phase1_staff'] * assumptions['salary'],
        'Phase 2': assumptions['phase2_staff'] * assumptions['salary'],
        'Phase 3': assumptions['phase3_staff'] * assumptions['salary']
    })
    
    # Overhead Costs with defaults if necessary
    equipment_lease = assumptions.get('equipment_lease', 10378)
    instrument_running = assumptions.get('instrument_running', 2775)
    software_licenses = assumptions.get('software_licenses', 2000)
    qaqc_rate = assumptions.get('qaqc_rate', 0.08)
    
    overhead_base = equipment_lease + instrument_running + software_licenses
    timeline['Overhead Costs'] = overhead_base + overhead_base * qaqc_rate
    
    # Total Cost
    timeline['Total Cost'] = timeline['Staff Costs'] + timeline['Overhead Costs']
    
    # Revenue & Profit
    avg_test_price = assumptions.get('avg_test_price', 300)
    timeline['Revenue'] = timeline['Monthly Capacity'] * avg_test_price
    timeline['Profit'] = timeline['Revenue'] - timeline['Total Cost']
    
    # Goal Tracking using the user-specified goal:
    goal = assumptions.get('samples_goal', 1000)
    timeline['Goal Met?'] = timeline['Monthly Capacity'].apply(
        lambda x: "Goal Met" if x >= goal else "Under Target"
    )
    
    return timeline


def input_assumptions():
    with st.sidebar:
        st.header("Model Assumptions")
        
        assumptions = {
            'base_capacity': st.number_input("Base Capacity/Analyst/Month", 100, value=440),
            'phase1_staff': st.number_input("Phase 1 Staff", 1, value=3),
            'phase2_staff': st.number_input("Phase 2 Staff", 1, value=5),
            'phase3_staff': st.number_input("Phase 3 Staff", 1, value=5),
            'shift_multiplier': st.slider("Shift Multiplier (Phase 2+)", 1.0, 3.0, 2.0),
            'ai_efficiency': st.slider("AI Efficiency Boost (Phase 3)", 0.0, 1.0, 0.3),
            'salary': st.number_input("Monthly Salary/Analyst ($)", 1000, value=5000),
            'equipment_lease': st.number_input("Equipment Lease ($/month)", 0, value=10378),
            'instrument_running': st.number_input("Instrument Running ($/month)", 0, value=2775),
            'software_licenses': st.number_input("Software/Licenses ($/month)", 0, value=2000),
            'qaqc_rate': st.slider("QA/QC Rate (% of overhead)", 0.0, 0.2, 0.08),
            'avg_test_price': st.number_input("Average Test Price ($/sample)", 1, value=300),
            # New input for sample/month goal:
            'samples_goal': st.number_input("Monthly Sample Goal", 100, 10000, value=1000)
        }
        
        st.subheader("Phase Dates")
        assumptions['phase1_start'] = pd.to_datetime(st.date_input("Phase 1 Start", datetime(2025,4,1)))
        assumptions['phase2_start'] = pd.to_datetime(st.date_input("Phase 2 Start", datetime(2026,1,1)))
        assumptions['phase3_start'] = pd.to_datetime(st.date_input("Phase 3 Start", datetime(2027,1,1)))
        assumptions['phase3_end']   = pd.to_datetime(st.date_input("Model End Date", datetime(2027,12,31)))
        
        return assumptions


def scenario_management(assumptions):
    """Save/load scenarios from session state."""
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
            default=[list(st.session_state.scenarios.keys())[-1]] if st.session_state.scenarios else []
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

# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------
def render_base_visualizations(timeline):
    """Display line charts for capacity, revenue, and profit in the current scenario."""
    st.subheader("Monthly Capacity")
    fig_cap = px.line(
        timeline.reset_index(), x='Month', y='Monthly Capacity',
        title="Monthly Capacity Over Time"
    )
    fig_cap.add_hline(y=1000, line_dash="dot", line_color="red")
    st.plotly_chart(fig_cap, use_container_width=True)
    
    st.subheader("Monthly Revenue")
    fig_rev = px.line(
        timeline.reset_index(), x='Month', y='Revenue',
        title="Monthly Revenue Over Time"
    )
    st.plotly_chart(fig_rev, use_container_width=True)
    
    st.subheader("Monthly Profit")
    fig_profit = px.line(
        timeline.reset_index(), x='Month', y='Profit',
        title="Monthly Profit Over Time"
    )
    st.plotly_chart(fig_profit, use_container_width=True)
    
    # Show a data table with key columns
    st.dataframe(timeline[[
        "Phase", "Monthly Capacity", "Goal Met?", "Revenue", "Total Cost", "Profit"
    ]])

def render_comparison(selected_scenarios):
    """Compare capacity, revenue, and profit across multiple scenarios."""
    if len(selected_scenarios) < 1:
        return
    
    all_data = []
    for name in selected_scenarios:
        timeline = generate_timeline(st.session_state.scenarios[name])
        timeline['Scenario'] = name
        all_data.append(timeline)
    
    combined = pd.concat(all_data)
    
    # Capacity Comparison
    st.subheader("Scenario Comparison - Capacity")
    fig_cap = px.line(
        combined.reset_index(), x='Month', y='Monthly Capacity',
        color='Scenario', title="Monthly Capacity Across Scenarios"
    )
    fig_cap.add_hline(y=1000, line_dash="dot", line_color="red")
    st.plotly_chart(fig_cap, use_container_width=True)
    
    # Revenue Comparison
    st.subheader("Scenario Comparison - Revenue")
    fig_rev = px.line(
        combined.reset_index(), x='Month', y='Revenue',
        color='Scenario', title="Monthly Revenue Across Scenarios"
    )
    st.plotly_chart(fig_rev, use_container_width=True)
    
    # Profit Comparison
    st.subheader("Scenario Comparison - Profit")
    fig_profit = px.line(
        combined.reset_index(), x='Month', y='Profit',
        color='Scenario', title="Monthly Profit Across Scenarios"
    )
    st.plotly_chart(fig_profit, use_container_width=True)
    
    # Metrics Table
    st.subheader("Key Metrics")
    metrics = []
    for name in selected_scenarios:
        tl = generate_timeline(st.session_state.scenarios[name])
        # Identify the first month where capacity >= 1000
        goal_achieved = tl[tl['Monthly Capacity'] >= 1000]
        metrics.append({
            'Scenario': name,
            'Peak Capacity': tl['Monthly Capacity'].max(),
            'Total Revenue ($M)': tl['Revenue'].sum() / 1e6,
            'Total Cost ($M)': tl['Total Cost'].sum() / 1e6,
            'Total Profit ($M)': tl['Profit'].sum() / 1e6,
            'Goal Achieved Date': goal_achieved.index[0] if not goal_achieved.empty else "Not Reached"
        })
    
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

# -----------------------------------------------------------------------------
# MONTE CARLO SIMULATION
# -----------------------------------------------------------------------------
def run_monte_carlo(base_assumptions, n_simulations=500):
    results = []
    progress_bar = st.progress(0)
    
    for i in range(n_simulations):
        perturbed = base_assumptions.copy()
        
        # Perturb certain assumptions
        perturbed['base_capacity'] = np.random.normal(
            base_assumptions['base_capacity'],
            base_assumptions['base_capacity'] * 0.1
        )
        perturbed['ai_efficiency'] = np.random.triangular(
            left=0.1, mode=base_assumptions['ai_efficiency'], right=0.5
        )
        perturbed['phase2_staff'] = int(np.random.choice([
            base_assumptions['phase2_staff'] - 1,
            base_assumptions['phase2_staff'],
            base_assumptions['phase2_staff'] + 1
        ]))
        
        # Generate timeline and check if goal is met
        timeline = generate_timeline(perturbed)
        goal_met = timeline[timeline['Monthly Capacity'] >= base_assumptions['samples_goal']]
        
        results.append({
            'peak_capacity': timeline['Monthly Capacity'].max(),
            'months_to_goal': ((goal_met.index[0] - timeline.index[0]).days // 30)
                              if not goal_met.empty else None,
            'total_cost': timeline['Total Cost'].sum() / 1e6
        })
        progress_bar.progress((i + 1) / n_simulations)
    
    return pd.DataFrame(results)

def render_monte_carlo(base_assumptions):
    st.subheader("Risk Analysis (Monte Carlo Simulation)")
    
    with st.expander("⚙️ Simulation Settings"):
        n_simulations = st.number_input("Number of Simulations", 100, 5000, 500)
        run_sim = st.button("Run Simulation")
    
    if run_sim:
        with st.spinner(f"Running {n_simulations} simulations..."):
            results = run_monte_carlo(base_assumptions, n_simulations)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1 = px.histogram(
                results, x='peak_capacity',
                title="Peak Capacity Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Only include simulations where the goal was reached
            fig2 = px.histogram(
                results[~results['months_to_goal'].isna()],
                x='months_to_goal',
                title="Time to Reach Goal (Months)"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            fig3 = px.scatter(
                results, x='total_cost', y='peak_capacity',
                title="Cost vs Capacity Tradeoff"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Calculate summary statistics for interpretation
        avg_peak = results['peak_capacity'].mean()
        median_peak = results['peak_capacity'].median()
        std_peak = results['peak_capacity'].std()
        
        # Calculate time-to-goal statistics only for simulations where goal was reached
        valid_times = results[results['months_to_goal'].notna()]
        if not valid_times.empty:
            avg_time = valid_times['months_to_goal'].mean()
            median_time = valid_times['months_to_goal'].median()
        else:
            avg_time = None
            median_time = None
        
        # Calculate probability of reaching the sample goal (user-defined samples_goal)
        probability_goal = (1 - results['months_to_goal'].isna().mean()) * 100
        
        # Display key metric for quick view
        st.metric("Probability of Achieving Goal", f"{probability_goal:.1f}%")
        
        # Detailed contextual interpretation summary
        st.markdown("### Monte Carlo Simulation Interpretation")
        st.markdown(
            f"**Peak Capacity Distribution:** The simulations indicate an average peak capacity of **{avg_peak:.1f} tests/month** "
            f"(median: **{median_peak:.1f} tests/month**, standard deviation: **{std_peak:.1f}**). This shows the typical "
            "throughput you might expect, as well as the degree of variability driven by uncertainties in staffing and operational parameters."
        )
        if avg_time is not None:
            st.markdown(
                f"**Time to Reach Goal:** Among the simulations where the lab meets the sample goal of **{base_assumptions['samples_goal']} samples/month**, "
                f"the average time to achieve this milestone was **{avg_time:.1f} months** (median: **{median_time:.1f} months**). "
                "This provides an estimate of how long it might take under favorable conditions, though some simulations show longer delays."
            )
        else:
            st.markdown("**Time to Reach Goal:** In these simulations, the lab did not consistently reach the specified sample goal.")
        st.markdown(
            "**Cost vs Capacity Tradeoff:** The scatter plot illustrates that higher capacities tend to correlate with increased overall costs. "
            "This tradeoff is critical when planning for scalability, as investing in higher capacity may lead to significantly higher expenditures."
        )
        st.markdown(
            f"**Probability of Achieving Goal:** There is a **{probability_goal:.1f}%** chance that the lab will meet or exceed the target of "
            f"**{base_assumptions['samples_goal']} samples/month** within the modeled timeframe."
        )


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.title("Lab Scalability Modeling Tool")
    
    # Get user inputs and manage scenario state
    assumptions = input_assumptions()
    selected_scenarios = scenario_management(assumptions)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["Current Scenario", "Scenario Comparison", "Risk Analysis"])
    
    with tab1:
        # Single-scenario view
        timeline = generate_timeline(assumptions)
        render_base_visualizations(timeline)
    
    with tab2:
        render_comparison(selected_scenarios)
    
    with tab3:
        render_monte_carlo(assumptions)

if __name__ == "__main__":
    main()
