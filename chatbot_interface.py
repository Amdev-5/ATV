import streamlit as st

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="ATV Maintenance Assistant",
    page_icon="🚙",
    layout="wide"
)

import asyncio
from agent import MaintenanceAgent
from knowledge_base import ATVKnowledgeBase
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# Initialize components
@st.cache_resource
def init_components():
    agent = MaintenanceAgent()
    kb = ATVKnowledgeBase()
    
    # Add sample data if none exists
    if not kb.query_maintenance_history():
        sample_data = [
            {
                "atv_model": "Ranger-1000",
                "defect_description": "Engine overheating",
                "maintenance_action": "Replaced coolant and thermostat",
                "parts_used": ["coolant", "thermostat", "hose_clamp"],
                "timestamp": (datetime.now() - timedelta(days=5)).isoformat()
            },
            {
                "atv_model": "MRZR",
                "defect_description": "Brake system failure",
                "maintenance_action": "Replaced brake pads and rotors",
                "parts_used": ["brake_pads", "rotors", "brake_fluid"],
                "timestamp": (datetime.now() - timedelta(days=3)).isoformat()
            },
            {
                "atv_model": "RZR",
                "defect_description": "Transmission issues",
                "maintenance_action": "Replaced transmission fluid and filter",
                "parts_used": ["transmission_fluid", "filter", "gasket"],
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat()
            }
        ]
        
        for record in sample_data:
            kb.add_maintenance_record(
                atv_model=record["atv_model"],
                defect_description=record["defect_description"],
                maintenance_action=record["maintenance_action"],
                parts_used=record["parts_used"],
                timestamp=record["timestamp"]
            )
    
    return agent, kb

agent, kb = init_components()

# Sidebar
st.sidebar.title("ATV Maintenance Assistant")
atv_model = st.sidebar.selectbox(
    "Select ATV Model",
    ["Ranger-1000", "MRZR", "RZR", None],
    format_func=lambda x: "All Models" if x is None else x
)

# Main interface
st.title("ATV Maintenance Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["role"] == "assistant":
            with st.expander("View Context"):
                st.json(message["context"])

# Chat input
if prompt := st.chat_input("Ask about ATV maintenance..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run async response in sync context
            response = asyncio.run(agent.get_response(prompt, atv_model))
            st.markdown(response["response"])
            with st.expander("View Context"):
                st.json(response["context"])
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["response"],
                "context": response["context"]
            })

# Analytics section
st.divider()
st.header("Maintenance Analytics")

# Create tabs for different analytics views
tab1, tab2, tab3 = st.tabs(["Defect Patterns", "Parts Usage", "Maintenance History"])

with tab1:
    st.subheader("Common Defect Patterns")
    patterns = kb.analyze_defect_patterns(atv_model)
    
    if patterns["defect_patterns"]:
        df = pd.DataFrame(
            list(patterns["defect_patterns"].items()),
            columns=["Defect", "Count"]
        )
        fig = px.bar(df, x="Defect", y="Count", title="Defect Frequency")
        st.plotly_chart(fig)
    else:
        st.info("No defect patterns available yet.")

with tab2:
    st.subheader("Parts Usage Analysis")
    if patterns["common_parts_used"]:
        df = pd.DataFrame(
            list(patterns["common_parts_used"].items()),
            columns=["Part", "Usage Count"]
        )
        fig = px.pie(df, values="Usage Count", names="Part", title="Parts Usage Distribution")
        st.plotly_chart(fig)
    else:
        st.info("No parts usage data available yet.")

with tab3:
    st.subheader("Recent Maintenance History")
    history = kb.query_maintenance_history(atv_model)
    if history:
        df = pd.DataFrame(history)
        st.dataframe(
            df[["timestamp", "atv_model", "defect_description", "maintenance_action"]],
            hide_index=True
        )
    else:
        st.info("No maintenance history available yet.")

# Footer
st.divider()
st.markdown("*For assistance, contact the maintenance team.*") 