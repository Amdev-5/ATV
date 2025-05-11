import streamlit as st
from chatbot.agent import ATVMaintenanceAgent
from chatbot.knowledge_base import KnowledgeBase
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = ATVMaintenanceAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling"""
    if role == "user":
        st.write(f'👤 **You**: {content}')
    else:
        st.write(f'🤖 **Assistant**: {content}')

def display_analytics_summary():
    """Display analytics summary with visualizations"""
    summary = st.session_state.agent.get_analytics_summary()
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", summary['statistics']['total_records'])
    col2.metric("Unique Models", summary['statistics']['unique_models'])
    col3.metric("Unique Systems", summary['statistics']['unique_systems'])
    col4.metric("Unique Parts", summary['statistics']['unique_parts'])
    
    # Display summary text
    st.write("### System Summary")
    st.write(summary['summary'])

def main():
    st.title("ATV Maintenance Assistant")
    
    # Sidebar for analytics and settings
    with st.sidebar:
        st.header("Analytics Dashboard")
        if st.button("Show Analytics Summary"):
            display_analytics_summary()
        
        st.header("Chat Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Set minimum confidence level for responses"
        )
        
        st.header("Conversation Statistics")
        chat_stats = st.session_state.agent.memory.get_statistics()
        st.write(f"Total Messages: {chat_stats['total_messages']}")
        st.write(f"User Messages: {chat_stats['user_messages']}")
        st.write(f"Assistant Messages: {chat_stats['assistant_messages']}")
    
    # Main chat interface
    st.write("### Chat with ATV Maintenance Assistant")
    st.write("Ask questions about ATV maintenance, defects, and analytics.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message['role'], message['content'])
    
    # Chat input
    user_input = st.text_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get response from agent
        with st.spinner("Thinking..."):
            response = st.session_state.agent.process_query(user_input)
            
            # Check confidence threshold
            if response['confidence'] >= confidence_threshold:
                # Add response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['answer']
                })
                
                # Display suggested actions if any
                if response['suggested_actions']:
                    st.write("### Suggested Actions:")
                    for action in response['suggested_actions']:
                        st.write(f"- {action}")
                
                # Display relevant data
                if response['relevant_data']:
                    with st.expander("View Related Information"):
                        for data in response['relevant_data']:
                            st.write(data)
            else:
                st.warning("I'm not confident enough about this response. Please rephrase your question or provide more context.")
        
        # Rerun to update chat display
        st.experimental_rerun()
    
    # Additional features
    with st.expander("Advanced Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.agent.memory.clear_history()
                st.experimental_rerun()
        
        with col2:
            if st.button("Export Chat History"):
                chat_export = {
                    'history': st.session_state.chat_history,
                    'timestamp': datetime.now().isoformat(),
                    'statistics': chat_stats
                }
                st.download_button(
                    "Download Chat History",
                    data=json.dumps(chat_export, indent=2),
                    file_name="chat_history.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main() 