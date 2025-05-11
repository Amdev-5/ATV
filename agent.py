import google.generativeai as genai
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from memory_manager import ConversationMemory
from knowledge_base import ATVKnowledgeBase

load_dotenv()

class MaintenanceAgent:
    def __init__(self):
        # Initialize Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize components
        self.memory = ConversationMemory()
        self.knowledge_base = ATVKnowledgeBase()
        
        # Define system prompt
        self.system_prompt = """You are an expert ATV maintenance assistant. 
        Your role is to help diagnose issues, recommend maintenance actions, 
        and provide detailed technical guidance for ATVs including Ranger-1000, 
        MRZR, and RZR models. Use the maintenance history and technical documentation 
        to provide accurate and helpful responses."""

    def _generate_context(self, query: str) -> Dict[str, Any]:
        # Get relevant maintenance history
        maintenance_history = self.knowledge_base.query_maintenance_history(n_results=3)
        
        # Get conversation context
        conversation_context = self.memory.get_relevant_context(query)
        
        return {
            "maintenance_history": maintenance_history,
            "conversation_context": conversation_context
        }

    async def get_response(self, 
                          query: str, 
                          atv_model: str = None) -> Dict[str, Any]:
        # Generate context
        context = self._generate_context(query)
        
        # If ATV model is specified, get specific recommendations
        if atv_model:
            recommendations = self.knowledge_base.get_maintenance_recommendations(
                atv_model=atv_model,
                defect_description=query
            )
            context["recommendations"] = recommendations
        
        # Construct prompt with context
        prompt = f"{self.system_prompt}\n\nContext:\n"
        prompt += f"Maintenance History: {str(context['maintenance_history'])}\n"
        if atv_model:
            prompt += f"Recommendations for {atv_model}: {str(context['recommendations'])}\n"
        prompt += f"\nUser Query: {query}"
        
        try:
            # Generate response using Gemini
            response = await self.model.generate_content_async(prompt)
            response_text = response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            response_text = "I apologize, but I'm having trouble generating a response at the moment. Please try again."
        
        # Save conversation
        self.memory.add_message(
            role="user",
            content=query,
            context={"atv_model": atv_model}
        )
        self.memory.add_message(
            role="assistant",
            content=response_text,
            context=context
        )
        
        return {
            "response": response_text,
            "context": context
        }

    def analyze_maintenance_patterns(self, atv_model: str = None) -> Dict[str, Any]:
        """Analyze maintenance patterns and provide insights."""
        patterns = self.knowledge_base.analyze_defect_patterns(atv_model)
        
        # Generate insights using Gemini
        insights_prompt = f"""Analyze the following maintenance patterns for 
        {'all ATVs' if not atv_model else atv_model}:
        
        Defect Patterns: {patterns['defect_patterns']}
        Common Parts Used: {patterns['common_parts_used']}
        
        Provide key insights and recommendations based on this data."""
        
        try:
            response = self.model.generate_content(insights_prompt)
            insights = response.text
        except Exception as e:
            print(f"Error generating insights: {e}")
            insights = "Unable to generate insights at the moment."
        
        return {
            "patterns": patterns,
            "insights": insights
        }

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return self.memory.get_conversation_summary()

    def clear_conversation(self):
        """Clear the current conversation history."""
        self.memory.clear_conversation() 