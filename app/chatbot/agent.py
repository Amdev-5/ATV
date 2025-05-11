import google.generativeai as genai
from typing import Dict, List, Any
import os
from dotenv import load_dotenv
from .knowledge_base import KnowledgeBase
from .memory_manager import ConversationMemory

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

class ATVMaintenanceAgent:
    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.knowledge_base = KnowledgeBase()
        self.memory = ConversationMemory()
        
        # Define system prompt
        self.system_prompt = """
        You are an expert ATV maintenance and analytics advisor. Your role is to:
        1. Analyze maintenance data and provide insights
        2. Help with defect analysis and prediction
        3. Provide spare parts recommendations
        4. Explain maintenance schedules and priorities
        
        Use the provided knowledge base to support your responses with data.
        Always explain your reasoning and provide specific examples when possible.
        """
    
    def _generate_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate a detailed prompt combining user query, context, and history"""
        conversation_history = self.memory.get_recent_messages(5)
        
        prompt = f"{self.system_prompt}\n\n"
        prompt += "Previous conversation:\n"
        for msg in conversation_history:
            prompt += f"{msg['role']}: {msg['content']}\n"
        
        prompt += "\nRelevant context:\n"
        for ctx in context:
            prompt += f"- {ctx}\n"
        
        prompt += f"\nUser query: {query}\n"
        prompt += "\nProvide a detailed response with explanations and data-backed insights."
        
        return prompt
    
    def _extract_insights(self, context: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from the context"""
        insights = []
        
        for item in context['documents']:
            if 'fault' in item.lower():
                insights.append("Defect Analysis: " + item)
            if 'maintenance' in item.lower():
                insights.append("Maintenance Info: " + item)
            if 'stock' in item.lower():
                insights.append("Inventory Data: " + item)
        
        return insights
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query and generate response"""
        # Query knowledge base
        kb_results = self.knowledge_base.query_knowledge_base(query)
        insights = self._extract_insights(kb_results)
        
        # Generate prompt
        prompt = self._generate_prompt(query, insights)
        
        # Get response from Gemini
        response = self.model.generate_content(prompt)
        
        # Process and structure the response
        structured_response = {
            'answer': response.text,
            'relevant_data': insights,
            'confidence': self._calculate_confidence(kb_results['distances']),
            'suggested_actions': self._extract_actions(response.text)
        }
        
        # Update conversation memory
        self.memory.add_message('user', query)
        self.memory.add_message('assistant', structured_response['answer'])
        
        return structured_response
    
    def _calculate_confidence(self, distances: List[float]) -> float:
        """Calculate confidence score based on knowledge base query distances"""
        if not distances:
            return 0.0
        
        # Convert distances to confidence scores (inverse relationship)
        confidence_scores = [1 / (1 + d) for d in distances]
        return sum(confidence_scores) / len(confidence_scores)
    
    def _extract_actions(self, response: str) -> List[str]:
        """Extract suggested actions from the response"""
        actions = []
        
        # Simple rule-based action extraction
        sentences = response.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in 
                  ['should', 'recommend', 'suggest', 'need to', 'must']):
                actions.append(sentence.strip())
        
        return actions
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Generate a summary of current analytics"""
        stats = self.knowledge_base.get_system_statistics()
        
        prompt = f"""
        Generate a summary of the ATV maintenance system based on these statistics:
        - Total Records: {stats['total_records']}
        - Unique Models: {stats['unique_models']}
        - Unique Systems: {stats['unique_systems']}
        - Unique Parts: {stats['unique_parts']}
        
        Provide insights and recommendations.
        """
        
        response = self.model.generate_content(prompt)
        
        return {
            'statistics': stats,
            'summary': response.text,
            'last_updated': self.knowledge_base.get_last_update_time()
        }
    
    def explain_decision(self, decision_type: str, data: Dict[str, Any]) -> str:
        """Explain the reasoning behind a specific decision"""
        explanation_prompt = f"""
        Explain the reasoning behind this {decision_type} decision:
        {data}
        
        Consider:
        1. Key factors influencing the decision
        2. Data points used
        3. Expected outcomes
        4. Alternative scenarios
        """
        
        response = self.model.generate_content(explanation_prompt)
        return response.text 