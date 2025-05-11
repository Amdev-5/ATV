import os
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from memory_manager import ConversationMemory
from knowledge_base import ATVKnowledgeBase

load_dotenv()

class MaintenanceAgent:
    def __init__(self):
        # Initialize Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize LangChain components
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        
        # Initialize other components
        self.memory = ConversationMemory()
        self.knowledge_base = ATVKnowledgeBase()
        
        # Define prompt templates
        self.qa_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert ATV maintenance assistant. 
            Use the following context to answer the question:
            
            Context: {context}
            
            Question: {question}
            
            Provide a detailed and helpful response."""
        )
        
        # Initialize chains
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=self.qa_template
        )

    def _generate_context(self, query: str) -> str:
        # Get relevant maintenance history
        maintenance_history = self.knowledge_base.query_maintenance_history(n_results=3)
        
        # Get conversation context
        conversation_context = self.memory.get_relevant_context(query)
        
        # Format context as string
        context = "Maintenance History:\n"
        for record in maintenance_history:
            context += f"- Model: {record.get('atv_model', 'Unknown')}\n"
            context += f"  Issue: {record.get('defect_description', 'Unknown')}\n"
            context += f"  Action: {record.get('maintenance_action', 'Unknown')}\n\n"
        
        context += "\nRecent Conversation:\n"
        for msg in conversation_context:
            context += f"{msg.get('role', 'unknown')}: {msg.get('content', '')}\n"
        
        return context

    def get_response(self, 
                    query: str, 
                    atv_model: str = None) -> Dict[str, Any]:
        try:
            # Generate context
            context = self._generate_context(query)
            
            # Add model-specific context if provided
            if atv_model:
                recommendations = self.knowledge_base.get_maintenance_recommendations(
                    atv_model=atv_model,
                    defect_description=query
                )
                context += f"\nSpecific Recommendations for {atv_model}:\n"
                for rec in recommendations:
                    context += f"- Action: {rec.get('recommended_action', '')}\n"
                    context += f"  Parts: {', '.join(rec.get('suggested_parts', []))}\n"
            
            # Create a proper Document object for the QA chain
            doc = Document(
                page_content=context,
                metadata={"source": "maintenance_history"}
            )

            try:
                # Get response using QA chain
                chain_response = self.qa_chain({
                    "input_documents": [doc],
                    "question": query
                })
                
                response_text = chain_response.get('output_text', 
                    "I apologize, but I couldn't generate a response at the moment.")
            except Exception as chain_error:
                print(f"Chain error: {chain_error}")
                # Fallback to direct Gemini API if chain fails
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = f"{self.qa_template.format(context=context, question=query)}"
                response = model.generate_content(prompt)
                response_text = response.text
            
            # Save conversation
            self.memory.add_message(
                role="user",
                content=query,
                context={"atv_model": atv_model}
            )
            self.memory.add_message(
                role="assistant",
                content=response_text,
                context={"context": context}
            )
            
            return {
                "response": response_text,
                "context": context
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I'm having trouble generating a response. Please try again.",
                "context": str(e)
            }

    def analyze_maintenance_patterns(self, atv_model: str = None) -> Dict[str, Any]:
        """Analyze maintenance patterns and provide insights."""
        try:
            patterns = self.knowledge_base.analyze_defect_patterns(atv_model)
            
            # Create prompt for analysis
            analysis_template = PromptTemplate(
                input_variables=["patterns"],
                template="""Analyze the following maintenance patterns:
                
                {patterns}
                
                Provide key insights and recommendations based on this data."""
            )
            
            try:
                # Create and run analysis chain
                analysis_chain = LLMChain(llm=self.llm, prompt=analysis_template)
                response = analysis_chain({
                    "patterns": f"Defect Patterns: {patterns['defect_patterns']}\n"
                               f"Common Parts Used: {patterns['common_parts_used']}"
                })
                insights = response.get('text', "Unable to generate insights.")
            except Exception as chain_error:
                print(f"Chain error in analysis: {chain_error}")
                # Fallback to direct Gemini API
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = analysis_template.format(
                    patterns=f"Defect Patterns: {patterns['defect_patterns']}\n"
                            f"Common Parts Used: {patterns['common_parts_used']}"
                )
                response = model.generate_content(prompt)
                insights = response.text
            
            return {
                "patterns": patterns,
                "insights": insights
            }
            
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return {
                "patterns": {},
                "insights": f"Error analyzing patterns: {str(e)}"
            }

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return self.memory.get_conversation_summary()

    def clear_conversation(self):
        """Clear the current conversation history."""
        self.memory.clear_conversation() 