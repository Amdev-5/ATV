from typing import List, Dict, Any
from datetime import datetime
import json
import os

class ConversationMemory:
    def __init__(self, max_messages: int = 100, persist_path: str = "data/conversation_history"):
        self.max_messages = max_messages
        self.persist_path = persist_path
        self.messages = []
        self.load_history()
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        self.messages.append(message)
        
        # Trim history if it exceeds max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Persist to disk
        self.save_history()
    
    def get_recent_messages(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent messages"""
        return self.messages[-n:]
    
    def get_context_window(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant context based on the query"""
        # Simple keyword-based context retrieval
        keywords = query.lower().split()
        relevant_messages = []
        
        for message in self.messages:
            if any(keyword in message['content'].lower() for keyword in keywords):
                relevant_messages.append(message)
        
        return relevant_messages[-5:]  # Return up to 5 most recent relevant messages
    
    def save_history(self) -> None:
        """Save conversation history to disk"""
        os.makedirs(self.persist_path, exist_ok=True)
        history_file = os.path.join(self.persist_path, 'conversation_history.json')
        
        with open(history_file, 'w') as f:
            json.dump({
                'messages': self.messages,
                'last_updated': datetime.now().isoformat()
            }, f)
    
    def load_history(self) -> None:
        """Load conversation history from disk"""
        history_file = os.path.join(self.persist_path, 'conversation_history.json')
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.messages = data['messages']
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.messages = []
        self.save_history()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.messages:
            return {
                'total_messages': 0,
                'user_messages': 0,
                'assistant_messages': 0,
                'average_response_length': 0
            }
        
        user_messages = sum(1 for m in self.messages if m['role'] == 'user')
        assistant_messages = sum(1 for m in self.messages if m['role'] == 'assistant')
        
        assistant_responses = [m['content'] for m in self.messages if m['role'] == 'assistant']
        avg_response_length = sum(len(r) for r in assistant_responses) / len(assistant_responses) if assistant_responses else 0
        
        return {
            'total_messages': len(self.messages),
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'average_response_length': avg_response_length
        }
    
    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search conversation history for relevant messages"""
        query_terms = query.lower().split()
        matches = []
        
        for message in self.messages:
            content = message['content'].lower()
            if any(term in content for term in query_terms):
                matches.append({
                    'message': message,
                    'relevance_score': sum(1 for term in query_terms if term in content) / len(query_terms)
                })
        
        # Sort by relevance score
        matches.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return [m['message'] for m in matches[:5]]  # Return top 5 matches 