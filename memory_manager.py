from typing import List, Dict, Any
import json
from datetime import datetime
import os

class ConversationMemory:
    def __init__(self, storage_path: str = "./conversation_history"):
        self.storage_path = storage_path
        self.current_conversation: List[Dict[str, Any]] = []
        self.max_history_length = 10
        
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

    def add_message(self, role: str, content: str, context: Dict[str, Any] = None):
        timestamp = datetime.now().isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "context": context or {}
        }
        
        self.current_conversation.append(message)
        
        # Trim conversation if it gets too long
        if len(self.current_conversation) > self.max_history_length:
            self.current_conversation = self.current_conversation[-self.max_history_length:]
        
        # Save conversation to disk
        self._save_conversation()

    def get_conversation_history(self, n_messages: int = None) -> List[Dict[str, Any]]:
        if n_messages is None:
            return self.current_conversation
        return self.current_conversation[-n_messages:]

    def get_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        # In a more sophisticated implementation, this could use
        # semantic search to find relevant past conversations
        return self.get_conversation_history(n_messages=3)

    def _save_conversation(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        filepath = os.path.join(self.storage_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "messages": self.current_conversation
            }, f, indent=2)

    def load_conversation(self, conversation_id: str):
        filepath = os.path.join(self.storage_path, f"conversation_{conversation_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.current_conversation = data["messages"]
        else:
            raise FileNotFoundError(f"Conversation {conversation_id} not found")

    def clear_conversation(self):
        self.current_conversation = []
        
    def get_conversation_summary(self) -> Dict[str, Any]:
        if not self.current_conversation:
            return {"message_count": 0, "duration": 0}
        
        first_msg = self.current_conversation[0]
        last_msg = self.current_conversation[-1]
        
        first_time = datetime.fromisoformat(first_msg["timestamp"])
        last_time = datetime.fromisoformat(last_msg["timestamp"])
        
        return {
            "message_count": len(self.current_conversation),
            "duration": (last_time - first_time).total_seconds(),
            "start_time": first_msg["timestamp"],
            "end_time": last_msg["timestamp"]
        } 