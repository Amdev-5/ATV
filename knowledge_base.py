import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.config import Settings
import pandas as pd
from typing import List, Dict, Any
import json
from datetime import datetime

class ATVKnowledgeBase:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.maintenance_collection = self.client.get_or_create_collection(
            name="maintenance_records",
            metadata={"description": "ATV maintenance and defect records"}
        )
        self.parts_collection = self.client.get_or_create_collection(
            name="spare_parts",
            metadata={"description": "Spare parts inventory and consumption"}
        )

    def add_maintenance_record(self, 
                             atv_model: str,
                             defect_description: str,
                             maintenance_action: str,
                             parts_used: List[str],
                             timestamp: str = None) -> str:
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        record_id = f"maint_{timestamp}_{atv_model}"
        
        self.maintenance_collection.add(
            documents=[json.dumps({
                "atv_model": atv_model,
                "defect_description": defect_description,
                "maintenance_action": maintenance_action,
                "parts_used": parts_used,
                "timestamp": timestamp
            })],
            ids=[record_id],
            metadatas=[{
                "atv_model": atv_model,
                "timestamp": timestamp
            }]
        )
        return record_id

    def query_maintenance_history(self, 
                                atv_model: str = None,
                                n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            where_clause = {"atv_model": atv_model} if atv_model else None
            
            # Get all records from the collection
            results = self.maintenance_collection.get()
            
            if not results or not results['documents']:
                return []
            
            # Parse and filter results
            parsed_results = []
            for i, doc in enumerate(results['documents']):
                record = json.loads(doc)
                if atv_model is None or record.get('atv_model') == atv_model:
                    parsed_results.append(record)
            
            # Sort by timestamp and limit results
            parsed_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return parsed_results[:n_results]
            
        except Exception as e:
            print(f"Error querying maintenance history: {e}")
            return []

    def add_parts_record(self,
                        part_number: str,
                        part_name: str,
                        compatible_models: List[str],
                        quantity_used: int,
                        timestamp: str = None) -> str:
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        record_id = f"part_{timestamp}_{part_number}"
        
        self.parts_collection.add(
            documents=[json.dumps({
                "part_number": part_number,
                "part_name": part_name,
                "compatible_models": compatible_models,
                "quantity_used": quantity_used,
                "timestamp": timestamp
            })],
            ids=[record_id],
            metadatas=[{
                "part_number": part_number,
                "timestamp": timestamp
            }]
        )
        return record_id

    def get_parts_consumption(self, 
                            part_number: str = None,
                            n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            where_clause = {"part_number": part_number} if part_number else None
            
            # Get all records from the collection
            results = self.parts_collection.get()
            
            if not results or not results['documents']:
                return []
            
            # Parse and filter results
            parsed_results = []
            for i, doc in enumerate(results['documents']):
                record = json.loads(doc)
                if part_number is None or record.get('part_number') == part_number:
                    parsed_results.append(record)
            
            # Sort by timestamp and limit results
            parsed_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return parsed_results[:n_results]
            
        except Exception as e:
            print(f"Error querying parts consumption: {e}")
            return []

    def analyze_defect_patterns(self, atv_model: str = None) -> Dict[str, Any]:
        records = self.query_maintenance_history(atv_model=atv_model, n_results=100)
        
        defect_counts = {}
        parts_frequency = {}
        
        for record in records:
            defect = record.get("defect_description", "Unknown")
            defect_counts[defect] = defect_counts.get(defect, 0) + 1
            
            for part in record.get("parts_used", []):
                parts_frequency[part] = parts_frequency.get(part, 0) + 1
        
        return {
            "defect_patterns": defect_counts,
            "common_parts_used": parts_frequency
        }

    def get_maintenance_recommendations(self, 
                                     atv_model: str,
                                     defect_description: str) -> List[Dict[str, Any]]:
        try:
            # Get all records and filter for the specific model
            results = self.maintenance_collection.get()
            
            if not results or not results['documents']:
                return []
            
            recommendations = []
            for i, doc in enumerate(results['documents']):
                case = json.loads(doc)
                if case.get('atv_model') == atv_model:
                    recommendations.append({
                        "recommended_action": case.get('maintenance_action', ''),
                        "suggested_parts": case.get('parts_used', []),
                        "based_on_case": case.get('timestamp', '')
                    })
            
            # Sort by timestamp and return most recent recommendations
            recommendations.sort(key=lambda x: x['based_on_case'], reverse=True)
            return recommendations[:3]
            
        except Exception as e:
            print(f"Error getting maintenance recommendations: {e}")
            return [] 