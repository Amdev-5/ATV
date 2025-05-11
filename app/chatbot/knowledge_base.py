import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
import json

class KnowledgeBase:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.create_collection(
            name="atv_knowledge",
            metadata={"description": "ATV maintenance and defect analysis knowledge base"}
        )
        
    def ingest_maintenance_data(self, data: pd.DataFrame) -> None:
        """Ingest maintenance records into the knowledge base"""
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in data.iterrows():
            doc = f"""
            Vehicle: {row['vehicle_id']}
            Model: {row['model']}
            Total Faults: {row['total_faults']}
            Days Since Maintenance: {row['days_since_last_maintenance']}
            Critical System Faults: {row['critical_system_faults']}
            """
            documents.append(doc)
            metadatas.append({
                "vehicle_id": row['vehicle_id'],
                "model": row['model'],
                "total_faults": str(row['total_faults'])
            })
            ids.append(f"maintenance_{idx}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def ingest_defect_data(self, data: pd.DataFrame) -> None:
        """Ingest defect analysis data into the knowledge base"""
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in data.iterrows():
            doc = f"""
            Model: {row['Model']}
            System: {row['System']}
            Fault Count: {row['Fault_Count']}
            Details: {row['Detail_Defect']}
            """
            documents.append(doc)
            metadatas.append({
                "model": row['Model'],
                "system": row['System'],
                "fault_count": str(row['Fault_Count'])
            })
            ids.append(f"defect_{idx}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def ingest_spares_data(self, data: pd.DataFrame) -> None:
        """Ingest spare parts data into the knowledge base"""
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in data.iterrows():
            doc = f"""
            Part: {row['Part_Name']}
            Quantity Consumed: {row['Quantity_Consumed']}
            Proposed Stock: {row['Proposed_Stock']}
            """
            documents.append(doc)
            metadatas.append({
                "part_name": row['Part_Name'],
                "quantity": str(row['Quantity_Consumed'])
            })
            ids.append(f"spare_{idx}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query_knowledge_base(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the knowledge base for relevant information"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        all_metadatas = self.collection.get()['metadatas']
        
        stats = {
            'total_records': len(all_metadatas),
            'models': set(),
            'systems': set(),
            'parts': set()
        }
        
        for metadata in all_metadatas:
            if 'model' in metadata:
                stats['models'].add(metadata['model'])
            if 'system' in metadata:
                stats['systems'].add(metadata['system'])
            if 'part_name' in metadata:
                stats['parts'].add(metadata['part_name'])
        
        return {
            'total_records': stats['total_records'],
            'unique_models': len(stats['models']),
            'unique_systems': len(stats['systems']),
            'unique_parts': len(stats['parts'])
        } 