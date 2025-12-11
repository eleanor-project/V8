import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class PrecedentStorage:
    """Simplified JSON-based precedent storage."""
    
    def __init__(self, storage_dir: str = "data/precedents"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "index.json"
        self._load_index()
    
    def _load_index(self):
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"precedents": [], "next_id": 1}
    
    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def insert_precedent(
        self,
        input_text: str,
        decision: Dict,
        critics: Dict,
        tags: Optional[List[str]] = None
    ) -> int:
        precedent_id = self.index["next_id"]
        timestamp = datetime.utcnow().isoformat()
        
        precedent = {
            "id": precedent_id,
            "timestamp": timestamp,
            "input_text": input_text,
            "decision": decision,
            "critics": critics,
            "tags": tags or [],
            "citation": f"ELEANOR-{precedent_id:04d}"
        }
        
        filename = self.storage_dir / f"precedent_{precedent_id:04d}.json"
        with open(filename, 'w') as f:
            json.dump(precedent, f, indent=2)
        
        self.index["precedents"].append({
            "id": precedent_id,
            "timestamp": timestamp,
            "citation": precedent["citation"],
            "input_preview": input_text[:100]
        })
        self.index["next_id"] += 1
        self._save_index()
        
        return precedent_id
    
    def get_precedent(self, precedent_id: int) -> Optional[Dict]:
        filename = self.storage_dir / f"precedent_{precedent_id:04d}.json"
        if filename.exists():
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    
    def get_all_precedents(self) -> List[Dict]:
        precedents = []
        for entry in self.index["precedents"]:
            prec = self.get_precedent(entry["id"])
            if prec:
                precedents.append(prec)
        return precedents
    
    def count(self) -> int:
        return len(self.index["precedents"])
