"""
RAG Engine for SarkariGPT
Implements Semantic Search via ChromaDB and Sentence-Transformers,
combined with structured eligibility evaluation.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
CHROMA_DIR = os.path.join(CURRENT_DIR, "chroma_db")
SCHEMES_FILE = os.path.join(DATA_DIR, "schemes_data.json")

class SarkariRAGEngine:
    def __init__(self):
        self.chroma_dir = CHROMA_DIR
        os.makedirs(self.chroma_dir, exist_ok=True)
        self.schemes_data = self.load_schemes_raw()
        self.collection_name = "sarkari_schemes"
        
        # Initialize embedding function
        self.embedding_fn = self._get_embedding_function()
        
        # Initialize Chroma persistent client
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Indian Government Schemes Knowledge Base"}
        )
        
        # Auto-index if collection is empty
        if self.collection.count() == 0:
            print("[RAG] Ingesting schemes into ChromaDB...")
            self.index_schemes()

    def _get_embedding_function(self):
        """Get SentenceTransformer embedding function with graceful fallback."""
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"[RAG] SentenceTransformer load warning: {e}. Using Chroma default.")
            return embedding_functions.DefaultEmbeddingFunction()

    def load_schemes_raw(self) -> List[Dict[str, Any]]:
        """Load curated schemes list from JSON."""
        if os.path.exists(SCHEMES_FILE):
            try:
                with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading schemes file: {e}")
        return []

    def get_all_schemes(self) -> List[Dict[str, Any]]:
        """Return all schemes from dataset."""
        return self.schemes_data

    def get_scheme_by_id(self, scheme_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details of a specific scheme by ID."""
        for scheme in self.schemes_data:
            if scheme["id"] == scheme_id:
                return scheme
        return None

    def index_schemes(self, force_reload: bool = False):
        """Index all schemes into ChromaDB with structured chunking."""
        if force_reload:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
            self.schemes_data = self.load_schemes_raw()

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for scheme in self.schemes_data:
            s_id = scheme["id"]
            s_name = scheme["name"]
            s_cat = scheme["category"]
            s_url = scheme.get("official_url", "")
            s_elig = scheme.get("eligibility", {})
            
            # Chunk 1: Overview and Summary
            overview_doc = (
                f"Scheme Name: {s_name}\n"
                f"Category: {s_cat}\n"
                f"Ministry: {scheme.get('ministry', '')}\n"
                f"Target Beneficiaries: {', '.join(scheme.get('target_beneficiaries', []))}\n"
                f"Summary: {scheme.get('summary', '')}"
            )
            documents.append(overview_doc)
            metadatas.append({
                "scheme_id": s_id,
                "scheme_name": s_name,
                "category": s_cat,
                "chunk_type": "overview",
                "url": s_url
            })
            ids.append(f"{s_id}_overview")

            # Chunk 2: Eligibility Criteria
            elig_text = s_elig.get("criteria", "")
            occupations = ", ".join(s_elig.get("occupations", []))
            categories = ", ".join(s_elig.get("categories", []))
            genders = ", ".join(s_elig.get("genders", []))
            max_income = s_elig.get("max_annual_income")
            income_str = f"Up to ₹{max_income:,}/yr" if max_income else "No strict income cap"

            elig_doc = (
                f"Scheme Name: {s_name}\n"
                f"Category: {s_cat}\n"
                f"Eligibility Criteria: {elig_text}\n"
                f"Eligible Occupations: {occupations}\n"
                f"Eligible Social Categories: {categories}\n"
                f"Target Genders: {genders}\n"
                f"Income Limit: {income_str}\n"
                f"Age Range: {s_elig.get('age_min', 0)} to {s_elig.get('age_max') or 'No upper limit'}"
            )
            documents.append(elig_doc)
            metadatas.append({
                "scheme_id": s_id,
                "scheme_name": s_name,
                "category": s_cat,
                "chunk_type": "eligibility",
                "url": s_url
            })
            ids.append(f"{s_id}_eligibility")

            # Chunk 3: Benefits
            benefits_doc = (
                f"Scheme Name: {s_name}\n"
                f"Category: {s_cat}\n"
                f"Benefits and Financial Assistance: {scheme.get('benefits', '')}"
            )
            documents.append(benefits_doc)
            metadatas.append({
                "scheme_id": s_id,
                "scheme_name": s_name,
                "category": s_cat,
                "chunk_type": "benefits",
                "url": s_url
            })
            ids.append(f"{s_id}_benefits")

            # Chunk 4: Application Process & Required Documents
            steps = "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(scheme.get("application_steps", []))])
            docs = ", ".join(scheme.get("documents_required", []))
            process_doc = (
                f"Scheme Name: {s_name}\n"
                f"How to Apply:\n{steps}\n"
                f"Required Documents: {docs}\n"
                f"Official Portal: {s_url}"
            )
            documents.append(process_doc)
            metadatas.append({
                "scheme_id": s_id,
                "scheme_name": s_name,
                "category": s_cat,
                "chunk_type": "process",
                "url": s_url
            })
            ids.append(f"{s_id}_process")

        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[RAG] Successfully indexed {len(documents)} chunks for {len(self.schemes_data)} schemes.")

    def semantic_search(self, query: str, n_results: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic similarity search on scheme chunks."""
        where_filter = None
        if category and category != "All":
            where_filter = {"category": category}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )

            hits = []
            if results and results.get("documents") and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    doc = results["documents"][0][i]
                    meta = results["metadatas"][0][i]
                    dist = results["distances"][0][i] if results.get("distances") else 0.0
                    hits.append({
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "relevance_score": round(max(0.0, 1.0 - (dist / 2.0)) * 100, 1)
                    })
            return hits
        except Exception as e:
            print(f"[RAG] Search error: {e}")
            return []

    def evaluate_eligibility(self, scheme: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[int, List[str], List[str]]:
        """
        Evaluate user eligibility for a specific scheme.
        Returns:
            - score (0-100)
            - match_reasons (List[str])
            - warning_reasons (List[str])
        """
        if not profile:
            return 50, ["General eligibility assessment (Profile incomplete)"], []

        score = 100
        matches = []
        warnings = []
        elig = scheme.get("eligibility", {})

        user_age = profile.get("age")
        user_gender = profile.get("gender")
        user_income = profile.get("annual_income")
        user_occ = profile.get("occupation")
        user_cat = profile.get("category")
        user_state = profile.get("state")

        # Gender check
        target_genders = elig.get("genders", [])
        if target_genders and user_gender:
            if user_gender in target_genders:
                matches.append(f"Gender match: Eligible for {user_gender}")
            else:
                score -= 60
                warnings.append(f"Targeted specifically for {', '.join(target_genders)}")

        # Occupation check
        target_occs = elig.get("occupations", [])
        if target_occs and user_occ:
            if user_occ in target_occs:
                matches.append(f"Occupation match: Eligible for {user_occ}")
            else:
                score -= 30
                warnings.append(f"Primarily tailored for: {', '.join(target_occs[:3])}")

        # Income check
        max_income = elig.get("max_annual_income")
        if max_income is not None and user_income is not None:
            if user_income <= max_income:
                matches.append(f"Income within limit (₹{user_income:,} <= ₹{max_income:,})")
            else:
                score -= 50
                warnings.append(f"Exceeds scheme income ceiling of ₹{max_income:,}")

        # Category check
        target_cats = elig.get("categories", [])
        if target_cats and user_cat:
            if user_cat in target_cats:
                matches.append(f"Social category eligible: {user_cat}")
            else:
                score -= 20
                warnings.append(f"Specific to: {', '.join(target_cats)}")

        # Age check
        age_min = elig.get("age_min")
        age_max = elig.get("age_max")
        if user_age is not None:
            if age_min is not None and user_age < age_min:
                score -= 40
                warnings.append(f"Minimum age required is {age_min} years")
            elif age_max is not None and user_age > age_max:
                score -= 40
                warnings.append(f"Maximum age limit is {age_max} years")
            else:
                matches.append("Age criteria satisfied")

        final_score = max(10, min(100, score))
        return final_score, matches, warnings

    def get_ranked_schemes_for_profile(self, profile: Dict[str, Any], query: str = "") -> List[Dict[str, Any]]:
        """
        Rank schemes according to user profile eligibility and semantic query match.
        """
        scored_schemes = []

        # If query provided, get semantic scores
        semantic_map = {}
        if query.strip():
            hits = self.semantic_search(query, n_results=12)
            for hit in hits:
                s_id = hit["metadata"]["scheme_id"]
                semantic_map[s_id] = max(semantic_map.get(s_id, 0.0), hit["relevance_score"])

        for scheme in self.schemes_data:
            s_id = scheme["id"]
            elig_score, matches, warnings = self.evaluate_eligibility(scheme, profile)
            
            # Hybrid combined score
            sem_score = semantic_map.get(s_id, 50.0 if not query.strip() else 10.0)
            if query.strip():
                combined = (elig_score * 0.4) + (sem_score * 0.6)
            else:
                combined = elig_score

            scored_schemes.append({
                "scheme": scheme,
                "score": round(combined, 1),
                "eligibility_score": elig_score,
                "semantic_score": sem_score if query.strip() else None,
                "matches": matches,
                "warnings": warnings
            })

        scored_schemes.sort(key=lambda x: x["score"], reverse=True)
        return scored_schemes
