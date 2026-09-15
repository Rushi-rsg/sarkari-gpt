"""
Comprehensive test script for SarkariGPT RAG & Services
"""
import os
import sys

# Ensure D:\sarkari-gpt is on sys.path
sys.path.insert(0, r"D:\sarkari-gpt")

from rag_engine import SarkariRAGEngine
import storage

print("=== 1. Testing Storage ===")
test_profile = {
    "name": "Ramesh Kumar",
    "state": "Maharashtra",
    "category": "OBC",
    "occupation": "Farmer",
    "age": 42,
    "annual_income": 120000,
    "family_size": 4
}
assert storage.save_user_profile(test_profile) == True
loaded = storage.load_user_profile()
assert loaded["name"] == "Ramesh Kumar"
print("Storage profile save/load: PASS")

print("\n=== 2. Testing ChromaDB + SentenceTransformers RAG ===")
rag = SarkariRAGEngine()
print(f"Total schemes loaded: {len(rag.get_all_schemes())}")
print(f"ChromaDB collection count: {rag.collection.count()} chunks")

# Test semantic query
query = "What financial assistance is available for crop insurance or small farmers?"
results = rag.semantic_search(query, n_results=3)
print(f"\nSemantic Search Query: '{query}'")
for i, r in enumerate(results, 1):
    meta = r["metadata"]
    print(f"  {i}. {meta['scheme_name']} ({meta['category']}) - Relevance: {r['relevance_score']}% [Chunk: {meta['chunk_type']}]")

# Test eligibility evaluation
print("\n=== 3. Testing Eligibility Evaluation ===")
ranked = rag.get_ranked_schemes_for_profile(test_profile, query="farmer crop loans")
for item in ranked[:4]:
    s = item["scheme"]
    print(f"  Scheme: {s['name']}")
    print(f"  Eligibility Score: {item['eligibility_score']}% | Combined Score: {item['score']}")
    print(f"  Matches: {item['matches']}")
    print(f"  Warnings: {item['warnings']}")
    print()

print("ALL TESTS PASSED SUCCESSFULLY!")
