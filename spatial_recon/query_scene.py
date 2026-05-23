import json
import argparse
from pathlib import Path

def load_memory(run_dir: Path):
    memory_path = run_dir / "exports" / "scene_memory.json"
    if not memory_path.exists():
        raise FileNotFoundError(f"Memory file not found at {memory_path}. Run memory builder first.")
    with open(memory_path, "r") as f:
        return json.load(f)

def query_memory(memory, query_type, target=None):
    results = []
    
    if query_type == "affordance":
        # e.g., "Find placeable surfaces." -> target="placeable"
        for obj_id, data in memory.items():
            if target in data.get("affordances", []):
                results.append(obj_id)
                
    elif query_type == "largest":
        # e.g., "Find the largest support surface." -> target="support_surface"
        candidates = []
        for obj_id, data in memory.items():
            if target in data.get("affordances", []) or target == data["label"]:
                candidates.append((obj_id, data["volume"]))
        if candidates:
            # Sort by volume descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            results.append(candidates[0][0])
            
    elif query_type == "near":
        # e.g., "What objects are near the desk?" -> target="desk"
        target_ids = [obj_id for obj_id, data in memory.items() if data["label"] == target]
        for t_id in target_ids:
            results.extend(memory[t_id].get("relations", {}).get("near", []))
        results = sorted(set(results))

    elif query_type == "on_top_of":
        # e.g., "What objects are on top of the desk?" -> target="desk"
        target_ids = {obj_id for obj_id, data in memory.items() if data["label"] == target}
        for target_id in target_ids:
            results.extend(memory[target_id].get("relations", {}).get("supports", []))
        results = sorted(set(results))
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Query object-level scene memory.")
    parser.add_argument("--run", type=str, required=True, help="Path to the run directory")
    args = parser.parse_args()
    
    memory = load_memory(Path(args.run))
    
    print("\n--- Multimodal Semantic Search ---")
    
    # Pre-defined query mappings for the scene memory demo.
    queries = [
        ("Find placeable surfaces.", "affordance", "placeable"),
        ("Find sittable objects.", "affordance", "sittable"),
        ("What objects are near the desk?", "near", "desk"),
        ("What objects are on top of the desk?", "on_top_of", "desk"),
        ("Find the largest support surface.", "largest", "support_surface"),
        ("Find movable objects.", "affordance", "movable")
    ]
    
    for question, q_type, target in queries:
        print(f"\nQ: {question}")
        answers = query_memory(memory, q_type, target)
        if answers:
            print(f"A: {', '.join(answers)}")
        else:
            print("A: No matching objects found in memory.")
            
if __name__ == "__main__":
    main()
