#!/usr/bin/env python3
import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Set


def collect_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from collect_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from collect_strings(v)


def extract_api_ids_from_ops_flat(data: Any) -> Set[str]:
    known: Set[str] = set()

    # Case 1: dict mapping api_id -> info
    if isinstance(data, dict):
        known.update(k for k in data.keys() if isinstance(k, str))
        # Also scan nested values for explicit ids
        for v in data.values():
            if isinstance(v, dict):
                for key in ("api_id", "id", "full_id", "name", "full_name"):
                    val = v.get(key)
                    if isinstance(val, str) and val.startswith("bpy.ops."):
                        known.add(val)

    # Case 2: list of entries
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key in ("api_id", "id", "full_id", "name", "full_name"):
                    val = item.get(key)
                    if isinstance(val, str) and val.startswith("bpy.ops."):
                        known.add(val)

    # Fallback: regex scan all strings
    pattern = re.compile(r"\bbpy\.ops\.[A-Za-z0-9_.]+")
    for s in collect_strings(data):
        for m in pattern.findall(s):
            known.add(m)

    return known


def extract_api_ids_from_types_flat(data: Any) -> Set[str]:
    """Extract all bpy.types.* method IDs from bpy_types_flat.json"""
    known: Set[str] = set()
    
    # bpy_types_flat.json is a list of class objects
    if isinstance(data, list):
        for class_info in data:
            if isinstance(class_info, dict):
                # Get class_id
                class_id = class_info.get('class_id', '')
                if class_id and class_id.startswith('bpy.types.'):
                    known.add(class_id)
                
                # Get methods from this class
                methods = class_info.get('methods', [])
                if isinstance(methods, list):
                    for method in methods:
                        if isinstance(method, dict):
                            method_id = method.get('method_id', '')
                            if method_id and method_id.startswith('bpy.types.'):
                                known.add(method_id)
    
    # Fallback: regex scan all strings
    pattern = re.compile(r"\bbpy\.types\.[A-Za-z0-9_.]+")
    for s in collect_strings(data):
        for m in pattern.findall(s):
            known.add(m)
    
    return known


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    base_dir = Path(__file__).parent

    queries_path = Path(sys.argv[1]) if len(sys.argv) > 1 else base_dir / "queries.json"
    ops_flat_path = Path(sys.argv[2]) if len(sys.argv) > 2 else base_dir / "structured_docs" / "bpy_ops_flat.json"
    types_flat_path = Path(sys.argv[3]) if len(sys.argv) > 3 else base_dir / "structured_docs" / "bpy_types_flat.json"

    if not queries_path.exists():
        print(f"ERROR: queries.json not found: {queries_path}")
        sys.exit(2)
    if not ops_flat_path.exists():
        print(f"ERROR: bpy_ops_flat.json not found: {ops_flat_path}")
        sys.exit(2)
    if not types_flat_path.exists():
        print(f"ERROR: bpy_types_flat.json not found: {types_flat_path}")
        sys.exit(2)

    queries = load_json(queries_path)
    ops_flat = load_json(ops_flat_path)
    types_flat = load_json(types_flat_path)
    
    # Extract known API IDs from both sources
    known_ops_ids = extract_api_ids_from_ops_flat(ops_flat)
    known_types_ids = extract_api_ids_from_types_flat(types_flat)
    known_api_ids = known_ops_ids | known_types_ids

    print(f"Loaded {len(known_ops_ids)} ops APIs and {len(known_types_ids)} types APIs")

    missing = []
    total = 0
    ops_count = 0
    types_count = 0

    cases: Dict[str, Any] = queries.get("cases", {})
    for case_name, case_data in cases.items():
        ops: Dict[str, Any] = case_data.get("operations", {})
        for op_name, op_data in ops.items():
            expected_api = op_data.get("expected_api", "")
            if not expected_api:
                continue
            total += 1
            
            # Track ops vs types
            if expected_api.startswith("bpy.ops."):
                ops_count += 1
            elif expected_api.startswith("bpy.types."):
                types_count += 1
            
            if expected_api not in known_api_ids:
                missing.append((case_name, op_name, expected_api))

    if missing:
        print("\nMissing expected_api entries:")
        for case_name, op_name, api in missing:
            api_type = "ops" if api.startswith("bpy.ops.") else "types"
            print(f"- [{api_type}] {case_name} / {op_name}: {api}")
        print(f"\nSummary: {len(missing)} missing out of {total} expected_api entries")
        print(f"  - ops APIs: {ops_count}")
        print(f"  - types APIs: {types_count}")
        sys.exit(1)
    else:
        print(f"\n✓ All expected_api entries found!")
        print(f"Total checked: {total}")
        print(f"  - ops APIs: {ops_count}")
        print(f"  - types APIs: {types_count}")
        sys.exit(0)


if __name__ == "__main__":
    main()


