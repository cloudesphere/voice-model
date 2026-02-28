import json, os, glob

INDEX = "artifacts/exercise_index.json"
ROOT = os.path.join("artifacts", "exercises")

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    if not os.path.exists(INDEX):
        raise SystemExit(f"Index not found: {INDEX}")

    idx = load_json(INDEX)

    # normalize index items list in-place
    items = None
    if isinstance(idx, dict):
        if isinstance(idx.get("exercises"), list):
            items = idx["exercises"]
        elif isinstance(idx.get("items"), list):
            items = idx["items"]
        else:
            # create exercises list if unknown
            idx["exercises"] = []
            items = idx["exercises"]
    elif isinstance(idx, list):
        # turn list into dict
        items = idx
        idx = {"exercises": items}
    else:
        raise SystemExit("Index has unsupported shape")

    # build map from exercise JSON files
    ex_files = glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True)
    ex_files = [p for p in ex_files if not p.endswith("exercise_index.json")]

    ex_by_id = {}
    for p in ex_files:
        try:
            obj = load_json(p)
            if isinstance(obj, dict) and "exercise" in obj and isinstance(obj["exercise"], dict):
                obj = obj["exercise"]
            if not isinstance(obj, dict):
                continue
            ex_id = obj.get("id") or obj.get("exercise_id") or obj.get("key")
            if not ex_id:
                continue
            obj["_path"] = p
            ex_by_id[str(ex_id)] = obj
        except Exception:
            continue

    updated = 0
    for rec in items:
        if not isinstance(rec, dict):
            continue
        rid = rec.get("id") or rec.get("exercise_id") or rec.get("key")
        if not rid:
            continue
        ex = ex_by_id.get(str(rid))
        if not ex:
            continue

        # hydrate fields if missing/null
        for k_src, k_dst in [
            ("type", "type"),
            ("title", "title"),
            ("name", "title"),
            ("prompt", "prompt"),
            ("instruction", "prompt"),
            ("text", "prompt"),
            ("targets", "targets"),
            ("tags", "tags"),
        ]:
            v = ex.get(k_src)
            if v is None:
                continue
            if rec.get(k_dst) in (None, "", [], {}):
                rec[k_dst] = v

        updated += 1

    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

    print(f"✅ Hydrated index: {INDEX}")
    print(f"   exercise files found: {len(ex_by_id)}")
    print(f"   records hydrated: {updated}")

if __name__ == "__main__":
    main()
