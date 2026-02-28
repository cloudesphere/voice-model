import json, os, re

INDEX = "artifacts/exercise_index.json"

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def pretty_title(ex_id: str) -> str:
    s = ex_id.replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def default_prompt(ex_type: str, ex_id: str) -> str:
    t = (ex_type or "").lower()
    if t == "makharij":
        return f"تمرّن على المخرج المستهدف في هذا التمرين ({ex_id}). كرر الكلمة ببطء 5 مرات ثم بسرعة طبيعية 3 مرات."
    if t == "phrase":
        return f"كرر العبارة المرتبطة بهذا التمرين ({ex_id}) 3 مرات مع وضوح الحروف وعدم دمج الكلمات."
    if t == "control":
        return f"اقرأ المقطع طبيعيًا بدون مبالغة ({ex_id}) مع ثبات السرعة ووضوح النطق."
    return f"نفّذ التمرين ({ex_id}) وفق الإرشادات."

def extract_targets(ex_obj: dict, ex_type: str, ex_id: str):
    # Best effort: look for common fields; fallback based on id/type
    for key in ("targets", "tags", "focus", "confusions", "makharij", "letters"):
        v = ex_obj.get(key)
        if isinstance(v, list) and v:
            return v
        if isinstance(v, str) and v:
            return [v]

    # heuristic from id
    if "sad_seen" in ex_id or "sad_vs_seen" in ex_id:
        return ["makharij", "ص↔س"]
    if "wala_daalin" in ex_id or "waddaalin" in ex_id:
        return ["phrase", "ولا الضالين"]
    return [ex_type] if ex_type else []

def main():
    if not os.path.exists(INDEX):
        raise SystemExit(f"Index not found: {INDEX}")

    idx = load_json(INDEX)

    # normalize items
    if isinstance(idx, dict):
        items = idx.get("exercises") or idx.get("items") or []
        if not isinstance(items, list):
            raise SystemExit("Index dict has no list in exercises/items")
        # unify under 'exercises'
        idx["exercises"] = items
        idx.pop("items", None)
    elif isinstance(idx, list):
        items = idx
        idx = {"exercises": items}
    else:
        raise SystemExit("Unsupported index shape")

    rebuilt = []
    missing_files = 0

    for rec in items:
        if not isinstance(rec, dict):
            continue
        ex_id = rec.get("exercise_id") or rec.get("id") or rec.get("key")
        path = rec.get("path")
        ex_type = rec.get("type")

        if not ex_id:
            continue

        ex_obj = {}
        if path and isinstance(path, str) and os.path.exists(path):
            try:
                ex_obj = load_json(path)
                if isinstance(ex_obj, dict) and "exercise" in ex_obj and isinstance(ex_obj["exercise"], dict):
                    ex_obj = ex_obj["exercise"]
                if not isinstance(ex_obj, dict):
                    ex_obj = {}
            except Exception:
                ex_obj = {}
        else:
            missing_files += 1

        title = ex_obj.get("title") or ex_obj.get("name") or pretty_title(str(ex_id))
        prompt = ex_obj.get("prompt") or ex_obj.get("instruction") or ex_obj.get("text") or default_prompt(str(ex_type), str(ex_id))
        targets = extract_targets(ex_obj, str(ex_type), str(ex_id))

        rebuilt.append({
            "id": str(ex_id),               # ✅ engine reads 'id'
            "exercise_id": str(ex_id),      # keep original
            "type": ex_type,
            "title": title,
            "prompt": prompt,
            "targets": targets,
            "tags": ex_obj.get("tags") or [],
            "difficulty": rec.get("difficulty") or ex_obj.get("difficulty"),
            "reader": rec.get("reader") or ex_obj.get("reader"),
            "segment_id": rec.get("segment_id") or ex_obj.get("segment_id"),
            "ref_clip": rec.get("ref_clip") or ex_obj.get("ref_clip"),
            "path": path,
        })

    idx["exercises"] = rebuilt
    dump_json(INDEX, idx)

    print(f"✅ Rebuilt rich index: {INDEX}")
    print(f"   exercises: {len(rebuilt)}")
    print(f"   missing path files: {missing_files}")

if __name__ == "__main__":
    main()
