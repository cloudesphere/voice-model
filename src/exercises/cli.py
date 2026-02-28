import argparse
import json
import sys
from pathlib import Path

from .store import ExerciseStore, DEFAULT_INDEX_PATH

def cmd_list(store: ExerciseStore, args: argparse.Namespace) -> int:
    items = store.list_items()
    for it in items:
        print(f"{it.exercise_id}\t{it.type}\t{it.difficulty}\t{it.ref_clip}")
    return 0

def cmd_random(store: ExerciseStore, args: argparse.Namespace) -> int:
    ex = store.get_random(type_filter=args.type, max_difficulty=args.max_difficulty)
    print(json.dumps(ex, ensure_ascii=False, indent=2))
    return 0

def cmd_show(store: ExerciseStore, args: argparse.Namespace) -> int:
    ex = store.get_by_id(args.exercise_id)
    print(json.dumps(ex, ensure_ascii=False, indent=2))
    return 0

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="exercises-cli")
    p.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help="Path to exercise_index.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="List exercises")
    sp_list.set_defaults(func=cmd_list)

    sp_rand = sub.add_parser("random", help="Pick a random exercise")
    sp_rand.add_argument("--type", default=None, help="Filter by type (makharij/phrase/control)")
    sp_rand.add_argument("--max-difficulty", type=int, default=None, help="Max difficulty")
    sp_rand.set_defaults(func=cmd_random)

    sp_show = sub.add_parser("show", help="Show an exercise by id")
    sp_show.add_argument("exercise_id")
    sp_show.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    store = ExerciseStore(index_path=Path(args.index))
    store.load()
    return args.func(store, args)

if __name__ == "__main__":
    raise SystemExit(main())
