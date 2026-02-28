import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_INDEX_PATH = Path("artifacts/exercise_index.json")

@dataclass(frozen=True)
class ExerciseIndexItem:
    exercise_id: str
    type: str
    difficulty: int
    reader: str
    segment_id: Optional[str]
    ref_clip: Optional[str]
    path: str

class ExerciseStore:
    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH):
        self.index_path = index_path
        self._index: Optional[Dict[str, Any]] = None
        self._items: List[ExerciseIndexItem] = []
        self._by_id: Dict[str, ExerciseIndexItem] = {}

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Exercise index not found: {self.index_path}")

        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        items = data.get("items", [])
        parsed: List[ExerciseIndexItem] = []

        for it in items:
            obj = ExerciseIndexItem(
                exercise_id=it["exercise_id"],
                type=it["type"],
                difficulty=int(it.get("difficulty", 1)),
                reader=it.get("reader", "husary"),
                segment_id=it.get("segment_id"),
                ref_clip=it.get("ref_clip"),
                path=it["path"],
            )
            parsed.append(obj)

        self._index = data
        self._items = parsed
        self._by_id = {x.exercise_id: x for x in parsed}

    def is_loaded(self) -> bool:
        return self._index is not None

    def list_items(self) -> List[ExerciseIndexItem]:
        if not self.is_loaded():
            self.load()
        return list(self._items)

    def get_by_id(self, exercise_id: str) -> Dict[str, Any]:
        if not self.is_loaded():
            self.load()
        item = self._by_id.get(exercise_id)
        if not item:
            raise KeyError(f"Exercise not found: {exercise_id}")
        path = Path(item.path)
        if not path.exists():
            raise FileNotFoundError(f"Exercise file missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_random(self, *, type_filter: Optional[str] = None, max_difficulty: Optional[int] = None) -> Dict[str, Any]:
        if not self.is_loaded():
            self.load()

        pool = self._items
        if type_filter:
            pool = [x for x in pool if x.type == type_filter]
        if max_difficulty is not None:
            pool = [x for x in pool if x.difficulty <= max_difficulty]

        if not pool:
            raise ValueError("No exercises match the given filters")

        pick = random.choice(pool)
        return self.get_by_id(pick.exercise_id)
