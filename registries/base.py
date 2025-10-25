from dataclasses import dataclass, field
from threading import RLock
from typing import Optional, Dict


@dataclass(frozen=True)
class PositionEntry:
    code: str
    parent: Optional[str] = None


@dataclass(frozen=True)
class TeamEntry:
    code: str
    name: Optional[str] = None
    source_map: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PlayerRecord:
    id: int
    cname: str
    display_name: str
    team: str
    position: str
    source: str
    extra: Dict = field(default_factory=dict)


class SingletonMeta(type):
    _instances = {}
    _lock = RLock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseRegistry(metaclass=SingletonMeta):
    def __init__(self):
        self._lock = RLock()

    def load(self):  # pragma: no cover - interface
        raise NotImplementedError

    def save(self):  # pragma: no cover - interface
        raise NotImplementedError
