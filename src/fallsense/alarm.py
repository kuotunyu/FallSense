"""連續高風險 windows、latching alarm、manual acknowledge 與 cooldown。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlarmState(str, Enum):
    NORMAL = "normal"
    PENDING = "pending"
    ALARM = "alarm"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class AlarmConfig:
    threshold: float = 0.5
    consecutive_windows: int = 2
    cooldown_windows: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold 必須介於 0 與 1")
        if self.consecutive_windows < 1 or self.cooldown_windows < 0:
            raise ValueError("consecutive_windows 必須 >=1，cooldown_windows 必須 >=0")


@dataclass(frozen=True)
class AlarmSnapshot:
    index: int
    probability: float
    state: AlarmState
    high_streak: int
    cooldown_remaining: int
    acknowledged: bool


class AlarmStateMachine:
    def __init__(self, config: AlarmConfig) -> None:
        self.config = config
        self.high_streak = 0
        self.cooldown_remaining = 0
        self.alarm_latched = False
        self.index = -1

    def step(self, probability: float, *, acknowledge: bool = False) -> AlarmSnapshot:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability 必須介於 0 與 1")
        self.index += 1
        acknowledged = acknowledge and self.alarm_latched
        if acknowledged:
            self.alarm_latched = False
            self.high_streak = 0
            self.cooldown_remaining = self.config.cooldown_windows

        if self.cooldown_remaining > 0:
            state = AlarmState.COOLDOWN
            self.cooldown_remaining -= 1
        elif self.alarm_latched:
            state = AlarmState.ALARM
        elif probability >= self.config.threshold:
            self.high_streak += 1
            if self.high_streak >= self.config.consecutive_windows:
                self.alarm_latched = True
                state = AlarmState.ALARM
            else:
                state = AlarmState.PENDING
        else:
            self.high_streak = 0
            state = AlarmState.NORMAL

        return AlarmSnapshot(
            index=self.index,
            probability=probability,
            state=state,
            high_streak=self.high_streak,
            cooldown_remaining=self.cooldown_remaining,
            acknowledged=acknowledged,
        )
