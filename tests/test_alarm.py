from __future__ import annotations

from fallsense.alarm import AlarmConfig, AlarmState, AlarmStateMachine


def test_alarm_requires_consecutive_windows_and_latches() -> None:
    machine = AlarmStateMachine(AlarmConfig(threshold=0.6, consecutive_windows=2))
    states = [machine.step(value).state for value in (0.7, 0.2, 0.8, 0.9, 0.1)]
    assert states == [
        AlarmState.PENDING,
        AlarmState.NORMAL,
        AlarmState.PENDING,
        AlarmState.ALARM,
        AlarmState.ALARM,
    ]


def test_acknowledge_enters_cooldown_then_rearms() -> None:
    machine = AlarmStateMachine(
        AlarmConfig(threshold=0.5, consecutive_windows=1, cooldown_windows=2)
    )
    assert machine.step(0.8).state is AlarmState.ALARM
    acknowledged = machine.step(0.8, acknowledge=True)
    assert acknowledged.acknowledged
    assert acknowledged.state is AlarmState.COOLDOWN
    assert machine.step(0.9).state is AlarmState.COOLDOWN
    assert machine.step(0.9).state is AlarmState.ALARM
