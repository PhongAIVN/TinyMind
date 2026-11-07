import pytest

from trainer.lr_scheduler import get_lr

MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP = 10
MAX_STEPS = 100
SCHEDULE = dict(max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=WARMUP, max_steps=MAX_STEPS)


def test_first_step_is_small():
    assert get_lr(0, **SCHEDULE) == pytest.approx(MAX_LR / WARMUP)


def test_warmup_ends_at_max_lr():
    assert get_lr(WARMUP - 1, **SCHEDULE) == pytest.approx(MAX_LR)
    assert get_lr(WARMUP, **SCHEDULE) == pytest.approx(MAX_LR)


def test_last_step_is_close_to_min_lr():
    assert get_lr(MAX_STEPS - 1, **SCHEDULE) == pytest.approx(MIN_LR, rel=0.01)


def test_after_max_steps_stays_at_min_lr():
    assert get_lr(MAX_STEPS, **SCHEDULE) == MIN_LR
    assert get_lr(MAX_STEPS * 10, **SCHEDULE) == MIN_LR


def test_lr_increases_during_warmup():
    values = [get_lr(step, **SCHEDULE) for step in range(WARMUP)]

    assert values == sorted(values)
    assert values[0] < values[-1]


def test_lr_decreases_after_warmup():
    values = [get_lr(step, **SCHEDULE) for step in range(WARMUP, MAX_STEPS)]

    assert values == sorted(values, reverse=True)


def test_lr_always_within_bounds():
    for step in range(MAX_STEPS + 20):
        assert 0 < get_lr(step, **SCHEDULE) <= MAX_LR


def test_no_warmup_starts_at_max_lr():
    assert get_lr(0, max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=0, max_steps=MAX_STEPS) == MAX_LR


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(step=-1, max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=WARMUP, max_steps=MAX_STEPS),
        dict(step=0, max_lr=MIN_LR, min_lr=MAX_LR, warmup_steps=WARMUP, max_steps=MAX_STEPS),
        dict(step=0, max_lr=MAX_LR, min_lr=MIN_LR, warmup_steps=MAX_STEPS + 1, max_steps=MAX_STEPS),
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        get_lr(**kwargs)