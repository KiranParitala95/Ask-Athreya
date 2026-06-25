"""Tests for the exit-intent detector.

The detector has to balance two goals: catch the many natural ways a user says
they want to leave, while never mistaking a real data question for an exit.
"""

import pytest

from main import is_exit_request


@pytest.mark.parametrize(
    "text",
    [
        "exit",
        "quit",
        "q",
        "bye",
        "Bye!",
        "goodbye",
        "goodbye.",
        "stop",
        "end",
        "i want to exit",
        "i want to quit",
    ],
)
def test_exit_phrases_are_detected(text):
    assert is_exit_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "how many students got an A",
        "who scored the highest in math",
        "average percentage",
        "which subject has the lowest average",
        "name the top 5 students",
        "",
    ],
)
def test_real_questions_are_not_treated_as_exit(text):
    assert is_exit_request(text) is False
