import pytest

from stuntd.jev.state import serialize_state


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("plain text", "plain text"),
        ("спам?", "спам?"),
        ('{"already": "json"}', '{"already": "json"}'),
        ({"board": [1, 2], "turn": "white"}, '{"board": [1, 2], "turn": "white"}'),
        ({"кто": "ход"}, '{"кто": "ход"}'),
        ([{"role": "user", "content": "привет"}], '[{"role": "user", "content": "привет"}]'),
        ([], "[]"),
    ],
    ids=["text", "non-ascii-text", "json-text", "object", "non-ascii-object", "turns", "empty"],
)
def test_state_is_serialized_the_way_laya_does(state, expected):
    assert serialize_state(state) == expected


def test_object_keys_keep_their_order():
    assert serialize_state({"b": 1, "a": 2}) == '{"b": 1, "a": 2}'
