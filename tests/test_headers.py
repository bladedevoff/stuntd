import pytest

from stuntd.proxy.headers import forwardable, forwardable_raw, jev_header, stuntd_header


def test_forwardable_drops_not_forwarded_names():
    src = [
        ("host", "proxy"),
        ("connection", "keep-alive"),
        ("transfer-encoding", "chunked"),
        ("authorization", "Bearer k"),
        ("content-type", "application/json"),
        ("x-custom", "1"),
    ]
    assert forwardable(src) == [
        ("authorization", "Bearer k"),
        ("content-type", "application/json"),
        ("x-custom", "1"),
    ]


def test_forwardable_raw_filters_without_decoding():
    src = [
        (b"Connection", b"keep-alive"),
        (b"Set-Cookie", b"a=1"),
        (b"Set-Cookie", b"b=2"),
        (b"X-Uni", "café 漢".encode()),
    ]
    assert forwardable_raw(src) == [
        (b"Set-Cookie", b"a=1"),
        (b"Set-Cookie", b"b=2"),
        (b"X-Uni", "café 漢".encode()),
    ]


def test_stuntd_header_format():
    assert stuntd_header("passthrough", reason="no-schema") == "passthrough; reason=no-schema"
    assert stuntd_header("collect", site="abc123") == "collect; site=abc123"
    assert stuntd_header("passthrough") == "passthrough"


def test_stuntd_header_names_a_missing_upstream():
    assert stuntd_header("passthrough", reason="no-upstream") == "passthrough; reason=no-upstream"


def test_stuntd_header_carries_reason_and_confidence():
    assert stuntd_header("live", site="s", confidence=0.973) == "live; site=s; confidence=0.97"
    assert stuntd_header("collect", site="s", reason="check") == "collect; site=s; reason=check"


@pytest.mark.parametrize(
    "field", [{"site": "a\r\nx-evil: 1"}, {"reason": "not a token"}], ids=["crlf", "space"]
)
def test_stuntd_header_rejects_a_value_outside_its_charset(field):
    with pytest.raises(ValueError):
        stuntd_header("collect", **field)


def test_jev_header_format():
    assert jev_header("local", questions=3, live=1, shadow=1, zeroshot=1) == (
        "jev; mode=local; questions=3; live=1; shadow=1; zeroshot=1"
    )
    assert jev_header("local", reason="no-runtime") == "jev; mode=local; reason=no-runtime"
    assert jev_header("proxy", questions=2, live=0, shadow=1, check=1) == (
        "jev; mode=proxy; questions=2; live=0; shadow=1; check=1"
    )
    assert jev_header("proxy", questions=0) == "jev; mode=proxy; questions=0"


@pytest.mark.parametrize(
    "field", [{"reason": "no runtime"}, {"mode": "local\r\nx-evil: 1"}], ids=["space", "crlf"]
)
def test_jev_header_rejects_a_value_outside_its_charset(field):
    mode = field.pop("mode", "local")
    with pytest.raises(ValueError):
        jev_header(mode, **field)
