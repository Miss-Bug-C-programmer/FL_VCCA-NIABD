import pytest

from method_switches import resolve_method_switches, strategy_name


def test_no_method_arguments_resolve_to_baseline():
    switches = resolve_method_switches(None)
    assert switches.method == "baseline"
    assert switches.enable_vcaa is False
    assert switches.enable_niabd is False
    assert switches.admission_enabled is False
    assert switches.defense_enabled is False


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("baseline", (False, False)),
        ("vcaa", (True, False)),
        ("niabd", (False, True)),
        ("vcaa-niabd", (True, True)),
    ],
)
def test_method_aliases_map_to_independent_switches(method, expected):
    switches = resolve_method_switches(method)
    assert (switches.enable_vcaa, switches.enable_niabd) == expected
    assert switches.method == method
    assert switches.admission_enabled is expected[0]
    assert switches.defense_enabled is expected[1]


@pytest.mark.parametrize(
    ("enable_vcaa", "enable_niabd", "expected_method"),
    [
        (False, False, "baseline"),
        (True, False, "vcaa"),
        (False, True, "niabd"),
        (True, True, "vcaa-niabd"),
    ],
)
def test_legacy_enable_flags_remain_supported(
    enable_vcaa,
    enable_niabd,
    expected_method,
):
    switches = resolve_method_switches(
        None,
        enable_vcaa=enable_vcaa,
        enable_niabd=enable_niabd,
    )
    assert switches.method == expected_method


def test_method_and_conflicting_legacy_flags_fail_closed():
    with pytest.raises(ValueError, match="conflicts"):
        resolve_method_switches("niabd", enable_vcaa=True)


def test_strategy_name_has_no_implicit_mechanism_enablement():
    assert strategy_name(False, False) == "baseline"
    assert strategy_name(True, False) == "vcaa"
    assert strategy_name(False, True) == "niabd"
    assert strategy_name(True, True) == "vcaa-niabd"
