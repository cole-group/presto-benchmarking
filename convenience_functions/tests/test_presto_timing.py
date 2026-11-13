from convenience_functions.presto_timing import SpanTimer


def test_other_subtracts_only_direct_children() -> None:
    """``(other)`` rows hold the time a span spends outside its direct children.

    With nested spans a -> b -> c, ``a/(other)`` must be ``a - b``, not
    ``a - b - c``: c's time is already inside b, so subtracting it again would
    count it twice and understate the unaccounted time. Leaf spans (c) get no
    ``(other)`` row.
    """
    timer = SpanTimer()
    c = timer.wrap(lambda: None, "c")
    b = timer.wrap(lambda: c(), "b")
    a = timer.wrap(lambda: b(), "a")
    a()

    rows = timer.summary()
    total = {path: row["total_s"] for path, row in rows.items()}
    assert set(total) == {"a", "a/b", "a/b/c", "a/(other)", "a/b/(other)"}
    assert total["a/(other)"] == total["a"] - total["a/b"]
    assert total["a/b/(other)"] == total["a/b"] - total["a/b/c"]
