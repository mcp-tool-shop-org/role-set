"""Test hooks shared by the suite."""


def pytest_runtest_logreport(report):
    """xdist keeps a worker's print on the worker. The controller repeats a stratum line."""

    if report.when != "call" or not hasattr(report, "node"):
        return
    for key, value in report.user_properties:
        if key == "stratum":
            print("\n" + value, flush=True)
