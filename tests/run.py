"""Run behavioral tests with concise per-test status output."""

from __future__ import annotations

import shutil
import sys
import unittest


class OutputStream:
    """Add unittest's ``writeln`` protocol to a regular text stream."""

    def __init__(self, stream: object) -> None:
        self.stream = stream

    def write(self, text: str) -> None:
        self.stream.write(text)  # type: ignore[attr-defined]

    def flush(self) -> None:
        self.stream.flush()  # type: ignore[attr-defined]

    def writeln(self, text: str = "") -> None:
        self.write(f"{text}\n")


class ConciseResult(unittest.TextTestResult):
    """Print a test method name once, followed by its final status."""

    def _status_line(self, test: unittest.TestCase, status: str) -> str:
        columns = min(shutil.get_terminal_size((80, 20)).columns, 120)
        name = test._testMethodName
        padding = max(1, columns - 2 - len(name) - len(status))
        return f"{name}{' ' * padding}{status}"

    def startTest(self, test: unittest.TestCase) -> None:
        unittest.TestResult.startTest(self, test)

    def addSuccess(self, test: unittest.TestCase) -> None:
        unittest.TestResult.addSuccess(self, test)
        self.stream.writeln(self._status_line(test, "OK"))

    def addFailure(
        self, test: unittest.TestCase, err: tuple[object, object, object]
    ) -> None:
        unittest.TestResult.addFailure(self, test, err)
        self.stream.writeln(self._status_line(test, "FAIL"))

    def addError(
        self, test: unittest.TestCase, err: tuple[object, object, object]
    ) -> None:
        unittest.TestResult.addError(self, test, err)
        self.stream.writeln(self._status_line(test, "ERROR"))

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        unittest.TestResult.addSkip(self, test, reason)
        self.stream.writeln(self._status_line(test, f"SKIP ({reason})"))


def main() -> int:
    suite = unittest.defaultTestLoader.discover("tests")
    result = ConciseResult(
        stream=OutputStream(sys.stdout),
        descriptions=False,
        verbosity=0,
    )
    result.startTestRun()
    try:
        suite.run(result)
    finally:
        result.stopTestRun()
    if not result.wasSuccessful():
        result.printErrors()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
