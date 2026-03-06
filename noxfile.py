import nox  # type: ignore[import-not-found]


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[dev]")
    session.run("pytest", "tests/", *session.posargs)
