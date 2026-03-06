import nox  # type: ignore[import-not-found]


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[dev]")
    session.run("pytest", "tests/", *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """Build the Sphinx documentation."""
    session.install(".[docs]")
    session.run("sphinx-build", "docs", "docs/_build/html", *session.posargs)
