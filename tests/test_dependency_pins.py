"""Guard the FastMCP / MCP SDK major versions this server is written against.

This repo is deliberately **not** on PyPI: the documented install path is
``pip install git+https://github.com/AlanOgic/odoo-mcp-19.git``, which resolves
dependencies fresh and ignores ``uv.lock``. An unbounded ``fastmcp`` requirement
therefore drifts silently on every install — and FastMCP 4.x is not a drop-in:

  - it targets MCP spec 2026-07-28 (stateless core, no server-initiated
    requests), so ``ctx.elicit()`` in ``configure_odoo`` raises at runtime;
  - it moves to MCP SDK v2 (``mcp>=2.0`` + the split-out ``mcp-types``), which
    changes the ``mcp.types`` imports in ``app.py`` / ``skill_visibility.py``;
  - the ``[tasks]`` extra becomes the separate ``fastmcp-tasks`` package and
    task-enabled tools need an explicit ``TasksExtension``, so ``batch_execute``
    and ``execute_workflow`` fail to register without it.

Two independent assertions, because they catch different failures: the runtime
check catches a stale or bypassed lock, the declared check catches somebody
widening the constraint without doing the migration.

See ``docs/mcp-2026-07-28-migration.md`` before raising either bound.
"""

import importlib.metadata as metadata
import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only exercised on 3.10
    tomllib = pytest.importorskip("tomli", reason="3.10 needs tomli to read pyproject")

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"

MIGRATION_DOC = "docs/mcp-2026-07-28-migration.md"

# Majors this codebase is written against.
FASTMCP_MAJOR = 3
MCP_SDK_MAJOR = 1


def _declared_dependencies() -> list[str]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["dependencies"]


def _fastmcp_requirement() -> Requirement:
    for raw in _declared_dependencies():
        req = Requirement(raw)
        if req.name == "fastmcp":
            return req
    raise AssertionError("pyproject.toml declares no 'fastmcp' dependency")


class TestInstalledVersions:
    """The environment actually in use must match the supported majors."""

    def test_fastmcp_major_is_supported(self):
        installed = Version(metadata.version("fastmcp"))
        assert installed.major == FASTMCP_MAJOR, (
            f"fastmcp {installed} is installed but this server targets"
            f" {FASTMCP_MAJOR}.x. FastMCP 4.x breaks elicitation, background"
            f" tasks, and the mcp.types imports — see {MIGRATION_DOC}."
        )

    def test_mcp_sdk_major_is_supported(self):
        installed = Version(metadata.version("mcp"))
        assert installed.major == MCP_SDK_MAJOR, (
            f"mcp SDK {installed} is installed but this server targets"
            f" {MCP_SDK_MAJOR}.x. SDK v2 moves the wire types to the mcp-types"
            f" package and renames attributes to snake_case, which changes"
            f" app.py and skill_visibility.py — see {MIGRATION_DOC}."
        )

    def test_tasks_extra_is_installed(self):
        """``fastmcp[tasks]`` pulls pydocket on the 3.x line."""
        metadata.version("pydocket")


class TestDeclaredConstraint:
    """pyproject.toml must keep the ceiling that makes the above true."""

    def test_fastmcp_requirement_declares_tasks_extra(self):
        req = _fastmcp_requirement()
        assert "tasks" in req.extras, (
            "fastmcp must be requested as fastmcp[tasks]: batch_execute and"
            " execute_workflow are declared with task=True and will not"
            " register without the extra."
        )

    def test_fastmcp_requirement_excludes_next_major(self):
        req = _fastmcp_requirement()
        next_major = Version(f"{FASTMCP_MAJOR + 1}.0.0")
        assert not req.specifier.contains(next_major, prereleases=True), (
            f"pyproject.toml allows fastmcp {next_major}; the declared"
            f" constraint is {req.specifier or '(none)'}. Restore the '<"
            f"{FASTMCP_MAJOR + 1}' bound, or complete the migration in"
            f" {MIGRATION_DOC} first."
        )

    def test_fastmcp_floor_matches_tested_baseline(self):
        """The floor must not fall below the version the suite is run against."""
        req = _fastmcp_requirement()
        installed = Version(metadata.version("fastmcp"))
        older = Version(f"{installed.major}.{installed.minor}.0")
        if older >= installed:  # installed is an x.y.0 release
            older = Version(f"{installed.major}.{max(installed.minor - 1, 0)}.0")
        assert not req.specifier.contains(older), (
            f"pyproject.toml allows fastmcp {older}, older than the tested"
            f" baseline {installed}. Raise the floor so a fresh install gets"
            " the version this suite actually verified."
        )


def _dockerfile_instructions() -> str:
    """The Dockerfile with comment lines stripped.

    Comments legitimately *name* dependencies while explaining the rule, so
    they must not be mistaken for an install instruction.
    """
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    return "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))


def _pip_install_commands(instructions: str) -> list[str]:
    """Every `pip install` invocation, with backslash continuations joined.

    Joining matters: the drifted list this module guards against lived on
    continuation lines under a single ``pip install \\``, where a per-line
    scan would never see the package names next to the command.
    """
    joined = instructions.replace("\\\n", " ")
    return [line for line in joined.splitlines() if "pip install" in line]


class TestDockerfileUsesDeclaredDependencies:
    """The image must inherit pyproject's constraints, not restate them.

    The Dockerfile used to `pip install` a hand-copied requirement list and
    then `pip install --no-deps .`, so pyproject's `dependencies` never applied
    inside the image. That list had already drifted: it carried
    `fastmcp[tasks]>=3.2.0`, silently discarding the `<4` ceiling every other
    test in this module defends, and it omitted `cryptography>=42` entirely —
    token_crypto imported only because Authlib, a transitive FastMCP
    dependency, happens to pull cryptography in. Both are the same defect: a
    second, unchecked source of truth for the dependency set.
    """

    def test_dockerfile_does_not_restate_dependency_constraints(self):
        commands = _pip_install_commands(_dockerfile_instructions())
        restated = [
            req.name
            for req in map(Requirement, _declared_dependencies())
            if any(re.search(rf"\b{re.escape(req.name)}\b", command) for command in commands)
        ]
        assert not restated, (
            f"Dockerfile pins {', '.join(restated)} itself instead of taking"
            " them from pyproject.toml. A duplicated list drifts silently —"
            " that is how cryptography went missing and how the fastmcp <"
            f"{FASTMCP_MAJOR + 1} ceiling was lost. Install the project"
            " (`pip install .`) so the declared constraints apply."
        )

    def test_dockerfile_installs_the_project_with_dependencies(self):
        """At least one `pip install` must resolve deps from the project."""
        commands = _pip_install_commands(_dockerfile_instructions())
        installs_with_deps = [command for command in commands if "--no-deps" not in command]
        assert installs_with_deps, (
            "Dockerfile never runs a dependency-resolving `pip install`."
            " Every install is --no-deps, so the image would ship without"
            " fastmcp, requests, python-dotenv or cryptography."
        )
