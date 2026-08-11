import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from lmflow.agentic import EpisodeWorkspace, PreparedRepositoryCache, PreparedRepositoryCacheError

pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="repository cache locking requires POSIX"),
    pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for repository cache tests"),
]


def _git(repo, *arguments):
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        capture_output=True,
    ).stdout


def _create_source_repo(path):
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "LMFlow Test")
    _git(path, "config", "user.email", "lmflow@example.invalid")
    (path / "value.txt").write_text("first\n", encoding="utf-8")
    _git(path, "add", "value.txt")
    _git(path, "commit", "--quiet", "-m", "First fixture")
    return _git(path, "rev-parse", "HEAD").decode("ascii").strip()


def _new_cache(tmp_path, **kwargs):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    return PreparedRepositoryCache(cache_root, **kwargs)


def test_prepares_commit_for_episode_workspace_and_reuses_it_without_source(tmp_path):
    source = tmp_path / "source"
    revision = _create_source_repo(source)
    cache = _new_cache(tmp_path)

    prepared = cache.prepare(str(source), revision.upper())

    assert _git(prepared, "rev-parse", "--is-bare-repository").strip() == b"true"
    assert _git(prepared, "rev-parse", "refs/lmflow/base").decode("ascii").strip() == revision
    episode_root = tmp_path / "episodes"
    episode_root.mkdir()
    with EpisodeWorkspace.create(
        episode_root,
        source_repo=prepared,
        revision=revision,
        task_id="task-1",
        rollout_id="rollout-1",
    ) as workspace:
        assert (workspace.path / "value.txt").read_text(encoding="utf-8") == "first\n"

    source.rename(tmp_path / "source-moved")
    assert cache.prepare(str(source), revision) == prepared
    assert list(cache._staging_root.iterdir()) == []


def test_distinct_commits_publish_distinct_entries(tmp_path):
    source = tmp_path / "source"
    first_revision = _create_source_repo(source)
    (source / "value.txt").write_text("second\n", encoding="utf-8")
    _git(source, "add", "value.txt")
    _git(source, "commit", "--quiet", "-m", "Second fixture")
    second_revision = _git(source, "rev-parse", "HEAD").decode("ascii").strip()
    cache = _new_cache(tmp_path)

    first = cache.prepare(str(source), first_revision)
    second = cache.prepare(str(source), second_revision)

    assert first != second
    assert _git(first, "rev-parse", "refs/lmflow/base").decode("ascii").strip() == first_revision
    assert _git(second, "rev-parse", "refs/lmflow/base").decode("ascii").strip() == second_revision


def test_concurrent_callers_share_one_atomic_entry(tmp_path):
    source = tmp_path / "source"
    revision = _create_source_repo(source)
    cache = _new_cache(tmp_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        prepared = list(executor.map(lambda _: cache.prepare(str(source), revision), range(4)))

    assert len(set(prepared)) == 1
    assert len(list(cache._repositories_root.iterdir())) == 1
    assert list(cache._staging_root.iterdir()) == []


def test_failed_fetch_does_not_publish_or_leave_staging(tmp_path):
    source = tmp_path / "source"
    _create_source_repo(source)
    cache = _new_cache(tmp_path)

    with pytest.raises(PreparedRepositoryCacheError, match="git fetch failed"):
        cache.prepare(str(source), "f" * 40)

    assert list(cache._repositories_root.iterdir()) == []
    assert list(cache._staging_root.iterdir()) == []


def test_corrupt_published_entry_fails_closed(tmp_path):
    source = tmp_path / "source"
    revision = _create_source_repo(source)
    cache = _new_cache(tmp_path)
    prepared = cache.prepare(str(source), revision)
    (prepared.parent / "READY").unlink()

    with pytest.raises(PreparedRepositoryCacheError, match="incomplete or unsafe"):
        cache.prepare(str(source), revision)


@pytest.mark.parametrize(
    ("repository", "error"),
    [
        ("", "non-empty"),
        ("http://example.invalid/repo.git", "scheme"),
        ("https://token@example.invalid/repo.git", "credentials"),
        ("https://example.invalid/repo.git?token=secret", "query strings"),
        ("git@example.invalid:repo.git", "SCP-style"),
    ],
)
def test_rejects_unsupported_repository_sources(tmp_path, repository, error):
    cache = _new_cache(tmp_path)

    with pytest.raises((ValueError, PreparedRepositoryCacheError), match=error):
        cache.prepare(repository, "0" * 40)


def test_rejects_invalid_root_revision_and_timeouts(tmp_path):
    with pytest.raises(PreparedRepositoryCacheError, match="existing directory"):
        PreparedRepositoryCache(tmp_path / "missing")
    cache = _new_cache(tmp_path)
    with pytest.raises(ValueError, match="40-character"):
        cache.prepare(str(tmp_path / "source"), "main")
    with pytest.raises(ValueError, match="finite and positive"):
        PreparedRepositoryCache(tmp_path / "cache", lock_timeout_seconds=0)
