"""Module that defines data caching."""

from pathlib import Path

_cache_dir = (Path('~')/'.cache'/'esm_analysis').expanduser()
_cache_dir.mkdir(parents=True, exist_ok=True)

__all__ = ('clear_cache_dir',)


def clear_cache_dir():
    """Empty the cache directory.

    esm_analysis serializes run information and pickled datasets in the users
    cache directory (~/.cache/esm_analysis). To empty the cache directory, i.e
    if it takes up too much space this function can be called to empty the
    ENTIRE cached data.

    """
    _ = [f.unlink() for f in _cache_dir.rglob('*.[jp][sk][ol]*')]
