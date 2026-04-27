try:
    import fedot as fedot_api
except ImportError:  # pragma: no cover - optional dependency for local module-level imports
    fedot_api = None


__all__ = ['fedot_api']
__version__ = "0.5.0"
