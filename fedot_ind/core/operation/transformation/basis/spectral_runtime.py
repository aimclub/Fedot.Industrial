"""A corrected spectral transform must not read the legacy shape-only cache."""

from .abstract_basis import BasisDecompositionImplementation


class SpectralBasisImplementation(BasisDecompositionImplementation):
    def transform(self, input_data, use_cache=None):
        requested = self.params.get('use_cache', False) if use_cache is None else use_cache
        if not isinstance(requested, bool):
            raise ValueError('use_cache must be a boolean')
        if requested:
            raise ValueError('The shape-only spectral cache is unsafe; set use_cache=False')
        return super().transform(input_data, use_cache=False)
