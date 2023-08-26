"""Binary History Buffer."""

from __future__ import annotations
from typing import Any
from logging import DEBUG, Logger, NullHandler, getLogger

from numpy import int64, uint64, integer, uint8, int8, zeros, signedinteger, float64, arange, power, cumsum, log2
from numpy.typing import NDArray

from .binary_history_buffer import binary_history_buffer


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


class binary_history_buffer_z(binary_history_buffer):
    """Maintains a compressed history of a binary state."""

    def __init__(self, limit: int | integer = 0) -> None:
        """Create a compressed binary history buffer.
        
        Args
        ----

        limit: The maximum number of bits to store in the buffer. 0 = infinite.
        """
        self.limit: uint64 = uint64(limit)
        self.updates: uint64 = uint64(0)
        self.hits: uint64 = uint64(0)
        self._size: int = int(log2(limit)) - 5 if limit else 58
        self._last_index = 0
        self._last_index_updates: uint64 = uint64(0)
        self._hits: NDArray[uint64] = zeros(self._size, dtype=int8)
        self._bits: NDArray[uint64] = zeros(self._size, dtype=int8)
        self._data: NDArray[uint64] = zeros(self._size, dtype=int8)
        self._buffer: NDArray[uint64] = zeros(self._size, dtype=uint64)
        print(self)

    def _get_bit(self, index: int | integer) -> bool:
        """Get the state of a bit in the history."""
        if index < 0:
            index = max(0, self._last_index + index)
        elif index > self._last_index:
            index = self._last_index
        return bool(self._buffer[index >> 6] & (1 << (index & 0x3f)))
    
    def totals(self) -> tuple[uint64, uint64, float64]:
        """Get the total number of hits, updates & the ratio for the entry.

        Returns:
            (Total True updates, Total updates, Hit Ratio)
        """
        hits: uint64 = self._bhbl2t.hits[self._entry]
        updates: uint64 = self._bhbl2t.updates[self._entry]
        ratio: float64 = hits / updates
        return hits, updates, ratio
 
    def histories(self) -> tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]]:
        """Get the fraction of True updates for each history period.
        
        Args:
            length: The length of bit history to evaluate.

        Returns:
            (Hit Ratios, History lengths)
        """
        updates: uint64 = self._bhbl2t.updates[self._entry]
        data: NDArray[uint8] = self._bhbl2t._data[self._entry]
        hold_valid: NDArray[signedinteger[Any]] = data >> 2
        hold: NDArray[signedinteger[Any]] = (data >> 1) & hold_valid
        carry: NDArray[signedinteger[Any]] = data & 0x1
        fidelity: NDArray[int64] = power(2, arange(self._bhbl2t._length))
        store_lengths = (64 + carry + hold_valid) * fidelity  # Store may be 64, 65 or 66 bits long scaled by fidelity
        total_bits: NDArray[uint64] = cumsum(store_lengths)
        valid_stores = total_bits <= updates
        last_index: int = valid_stores.sum()
        if last_index < self._bhbl2t._length:
            total_bits[last_index:] = updates
        hits: NDArray[signedinteger[Any]] = self._bhbl2t._hits[self._entry] + carry + hold
        total_hits: NDArray[uint64] = cumsum(hits * fidelity)
        if _LOG_DEBUG:
            for s, hv, h, c, th, tb, r in zip(self._bhbl2t.buffer[self._entry], hold_valid, hold, carry, total_hits, total_bits, total_hits / total_bits):
                _logger.debug(f'Store: {s:064b}, HV: {hv}, H: {h}, C: {c}, # hits {th}, # bits {tb}, ratio {r}')
        return total_hits, total_bits, total_hits / total_bits

    def update(self, value: bool) -> None:
        """Insert a new value into the entry history buffer.

        Args:
            value (bool): The value to insert.
        """
        self._bhbl2t[self._entry] = value


# Aliases
bhbz = binary_history_buffer_z
bhbzt = binary_history_buffer_z_table