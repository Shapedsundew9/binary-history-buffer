"""Binary History Buffer."""

from __future__ import annotations
from typing import Any
from logging import DEBUG, Logger, NullHandler, getLogger

from numpy import uint64, uint32, uint8, zeros, float64
from numpy.typing import NDArray


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


class binary_history_buffer_log2_table():
    """Maintains a 64 bit history of a binary state with a log2 weighting
    
    See https://github.com/Shapedsundew9/binary-history-buffer/blob/main/README.md for details.
    """

    def __init__(self, size: int = 1, nlsb: int = 0) -> None:
        """Create a binary history buffer.

        Args:
            size: Number of buffers to maintain in the table. Defaults to 1.
            nlsb: The number of guaranteed set oldest (least significant) bits. Defaults to 0.
        """
        self._size: int = size

        # Member descriptions:
        # updates: The total number of updates for each entry i.e. the number of bits in the history.
        # hits: The total number of True updates for each entry.
        # buffers: The history buffers.
        self.updates: NDArray[uint64] = zeros(self._size, dtype=uint64)
        self.hits: NDArray[uint64] = zeros(self._size, dtype=uint64)
        self.buffer: NDArray[uint64] = zeros(self._size, dtype=uint64)
        self.lsbs: uint64 = uint64((1 << nlsb) - 1)

    def __getitem__(self, entry: int) -> binary_history_buffer_log2:
        """Get the fraction of True updates for each store.

        Args:
            entry (int): The entry to get.

        Returns:
            NDArray[uint64]: The binary history buffer entry.
        """
        return binary_history_buffer_log2(self, entry)
   
    def __setitem__(self, entry: int, value: bool) -> None:
        """Insert a new value into the entry history buffer.

        Args:
            entry (int): The entry to insert the value into.
            value (bool): The value to insert.
        """
        self.updates[entry] += uint64(1)
        self.hits[entry] += uint64(value)
        self.buffer[entry] = (self.buffer[entry] >> 1) | uint64(value << 63) | self.lsbs

    def ratios(self) -> NDArray[float64]:
        """Return the normalized weighted history ratios."""
        return self.buffer / self.buffer.sum()

class binary_history_buffer_log2():
    """Maintains a compressed history of a binary state."""

    def __init__(self, bhbl2t: binary_history_buffer_log2_table | None = None, entry: int = 0) -> None:
        """Create a binary history buffer.

        Args:
            bhbt: The table to use. If None a single entry table is created.
            entry: The entry to use. Defaults to 0.
            length: Log2(length of the buffer) - 6. Defaults to 6 (4096 bits). Only used if bhbt = None
        """
        self._bhbl2t: binary_history_buffer_log2_table = bhbl2t if bhbl2t is not None else binary_history_buffer_log2_table()
        self._entry: int = entry

    def totals(self) -> tuple[uint64, uint64, float64]:
        """Get the total number of hits, updates & the ratio for the entry.

        Returns:
            (Total True updates, Total updates, Hit Ratio)
        """
        hits: uint64 = self._bhbl2t.hits[self._entry]
        updates: uint64 = self._bhbl2t.updates[self._entry]
        ratio: float64 = hits / updates
        return hits, updates, ratio
 
    def history(self) -> tuple[uint64, uint64, float64]:
        """Get the fraction of True updates for each history period.
        
        Args:
            length: The length of bit history to evaluate.

        Returns:
            (Hit Ratios, History lengths)
        """
        buffer: uint64 = self._bhbl2t.buffer[self._entry]
        updates: uint64 = min(self._bhbl2t.updates[self._entry], uint64(64))
        hits = uint64(buffer.bit_count())
        return hits, updates, hits / updates

    def update(self, value: bool) -> None:
        """Insert a new value into the entry history buffer.

        Args:
            value (bool): The value to insert.
        """
        self._bhbl2t[self._entry] = value


# Aliases
bhbl2 = binary_history_buffer_log2
bhbl2t = binary_history_buffer_log2_table