"""Binary History Buffer."""

from __future__ import annotations
from typing import Any, Hashable
from logging import DEBUG, Logger, NullHandler, getLogger

from numpy import int64, uint64, uint32, uint8, int8, zeros, signedinteger, float64, arange, power, cumsum
from numpy.typing import NDArray


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


class binary_history_buffer_z_table():
    """Maintains a compressed history of a binary state.
    
    See https://github.com/Shapedsundew9/binary-history-buffer/blob/main/README.md for details.
    """

    def __init__(self, size: int = 1, length: uint8 | int = 6) -> None:
        """Create a binary history buffer.

        Args:
            size: Number of buffers to maintain in the table. Defaults to 1.
            length: Log2(length of the buffer) - 6. Defaults to 6 (4096 bits)
        """
        self._size: int = size
        self._length: uint8 = uint8(length)

        # Member descriptions:
        # updates: The total number of updates for each entry i.e. the number of bits in the history.
        # hits: The total number of True updates for each entry.
        # buffers: The history buffers.
        # _data: The carry, hold, and hold_valid bits for each store.
        # _hits: The number of True updates for each store (exclusing hold and carry bits)
        self.updates: NDArray[uint32] = zeros(self._size, dtype=uint64)
        self.hits: NDArray[uint32] = zeros(self._size, dtype=uint64)
        self.buffer: NDArray[uint64] = zeros((self._size, self._length), dtype=uint64)
        self._data: NDArray[uint8] = zeros((self._size, self._length), dtype=uint8)
        self._hits: NDArray[int8] = zeros((self._size, self._length), dtype=int8)

    def __getitem__(self, entry: int) -> binary_history_buffer_z:
        """Get the fraction of True updates for each store.

        Args:
            entry (int): The entry to get.

        Returns:
            NDArray[uint64]: The binary history buffer entry.
        """
        return binary_history_buffer_z(self, entry)
   
    def __setitem__(self, entry: int, value: bool) -> None:
        """Insert a new value into the entry history buffer.

        Args:
            entry (int): The entry to insert the value into.
            value (bool): The value to insert.
        """
        self.updates[entry] += 1
        self.hits[entry] += int8(value)
        for idx in range(self._length):
            data: uint8 = self._data[entry][idx]
            carry: bool = bool(data & 0x1)
            hold: bool = bool(data & 0x2)
            hold_valid: bool = bool(data & 0x4)
            store: uint64 = self.buffer[entry][idx]
            evicted_bit: uint8 = uint8(store >> uint64(63))
            self.buffer[entry][idx] = (store << uint64(1)) | uint64(value)
            self._hits[entry][idx] += int8(value) - int8(evicted_bit)
            if hold_valid:
                state: uint8 = hold + evicted_bit + carry
                carry = bool(state & 1)
                value = bool(state >= 2)
                self._data[entry][idx] = uint8(carry)
            else:
                self._data[entry][idx] = uint8(4 + evicted_bit * 2 + carry)
                break


class mapped_binary_history_buffer_log2_table(binary_history_buffer_z_table):
    """Maps a key to a binary history buffer compressed table entry."""

    def __init__(self, size: int = 1, length: int | uint8 = 6) -> None:
        """Create a binary history buffer log2 table with a key index.

        Args:
            size: Number of buffers to maintain in the table. Defaults to 1.
            length: Log2(length of the buffer) - 6. Defaults to 6 (4096 bits)
        """
        super().__init__(size, length)
        self._keys: dict[Hashable, int] = {}

    def __getitem__(self, key: Hashable) -> binary_history_buffer_z:
        """Get the binary history buffer associated with the key.

        Args:
            key: The key of the buffer to get.

        Returns:
            The binary history buffer.
        """
        return super()[self._keys[key]]

    def __setitem__(self, key: Hashable, value: bool) -> None:
        """Insert a new value into the entry history buffer.

        Args:
            key: The key of the buffer to insert the value into.
            value (bool): The value to insert.
        """
        super()[self._keys[key]] = value


class binary_history_buffer_z():
    """Maintains a compressed history of a binary state."""

    def __init__(self, bhbl2t: binary_history_buffer_z_table | None = None, entry: int = 0, length: uint8 | int = 6) -> None:
        """Create a binary history buffer.

        Args:
            bhbt: The table to use. If None a single entry table is created.
            entry: The entry to use. Defaults to 0.
            length: Log2(length of the buffer) - 6. Defaults to 6 (4096 bits). Only used if bhbt = None
        """
        self._bhbl2t: binary_history_buffer_z_table = bhbl2t if bhbl2t is not None else binary_history_buffer_z_table(length=length)
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