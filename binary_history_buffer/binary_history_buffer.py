"""Binary History Buffer."""

from numpy import uint64, uint32, uint8, zeros, unpackbits
from numpy.typing import NDArray


class binary_history_buffer():
    """Maintains a compressed history of a binary state.
    
    See https://github.com/Shapedsundew9/binary-history-buffer/blob/main/README.md for details.
    """

    def __init__(self, size: int = 13, length: int = 7) -> None:
        """Create a binary history buffer.

        Args:
            size (int, optional): log2 number of buffers to maintain. Defaults to 13 (8192 entries).
            length (int, optional): Length of the buffer in indexes. Defaults to 7 (8192 bits)
        """
        self._size: int = 2**size
        self._length: int = length
        self.updates: NDArray[uint32] = zeros(self._size, dtype=uint32)
        self.buffers: NDArray[uint64] = zeros((self._size, self._length), dtype=uint64)
        self._data: NDArray[uint8] = zeros((self._size, self._length), dtype=uint8)

    def __getitem__(self, entry: int) -> list[float]:
        """Get the fraction of True updates for each store.

        Args:
            entry (int): The entry to get.

        Returns:
            NDArray[uint64]: The binary history buffer entry.
        """
        buf: NDArray[uint64] = self.buffers[entry]
        updates: NDArray[uint32] = self.updates[entry]
        data: NDArray[uint32] = self._data[entry]
        set_bits: list[int] = [unpackbits(s).sum() for n, s in enumerate(buf, 6) if 2**n - 64 < updates]
        extra_set_bits: list[int] = [(d & 0x1) + (((d >> 1) & 0x1) & (d >> 2)) for d in data[:len(set_bits)]]
        num_bits: list[int] = [64 + (d & 0x1) + (d >> 2) for d in data[:len(set_bits)]]
        last_index: int = len(set_bits) - 1
        num_bits[-1] = int((self.updates[entry] - (2**(last_index+6) - 64)) / 2**last_index)
        return [(s + e) / n for s, e, n in zip(set_bits, extra_set_bits, num_bits)]

    def bit_ranges(self) -> tuple[tuple[int, int], ...]:
        """Get the bit range for each store."""
        return tuple((2**(n + 6) - 64, 2**(n + 7) - 65) for n in range(self._length))
    
    def __setitem__(self, entry: int, value: bool) -> None:
        """Insert a new value into the entry history buffer.

        Args:
            entry (int): The entry to insert the value into.
            value (bool): The value to insert.
        """
        self.updates[entry] += 1
        for idx in range(self._length):
            data: uint8 = self._data[entry][idx]
            carry: bool = bool(data & 0x1)
            hold: bool = bool(data & 0x2)
            hold_valid: bool = bool(data & 0x4)
            store: uint64 = self.buffers[entry][idx]
            evicted_bit: uint8 = uint8(store & 0x1)
            self.buffers[entry][idx] = (store >> 1) | (uint64(value) << 63)
            if hold_valid:
                state: uint8 = hold + evicted_bit + carry
                carry = bool(state & 1)
                value = bool(state >= 2)
                self._data[entry][idx] = uint8(carry)
            else:
                self._data[entry][idx] = uint8(4 + evicted_bit * 2 + carry)
                break

# Alias
bhb = binary_history_buffer
