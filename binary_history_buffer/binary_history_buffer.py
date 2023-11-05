"""Binary History Buffer."""

from typing import Self
from logging import DEBUG, Logger, NullHandler, getLogger
from numpy import uint64, float64, minimum, integer


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


class binary_history_buffer:
    """Maintains a compressed history of a binary state."""

    def __init__(self, limit: int | integer = 0, _initial_state: int = 1) -> None:
        """Create a binary history buffer.

        Args
        ----

        limit: The maximum number of bits to store in the buffer. 0 = infinite.
        _initial_state: The initial state of the buffer. This is for internal use only. Leave as default.
        """
        self.limit: uint64 = uint64(limit)
        self.buffer: int = int(_initial_state)
        self.updates: uint64 = uint64(_initial_state.bit_length() - 1)
        self.hits: uint64 = uint64(_initial_state.bit_count() - 1)

    def __len__(self) -> int:
        """Get the length of the history. The maximum length is the limit of the buffer.
        To get the total number of updates read the updates property.
        """
        return (
            int(self.updates)
            if not self.limit
            else int(minimum(self.updates, self.limit))
        )

    def __repr__(self) -> str:
        """Get the representation of the history."""
        return f"{self.__class__}(hits: {self.hits}, updates: {self.updates}, limit: {self.limit})"

    def __getitem__(self, index: int | integer | slice) -> bool | Self:
        """Get the state of a bit in the history."""
        limit: int = int(
            minimum(self.updates, self.limit) if self.limit else self.updates
        )
        if isinstance(index, slice):
            if not (index.step is None or index.step == 1):
                raise ValueError(f"Slice step must be 1 or None not {index.step}")
            start: int = int(index.start if index.start is not None else 0)
            stop: int = int(index.stop if index.stop is not None else limit)
            if start < 0:
                start = max(0, limit + start)
            elif start > limit:  # 0 length slice
                return self.__class__(self.limit)
            if stop < 0:
                stop = limit + stop
            elif stop > limit:
                stop = limit
            length: int = stop - start
            if length <= 0:  # 0 length slice
                return self.__class__(self.limit)
            return self.history(start, length)
        if index >= limit or index < -limit:
            raise IndexError(
                f"Index {index} is out of range for history of length {limit}"
            )
        if index < 0:
            index = int(limit - index)
        return self._get_bit(index)

    def _get_bit(self, index: int | integer) -> bool:
        """Get the state of a bit in the history. Index must be in the range [0:limit]."""
        return bool((self.buffer >> index) & 1)

    def totals(self) -> tuple[uint64, uint64, float64]:
        """Get the total number of hits, updates & the ratio for the entry.

        Returns:
            (Total True updates, Total updates, Hit Ratio)
        """
        return self.hits, self.updates, self.hits / self.updates

    def as_int(self, start: int | integer = 0, length: int | integer = 0) -> int:
        """Return the history[start:length] as an integer.

        The length of the history returned is the minimum of the requested length, the length of the
        history in the buffer and the limit of the buffer (all minus the starting position).

        NOTE: The MSb of the integer returned is a marker indicating the next bit is the start (oldest
        bit) of the history.

        Args
        ----
        length: The length of bit history to evaluate. Defaults to 0 (all).
        start: The starting bit position for the length. Defaults to 0.

        Returns
        -------
        The history as an integer.
        """
        if length < 0:  # 0 length history
            return 1  # Just the marker
        if start < 0:
            raise ValueError(f"Start {start} must be >= 0")

        limit: int = int(
            minimum(self.updates, self.limit) if self.limit else self.updates
        )
        if not length:
            length = limit
        if (start + length) > limit:
            _logger.debug("Reducing history length to fit buffer")
            length = int(limit - start)
        marker: int = 1 << int(length)
        return int((marker - 1) & (self.buffer >> start) | marker)

    def as_str(self, start: int | integer = 0, length: int | integer = 0) -> str:
        """Return the history[start:length] as a string.

        The length of the history returned is the minimum of the requested length, the length of the
        history in the buffer and the limit of the buffer (all minus the starting position).

        Args
        ----
        length: The length of bit history to evaluate. Defaults to 0 (all).
        start: The starting bit position for the length. Defaults to 0.

        Returns
        -------
        The history as a string of 1's and 0's oldest to newest. e.g. '000010101'
        where the left hand 0 is the oldest state recorded and the righand 1 is the most recent.
        """
        if length < 0:  # 0 length history
            return ""
        history: int = self.as_int(start, length)
        limit: int = int(
            minimum(self.updates, self.limit) if self.limit else self.updates
        )
        if (start + length) > limit:
            _logger.debug("Reducing history length to fit buffer")
            length = int(limit - start)
        print(f"History: {history:b}, Start: {start}, Length: {length}")
        return f"{{:0{length}b}}".format(history)[1:]

    def history(self, start: int | integer = 0, length: int | integer = 0) -> Self:
        """Return a copy of the history[start:length].

        The length of the buffer returned is the minimum of the requested length, the length of the
        history in the buffer and the limit of the buffer (all minus the starting position).

        NOTE: The defaults of start=0 and length=0 will return the entire history effectively
        creating a copy of the buffer.

        Args
        ----
        length: The length of bit history to evaluate. Defaults to 0 (all).
        start: The starting bit position for the length. Defaults to 0.

        Returns
        -------
        A new binary_history_buffer with the requested history.
        """
        return binary_history_buffer(self.limit, self.as_int(start, length))

    def history_totals(
        self, start: int | integer, length: int | integer = 0
    ) -> tuple[uint64, uint64, float64]:
        """Get the total number of hits, updates & the ratio in the history window [start:length].

        The totals returned are for the history up to the minimum of the requested length, the length of the
        history in the buffer and the limit of the buffer (all minus the starting position).

        Args
        ----
        length: The length of bit history to evaluate. Defaults to 0 (all).
        start: The starting bit position for the length. Defaults to 0.

        Returns
        -------
        (# True updates, # Updates, Ratio) for the history window.
        """
        history: int = self.as_int(start, length)
        hits: uint64 = uint64(history.bit_count() - 1)
        bits: uint64 = uint64(length) if length >= 0 else uint64(0)
        if _LOG_DEBUG:
            _logger.debug(
                f"History start {start}, length {length}, # hits {hits}, # bits {bits}, ratio {hits / bits}"
            )
            for nbit in range(0, length, 64):
                bit_str: str = f"{(history >> nbit) & ((1 << 64) - 1):064b}"
                _logger.debug(f"History #{nbit:06d} {bit_str}")
        return hits, bits, hits / bits

    def update(self, value: bool) -> None:
        """Insert a new value into the entry history buffer.

        Args:
            value (bool): The value to insert.
        """
        self.buffer = (self.buffer << 1) | int(value)
        if self.limit > 0:
            self.buffer &= (1 << int(self.limit + 1)) - 1
        self.updates += 1
        self.hits += int(value)


# Aliases
class bhb(binary_history_buffer):
    """Alias for binary_history_buffer."""

    pass
