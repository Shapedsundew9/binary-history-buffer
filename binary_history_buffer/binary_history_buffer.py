"""Binary History Buffer."""

from logging import DEBUG, Logger, NullHandler, getLogger
from numpy import uint64, float64, minimum


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


class binary_history_buffer():
    """Maintains a compressed history of a binary state."""

    def __init__(self, limit: int = 0) -> None:
        """Create a binary history buffer.
        
        Args
        ----

        limit: The maximum number of bits to store in the buffer. 0 = infinite
        """
        self.limit: uint64 = uint64(limit)
        self.buffer: int = 0
        self.updates: uint64 = uint64(0)
        self.hits: uint64 = uint64(0)

    def totals(self) -> tuple[uint64, uint64, float64]:
        """Get the total number of hits, updates & the ratio for the entry.

        Returns:
            (Total True updates, Total updates, Hit Ratio)
        """
        return self.hits, self.updates, self.hits / self.updates
    
    def history(self, length: int, start: int = 0) -> tuple[uint64, uint64, float64]:
        """Get the fraction of True updates for each history period.
        
        Args:
            length: The length of bit history to evaluate.
            start: The starting bit position for the length. Defaults to 0.
        Returns:
            (# True updates, # Updates, Ratio)
        """
        limit: uint64 = minimum(self.updates, self.limit) if self.limit else self.updates
        if (start + length) > limit:
            _logger.debug('Reducing history length to fit buffer')
            length = int(self.updates - start)
        history: int = ((1 << length) - 1) & (self.buffer >> start)
        hits: uint64 = uint64(history.bit_count())
        bits: uint64 = uint64(length)
        if _LOG_DEBUG:
            _logger.debug(f'History start {start}, length {length}, # hits {hits}, # bits {bits}, ratio {hits / bits}')
            for nbit in range(0, length, 64):
                bit_str: str = f'{(history >> nbit) & ((1 << 64) - 1):064b}'
                _logger.debug(f'History #{nbit:06d} {bit_str}')
        return hits, bits, hits / bits

    def update(self, value: bool) -> None:
        """Insert a new value into the entry history buffer.

        Args:
            value (bool): The value to insert.
        """
        self.buffer = (self.buffer << 1) | int(value)
        if self.limit > 0:
            self.buffer &= (1 << int(self.limit)) - 1
        self.updates += 1
        self.hits += int(value)


# Aliases
bhb = binary_history_buffer
