from logging import DEBUG, Logger, NullHandler, getLogger
from random import randint, seed
from typing import LiteralString, cast
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import pytest
from numpy import NAN, array, float64, int64, isclose, isnan, uint64, zeros
from numpy.random import Generator, default_rng, normal

from binary_history_buffer import bhb

_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


# (limit, pattern, length)
TEST_LEN_PATTERNS: tuple[tuple[int, str, int], ...] = (
    (0, '0' * 5, 5),
    (0, '0', 1),
    (0, '', 0),
    (0, '1', 1),
    (0, '1' * 5, 5),
    (0, '100000', 6),
    (0, '100001', 6),
    (0, '000001', 6),
    (10, '0' * 5, 5),
    (10, '0', 1),
    (10, '', 0),
    (10, '1', 1),
    (10, '1' * 5, 5),
    (10, '100000', 6),
    (10, '100001', 6),
    (10, '000001', 6),
    (3, '0' * 5, 3),
    (3, '0', 1),
    (3, '', 0),
    (3, '1', 1),
    (3, '1' * 5, 3),
    (3, '100000', 3),
    (3, '100001', 3),
    (3, '000001', 3),
    (1, '0' * 5, 1),
    (1, '0', 1),
    (1, '', 0),
    (1, '1', 1),
    (1, '1' * 5, 1),
    (1, '100000', 1),
    (1, '100001', 1),
    (1, '000001', 1),
    (0, '0' * 1017, 1017),
    (0, '1' * 3578, 3578),
    (4096, '0' * 1017, 1017),
    (4096, '1' * 3578, 3578),
    (1000, '0' * 1017, 1000),
    (1000, '1' * 3578, 1000)
)
# (limit, pattern, index, exception)
TEST_GET_INDEX_PATTERNS: tuple[tuple[int, str, int, bool], ...] = (
    (0, '0' * 5, 3, False),
    (0, '0', 1, True),
    (0, '', 0, True),
    (0, '1', 1, True),
    (0, '1' * 5, 5, True),
    (0, '100000', 5, False),
    (0, '100000', 0, False),
    (0, '000001', 3, False),
    (10, '0' * 5, 3, False),
    (10, '0', 0, False),
    (10, '', 0, True),
    (10, '1', 1, True),
    (10, '1' * 5, 5, True),
    (10, '100000', 5, False),
    (10, '100000', 0, False),
    (10, '000001', 3, False),
    (3, '0' * 5, 3, True),
    (3, '0', 0, False),
    (3, '', 0, True),
    (3, '1', 1, True),
    (3, '1' * 5, 5, True),
    (3, '100000', 5, True),
    (3, '100000', 0, False),
    (3, '000001', 3, True),
    (1, '0' * 5, 3, True),
    (1, '0', 0, False),
    (1, '', 0, True),
    (1, '1', 1, True),
    (1, '1' * 5, 5, True),
    (1, '100000', 5, True),
    (1, '100000', 0, False),
    (1, '000001', 3, True),
    (0, '0' * 1017, 387, False),
    (0, '1' * 3578, 2345, False),
    (4096, '0' * 1017, 1016, False),
    (4096, '1' * 3578, 3577, False),
    (1000, '0' * 1017, 1016, True),
    (1000, '1' * 3578, 3577, True)
)
# (limit, pattern, start, stop, exception)
TEST_GET_SLICE_PATTERNS: tuple[tuple[int, str, int, int, bool], ...] = (
    (0, '0' * 5, 0, 3, False),
    (0, '0', 0, 1, False),
    (0, '', 0, 0, False),
    (0, '1', 3, 3, False),
    (0, '1' * 5, 0, 6, False),
    (0, '100000', 3, 5, False),
    (0, '100000', 0, 0, False),
    (0, '000001', 0, 6, False),
    (10, '0' * 5, -1, 3, False),
    (10, '0', 1, 0, False),
    (10, '', 0, 0, False),
    (10, '1', 0, 1, False),
    (10, '1' * 5, -8975, 5, False),
    (10, '100000', 5, 9867, False),
    (4096, '0' * 1017, 0, 1016, False),
    (4096, '1' * 3578, 3577, 4100, False),
    (1000, '0' * 1017, -128, -9, False),
    (1000, '1' * 3578, -8, 3577, False)
)
# pattern, start, length, exception
TEST_AS_INT_PATTERNS: tuple[tuple[str, int, int, bool], ...] = (
    ('0' * 5, 0, -1, False),
    ('01' * 5, -1, 1, True),
    ('01' * 5, 0, 100, False),
    ('01' * 10, 0, 0, False),
    ('01' * 5, 0, 0, False)
)
TEST_AS_STR_PATTERNS: tuple[tuple[str, int, int, bool], ...] = TEST_AS_INT_PATTERNS


TEST_64_BIT_PATTERNS: tuple[str, ...] = (
    # 1st bit                                                64th bit
    '0000000000000000000000000000000000000000000000000000000000000000',
    '1111111111111111111111111111111111111111111111111111111111111111',
    '1010101010101010101010101010101010101010101010101010101010101010',
    '0101010101010101010101010101010101010101010101010101010101010101',
    '0000000000111111111100000000000010111111100000001111111111000000'
)


seed(1)  # Python random seed
rng: Generator = default_rng(1)
TEST_HF_RND_PATTERNS: list[LiteralString] = [
    ''.join([('0' * i, '1' * i)[n & 1] for n, i in enumerate((array(abs(normal(size=2048) * 4), dtype=int64) + 1))]) for _ in range(32)
]
TEST_LF_RND_PATTERNS: list[LiteralString] = [
    ''.join([('0' * i, '1' * i)[n & 1] for n, i in enumerate((array(abs(normal(size=512) * 32), dtype=int64) + 1))]) for _ in range(32)
]


@pytest.mark.parametrize('limit, pattern, length', TEST_LEN_PATTERNS)
def test_bhb_len(limit: int, pattern: str, length: int) -> None:
    """Test bhb length."""
    bhb64 = bhb(limit)
    for bit in pattern:
        bhb64.update(bit == '1')
    assert len(bhb64) == length


@pytest.mark.parametrize('pattern, start, length, exception', TEST_AS_INT_PATTERNS)
def test_bhb_as_int(pattern: str, start: int, length: int, exception: bool) -> None:
    """Test bhb length."""
    bhb64 = bhb(15)
    for bit in pattern:
        bhb64.update(bit == '1')
    if exception:
        with pytest.raises(ValueError):
            bhb64.as_int(start, length)
    else:
        if not length:
            result = int('1' + pattern[::-1][:15][start:][::-1], 2)
        else:
            result = int('1' + pattern[::-1][:15][start:max(start + length, start)][::-1], 2)
        assert bhb64.as_int(start, length) == result


@pytest.mark.parametrize('pattern, start, length, exception', TEST_AS_INT_PATTERNS)
def test_bhb_as_str(pattern: str, start: int, length: int, exception: bool) -> None:
    """Test bhb length."""
    bhb64 = bhb(15)
    for bit in pattern:
        bhb64.update(bit == '1')
    if exception:
        with pytest.raises(ValueError):
            bhb64.as_str(start, length)
    else:
        if not length:
            result: str = pattern[::-1][:15][start:][::-1]
        else:
            result = pattern[::-1][:15][start:max(start + length, start)][::-1]
        assert bhb64.as_str(start, length) == result


@pytest.mark.parametrize('limit, pattern, index, exception', TEST_GET_INDEX_PATTERNS)
def test_bhb_getitem_index(limit: int, pattern: str, index: int, exception: bool) -> None:
    """Test bhb__getitem(index)."""
    bhb64 = bhb(limit)
    for bit in pattern:
        bhb64.update(bit == '1')
    if limit:
        # NOTE: Bit numbering is right to left, string indexing is left to right
        pattern = pattern[::-1][:limit][::-1]
    if exception:
        with pytest.raises(IndexError):
            bhb64[index]
    else:
        assert bhb64[index] == (pattern[::-1][index] == '1')


@pytest.mark.parametrize('limit, pattern, start, stop, exception', TEST_GET_SLICE_PATTERNS)
def test_bhb_getitem_slice(limit: int, pattern: str, start: int, stop: int, exception: bool) -> None:
    """Test bhb__getitem(slice)."""
    bhb64 = bhb(limit)
    for bit in pattern:
        bhb64.update(bit == '1')
    if limit:
        # NOTE: Bit numbering is right to left, string indexing is left to right
        pattern = pattern[::-1][:limit][::-1]
    if exception:
        with pytest.raises(ValueError):
            bhb64[start:stop]
    else:
        assert cast(bhb, bhb64[start:stop]).as_str() == pattern[::-1][start:stop][::-1]


@pytest.mark.parametrize('pattern', TEST_HF_RND_PATTERNS)
def test_bhb_rnd_patterns(pattern) -> None:
    """Test random patterns."""
    bhb64 = bhb()
    for bit in pattern:
        bhb64.update(bit == '1')
    hits, updates, ratio = bhb64.totals()

    # Check total stats
    assert hits == pattern.count('1')
    assert updates == len(pattern)
    assert isclose(ratio, pattern.count('1') / len(pattern))

    # Check history stats.
    start: int = randint(0, len(pattern))
    pattern = pattern[::-1][start:start + randint(0, len(pattern) - start)]
    length: int = len(pattern)
    hits: uint64 = uint64(pattern.count('1'))
    updates: uint64 = uint64(length)
    ratio: float64 = hits / updates if updates else float64(NAN)  # Avoid the runtime warning

    if _LOG_DEBUG:
        value: int = int(pattern, 2) if pattern else 0
        _logger.debug(f'Random Test Pattern:')
        _logger.debug(f'start {start}, length {length}, # hits {hits}, # bits {updates}, ratio {ratio}')
        for nbit in range(0, len(pattern), 64):
            bit_str: str = f'{(value >> nbit) & ((1 << 64) - 1):064b}'
            _logger.debug(f'Pattern #{nbit:06d} {bit_str}')

    if not length:
        with catch_warnings():
            simplefilter('ignore', RuntimeWarning)
            hits, updates, ratio = bhb64.history(start, -1).totals()
        assert hits == 0
        assert updates == 0
        assert isnan(ratio)
        with catch_warnings():
            simplefilter('ignore', RuntimeWarning)
            hits, updates, ratio = bhb64.history_totals(start, -1)
        assert hits == 0
        assert updates == 0
        assert isnan(ratio)
    else:
        assert bhb64.history(start, length).totals() == bhb64.history_totals(start, length)
        assert bhb64.history_totals(start, length) == (hits, updates, ratio)


def test_plot():
    ratio_histories = zeros((2, 8, len(TEST_HF_RND_PATTERNS[0])), dtype=float64)
    bhb64 = bhb()
    bhbl264 = bhbz(length=8)
    for n, bit in enumerate(TEST_HF_RND_PATTERNS[0]):
        bhb64.update(bit == '1')
        bhbl264.update(bit == '1')
        for i, (l2, bh) in enumerate(zip(zip(*bhbl264.histories()), (bhb64.history(2**n - 64) for n in range(7, 15)))):
            ratio_histories[0, i, n] = l2[2]
            ratio_histories[1, i, n] = bh[2]

    for i, c in enumerate(('red', 'green', 'blue', 'black', 'yellow', 'cyan', 'magenta', 'orange')):
        #plt.plot(ratio_histories[0, i], linestyle='dashed', color=c)
        plt.plot(ratio_histories[0, i,:256] - ratio_histories[1, i, :256], linestyle='solid', color=c, linewidth=1)

    plt.show()