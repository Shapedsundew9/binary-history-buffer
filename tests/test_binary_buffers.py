import pytest
from typing import LiteralString
from pprint import pformat
from logging import DEBUG, Logger, NullHandler, getLogger
from binary_history_buffer import bhb, bhbz, bhbzt
from numpy.typing import NDArray
from numpy import uint64, float64, array, int64, isclose, zeros
from numpy.random import default_rng, Generator, normal
import matplotlib.pyplot as plt


_logger: Logger = getLogger(__name__)
_logger.addHandler(NullHandler())
_LOG_DEBUG: bool = _logger.isEnabledFor(DEBUG)


TEST_64_BIT_PATTERNS: tuple[str, ...] = (
    # 1st bit                                                64th bit
    '0000000000000000000000000000000000000000000000000000000000000000',
    '1111111111111111111111111111111111111111111111111111111111111111',
    '1010101010101010101010101010101010101010101010101010101010101010',
    '0101010101010101010101010101010101010101010101010101010101010101',
    '0000000000111111111100000000000010111111100000001111111111000000'
)


rng: Generator = default_rng(1)
TEST_HF_RND_PATTERNS: list[LiteralString] = [
    ''.join([('0' * i, '1' * i)[n & 1] for n, i in enumerate((array(abs(normal(size=2048) * 4), dtype=int64) + 1))]) for _ in range(32)
]
TEST_LF_RND_PATTERNS: list[LiteralString] = [
    ''.join([('0' * i, '1' * i)[n & 1] for n, i in enumerate((array(abs(normal(size=512) * 32), dtype=int64) + 1))]) for _ in range(32)
]


@pytest.mark.parametrize('pattern', TEST_64_BIT_PATTERNS)
def test_64_bit_patterns(pattern) -> None:
    """Test 64 bit patterns."""
    bhb64 = bhb()
    bhbl264 = bhbz()
    bhbl264t = bhbzt()
    for bit in pattern:
        bhb64.update(bit == '1')
        bhbl264.update(bit == '1')
        bhbl264t[0]= bit == '1'
    assert bhb64.buffer == int(pattern, 2)
    assert bhbl264._bhbl2t.buffer[0][0] == int(pattern, 2)
    assert bhbl264t.buffer[0][0] == int(pattern, 2)
    assert bhb64.totals() == bhbl264.totals() == bhbl264t[0].totals()
    bhbl264_history: tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]] = bhbl264.histories()
    bhbl264t_history: tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]] = bhbl264t[0].histories()
    assert all(all(a == b) for a, b in zip(bhbl264_history, bhbl264t_history))
    assert all(a == b for l2 in zip(*bhbl264_history) for a, b in zip(l2, bhb64.history(64))) 


@pytest.mark.parametrize('pattern', TEST_HF_RND_PATTERNS)
def test_hf_rnd_patterns_big_buffer(pattern) -> None:
    """Test random patterns."""
    if _LOG_DEBUG:
        value = int(pattern, 2)
        _logger.debug(f'Random Test Pattern:')
        for nbit in range(0, len(pattern), 64):
            bit_str: str = f'{(value >> nbit) & ((1 << 64) - 1):064b}'
            _logger.debug(f'Pattern #{nbit:06d} {bit_str}')
    bhb64 = bhb()
    bhbl264 = bhbz(length=8)
    bhbl264t = bhbzt(length=8)
    for bit in pattern:
        bhb64.update(bit == '1')
        bhbl264.update(bit == '1')
        bhbl264t[0]= bit == '1'
    assert bhb64.buffer == int(pattern, 2)
    if _LOG_DEBUG:
        _logger.debug(f'BHB totals   : {bhb64.totals()}')
        _logger.debug(f'BHBL2 totals : {bhbl264.totals()}')
        _logger.debug(f'BHBL2T totals: {bhbl264t[0].totals()}')
    assert bhb64.totals() == bhbl264.totals() == bhbl264t[0].totals()
    if _LOG_DEBUG:
        _logger.debug(f'BHB history:\n{pformat([bhb64.history(2**x - 64) for x in range(7, 14)])}')
        _logger.debug(f'BHBL2 history:\n{pformat(bhbl264.histories())}')
        _logger.debug(f'BHBL2T history:\n{pformat(bhbl264t[0].histories())}')
    bhbl264_history: tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]] = bhbl264.histories()
    bhbl264t_history: tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]] = bhbl264t[0].histories()
    assert all(all(a == b) for a, b in zip(bhbl264_history, bhbl264t_history))
    for l2, bh in zip(zip(*bhbl264_history), (bhb64.history(2**n - 64) for n in range(7, 14))):
        assert isclose(l2[2], bh[2], atol = 2/64, rtol = 0.0)


@pytest.mark.parametrize('pattern', TEST_HF_RND_PATTERNS)
def test_hf_rnd_patterns_small_buffer(pattern) -> None:
    """Test random patterns."""
    if _LOG_DEBUG:
        value = int(pattern, 2)
        _logger.debug(f'Random Test Pattern:')
        for nbit in range(0, len(pattern), 64):
            bit_str: str = f'{(value >> nbit) & ((1 << 64) - 1):064b}'
            _logger.debug(f'Pattern #{nbit:06d} {bit_str}')
    bhb64 = bhb()
    bhbl264 = bhbz(length=4)
    bhbl264t = bhbzt(length=4)
    for bit in pattern:
        bhb64.update(bit == '1')
        bhbl264.update(bit == '1')
        bhbl264t[0]= bit == '1'
    assert bhb64.buffer == int(pattern, 2)
    if _LOG_DEBUG:
        _logger.debug(f'BHB totals   : {bhb64.totals()}')
        _logger.debug(f'BHBL2 totals : {bhbl264.totals()}')
        _logger.debug(f'BHBL2T totals: {bhbl264t[0].totals()}')
    assert bhb64.totals() == bhbl264.totals() == bhbl264t[0].totals()
    if _LOG_DEBUG:
        _logger.debug(f'BHB history:\n{pformat([bhb64.history(2**x - 64) for x in range(7, 10)])}')
        _logger.debug(f'BHBL2 history:\n{pformat(bhbl264.histories())}')
        _logger.debug(f'BHBL2T history:\n{pformat(bhbl264t[0].histories())}')
    bhbl264_history: tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]] = bhbl264.histories()
    bhbl264t_history: tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]] = bhbl264t[0].histories()
    assert all(all(a == b) for a, b in zip(bhbl264_history, bhbl264t_history))
    for l2, bh in zip(zip(*bhbl264_history), (bhb64.history(2**n - 64) for n in range(7, 10))):
        assert isclose(l2[2], bh[2], atol = 2/64, rtol = 0.0)


@pytest.mark.parametrize('pattern', TEST_LF_RND_PATTERNS)
def test_lf_rnd_patterns_big_buffer(pattern) -> None:
    """Test random patterns."""
    if _LOG_DEBUG:
        value = int(pattern, 2)
        _logger.debug(f'Random Test Pattern:')
        for nbit in range(0, len(pattern), 64):
            bit_str: str = f'{(value >> nbit) & ((1 << 64) - 1):064b}'
            _logger.debug(f'Pattern #{nbit:06d} {bit_str}')
    bhb64 = bhb()
    bhbl264 = bhbz(length=8)
    bhbl264t = bhbzt(length=8)
    for bit in pattern:
        bhb64.update(bit == '1')
        bhbl264.update(bit == '1')
        bhbl264t[0]= bit == '1'
    assert bhb64.buffer == int(pattern, 2)
    if _LOG_DEBUG:
        _logger.debug(f'BHB totals   : {bhb64.totals()}')
        _logger.debug(f'BHBL2 totals : {bhbl264.totals()}')
        _logger.debug(f'BHBL2T totals: {bhbl264t[0].totals()}')
    assert bhb64.totals() == bhbl264.totals() == bhbl264t[0].totals()
    if _LOG_DEBUG:
        _logger.debug(f'BHB history:\n{pformat([bhb64.history(2**x - 64) for x in range(7, 14)])}')
        _logger.debug(f'BHBL2 history:\n{pformat(bhbl264.histories())}')
        _logger.debug(f'BHBL2T history:\n{pformat(bhbl264t[0].histories())}')
    bhbl264_history: tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]] = bhbl264.histories()
    bhbl264t_history: tuple[NDArray[uint64], NDArray[uint64], NDArray[float64]] = bhbl264t[0].histories()
    assert all(all(a == b) for a, b in zip(bhbl264_history, bhbl264t_history))
    for l2, bh in zip(zip(*bhbl264_history), (bhb64.history(2**n - 64) for n in range(7, 14))):
        assert isclose(l2[2], bh[2], atol = 2/64, rtol = 0.0)


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