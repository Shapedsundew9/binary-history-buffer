# Binary History Buffers

Binary history buffers record the state of a binary variable efficiently. There are 2 types implemented trading
off memory usage for accuracy:

| Class                   | Description                                   | Memory Usage        |
|:------------------------|:----------------------------------------------|:-------------------:|
| binary_history_buffer   | Full fidelity history                         | ~O(N) + C           |
| binary_history_buffer_z | Older history is compressed reducing fidelity | ~O(log2(N/64)) + C  |

Where N is the history buffer size in bits and C is a class overhead. Each class implements the same interface:

| Method            | Description                                      | Performance |
|:------------------|:-------------------------------------------------|:-----------:|
| get()             | Return the most recent state                     | bhb == bhbz |
| update()          | Set the push in a state                          | bhb == bhbz |
| totals()          | Total hits, updates & hit ratio                  | bhb > bhbz  |
| history()         | Return the history window                        | bhb > bhbz  |
| history_totals()  | Hits, updates & hit ratio for the history window | bhbz > bhb  |

Throughout the documentation the terms 'hit', 'set', '1', 'True' and 'miss', 'clear', '0', 'False' are
equivilent in the context of the value/state of an instance of the binary history.

NOTE: The accuracy of history*() methods reduces proportionally to log2(N/64) for bhbz. 


# Binary History Buffer Log2 (BHBL2)

BHB record the state history of a binary variable at reducing fidelity as the history ages. The primary use
case is in implementing temporal effectivness tables. e.g. Given N operations to choose from record the success
or failure of each application of each operation in a binary string. To prevent the binary string consuming too
much memory store longer segements of older history at reduced fidelity. 

# Definition

Binary history is stored in lengths of 64 bits. The index of the most recent 64 bits is 0 and is stored with 1 bit
fidelity i.e. 64 bits of history takes 64 bits of storage. The next most recent 128 bits (index 1) is stored with 2 bit fidelity
i.e. 64 bits of history now only needs 32 bits of storage. The next most recent 256 bits (index 2) is stored with 4 bit fidelity
i.e. 64 bits uses 16 bits of storage, and so on.

The storage at index N then holds:

| Index | Fidelity / bits | 1<sup>st</sup> Bit # | Expression           | Last Bit # | Expression            |
|:-----:|:---------------:|:--------------------:|:--------------------:|:----------:|:---------------------:|
| 0     |       1         |   0                  | 2<sup>6</sup> - 64   |   63       | 2<sup>7</sup> - 65    |
| 1     |       2         |   64                 | 2<sup>7</sup> - 64   |   191      | 2<sup>8</sup> - 65    |
| 2     |       4         |   192                | 2<sup>8</sup> - 64   |   447      | 2<sup>9</sup> - 65    |
| 3     |       8         |   448                | 2<sup>9</sup> - 64   |   959      | 2<sup>10</sup> - 65   |
| ...                                                                                                       ||
| N     |  2<sup>N</sup>  |   -                  | 2<sup>N+6</sup>-64   |   -        | 2<sup>N+7</sup> - 65  |

Each index is half the fidelity of the previous meaning that 2 bits from index N become 1 bit in index N+1. To minimize
error a carry bit is maintained. When index N is updated twice (2 bits inserted at the front push 2 bits out the back)
if 2 or 3 bits in the 2 evicited bits and the carry bit are set then a 1 is inserted in index N+1 else a 0 is inserted.
If 1 or 3 bits in the evicted bits and carry bit are set then the carry bit is set.

| Last 2 bits of N | Carry bit N | Update\* | 1st bit of N+1 | Carry bit N |
|:----------------:|:-----------:|:--------:|:--------------:|:-----------:|
| 00               |  0          |    -->   |   0            |  0          |
| 01               |  0          |    -->   |   0            |  1          |
| 10               |  0          |    -->   |   0            |  1          |
| 11               |  0          |    -->   |   1            |  0          |
| 00               |  1          |    -->   |   0            |  1          |
| 01               |  1          |    -->   |   1            |  0          |
| 10               |  1          |    -->   |   1            |  0          |
| 11               |  1          |    -->   |   1            |  1          |

\*An update in the table above is 2 bits inserted into the front of N pushing the last 2 bits out.

## Examples

| Last 8 bits in N | Update\* | First 4 bits in N+1 |
|:----------------:|:--------:|:-------------------:|
| 10101010         | -->      | 1010                |
| 00000111         | -->      | 0001                |
| 10000111         | -->      | 1001                |
| 00000001         | -->      | 0000                |
| 10000000         | -->      | 0000                |
| 10000001         | -->      | 1000                |
| 01111111         | -->      | 0111                |
| 01011111         | -->      | 1011                |

\*An update in the table above is 8 bits inserted into the front of N pushing the last 8 bits out. Initial
value of Carry Bit N is 0.

**NOTE**: Whilst overall the total counts of 1s and 0s are maintained the 'future bias' of the carry bit means
1s tend to move to the left.

## Implementation: The hold bit

In the examples above updates come in multiples of 2 which keeps the explaination of what is happening simple. In reality
updates come one at a time and the implementation must account for the fact only 1 bit has been pushed out of store index N
which is not enough to update store index N+1. In this case the evicted bit from store index N is preserved in memory
to be combined with the next evicted bit (and the carry bit) to determine the update for store index N+1.

The current implementation maintains these store extra bits or state in a structure called _data_. _data_ has the
shape (number of buffers, number of stores) and is of type numpy.uint8. Each data element has the structure:

| Bits 7-3 | Bit 2      | Bit 1 | Bit 0 |
|:--------:|:----------:|:-----:|:-----:|
| Reserved | Hold Valid | Hold  | Carry |

Reserved = Undefined & not guaranteed to be preserved.
Hold valid = If True indicates that the bit in Hold is valid. 
Hold: The 1st bit in the evicted pair needed to update the next store. If Hold Valid is True else 0.
Carry: As defined above.

When calculating the ratio of hits (set bits) in a history length or store the hold & carry bits are considered. i.e. the 
maximum number of bits in a store may be 66 if the hold bit is valid and the carry is set. Thus the error in a window may
be up to 1 in 32 ~ 3% in event history th

## Resources

Memory ~= # stores * # buffers * 10 bytes.

e.g. An 8 store (max N = 7 so last history bit = 2<sup>7+7</sup> + 65 = 16449) 128 buffer BHB object would
use 8 * 128 * 9 = 9216 bytes + class overhead.