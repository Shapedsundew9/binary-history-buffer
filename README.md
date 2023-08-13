# Binary History Buffer (BHB)

BHB record the state history of a binary varibale at reducing fidelity as the history ages. The primary use
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

## Resources

Memory ~= # stores * # buffers * 9 bytes.

e.g. An 8 store (max N = 7 so last history bit = 2<sup>7+7</sup> + 65 = 16449) 128 buffer BHB object would
use 8 * 128 * 9 = 9216 bytes + class overhead.