
# [Big and Little Endian Byte Order][understand-endian]

[little endian vs big endian][little-vs-big]
[C code][endianc]


## Introduction

Computer memory can be visualized as a sequence of bytes.

Endianness is the ordering of bytes in memory to represent some data.

```
              -------------------------------
    MSB -->   + b31 + b30 +  ...  + b1 + b0 +   <-- LSB
              -------------------------------
```

All you need to know about memory is that it's one large array of bytes where the address is the index of the array.

We sometimes say that memory is _byte-addresseable_.
This is a way of saying that each address stores one byte.

The term _endian_ denotes which end (outermost part) of the number comes first.


## Big Endian

A big endian machine stores data _big-end_ first.

When looking at multiple bytes, the first byte (lowest address) is the biggest.

Stores the _most significant byte_ (MSB) at the lowest address.

```
    byte
    address     00    01    02    03
              -------------------------
    MSB -->   +  A  +  B  +  C  +  D  +   <-- LSB
              -------------------------
```

  Example: Sun/SPARC, IBM/RISC 6000.


## Little Endian

A little endian machine stores data _little-end_ first.

When looking at multiple bytes, the first byte (lowest address) is smallest.

Stores the _least significant byte_ (LSB) at the lowest address.

```
    byte
    address     03    02    01    00
              -------------------------
    MSB -->   +  A  +  B  +  C  +  D  +   <-- LSB
              -------------------------
```

  Example: Intel Pentium Processors.


## Registers and endianness

Endianness only makes sense when you're breaking up a _multi-byte_ quantity and are trying to store the bytes at consecutive memory locations.

If you have a 32-bit register storing a 32-bit value, it makes no sense to talk about endianness. The register is neither big-endian nor little-endian; it's just a register holding a 32-bit value. The rightmost bit is the least significant bit (lsb), and the leftmost bit is the most significant bit (msb).

```
    1001 0101

              7     6     5     4     3     2     1     0
           +-----------------------------------------------+
  msb -->  +  1  +  0  +  0  +  1  +  0  +  1  +  0  +  1  +   <-- lsb
           +-----------------------------------------------+

                      Representation of a byte
```


## Storing Numbers as Data

This gives us a common starting point which makes our lives a bit easier.

- A bit has two values (on or off, 1 or 0).

- A byte is a sequence of 8 bits.

- The leftmost bit in a byte is the biggest.
  So, the binary sequence 00001001 is the decimal number 9.

- Bits are numbered from right-to-left.

```
    0000 1001 = (2^3 + 2^0 = 8 + 1 = 9).
```

So endian-ness does not matter if you have a single byte. If you have one byte, it's the only data you read so there's only one way to interpret it (again, because computers agree on what a byte is).

Problems start when you read _multi-byte_ data, where does the biggest byte
appear?


## Mylti-byte Data

Consider a 32-bit integer (in hex):  0xabcdef12

  1 byte = 8 binary digits (bits)

  1 hex digit = 4 bits

  AB =  10 11  =  1010 1011  =  1 byte

Since each hex digit is one byte (2^4 = 16), you need eight hex digits to represent the 32-bit value.

The four bytes are: ab, cd, ef, and 12. Therefore, this integer will occupy four bytes in memory.

If we store it at a memory address starting at 1000, there are `4! = 24` different orderings possible to store these 4 bytes in 4 locations (1000 - 1003).

The two most popular are _little endian_ and _big endian_.

![](http://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Endianessmap.svg/339px-Endianessmap.svg.png)

The good news is that _usually_ we don't need to care about endianness. It's taken care of by hardware platforms and compilers. But, in some scenarios we need to care about endianness.

A common scenario is when the data needs to be _exchanged_ between different systems. In such a situation, a standard layout is specified.

**The solution:** Send 4 byte quantities using _network byte order_ which is arbitrarily picked to be one of the endian-ness (not sure if it's big or little, but it's one of them). If your machine has the same endian-ness as network byte order, then great, no change is needed. If not, then you must reverse the bytes.


## Byte Example

Consider a sequence of 4 bytes, named W X Y and Z - We avoid naming them A B C D because they are hex digits, which would be confusing. So, each byte has a value and is made up of 8 bits.

```
    Byte Name:      W          X          Y          Z
    Location:       0          1          2          3
    Value (hex):    0x12       0x34       0x56       0x78
    Value (binary): 0001 0010  0011 0100  0101 0110  0111 1000
```

For example, `W` is an entire byte, `0x12` in hex or `0001 0010` in binary. If `W` were to be interpreted as a number, it would be `18` in decimal (by the way, there's nothing saying we have to interpret it as a number - it could be an ASCII character or something else entirely).

With me so far? We have 4 bytes, W X Y and Z, each with a different value.

Now suppose we have our 4 bytes (W X Y Z) stored the same way on a big and little-endian machine. That is, memory location 0 is `W` on both machines, memory location 1 is `X`, etc.

We can create this arrangement by remembering that bytes are machine-independent. We can walk memory, one byte at a time, and set the values we need. This will work on any machine.

```c
  char *c;      // c is a pointer to a single byte

  c = 0;        // point to location 0 (won't work on a real machine!)
  *c = 0x12;    // Set W's value

  c = 1;        // point to location 1
  *c = 0x34;    // Set X's value

  c = 2;        // point to location 2
  *c = 0x56;    // Set Y's value

  c = 3;        // point to location 3
  *c = 0x78;    // Set Z's value
```

This code will work on any machine, and we have both set up with bytes W, X, Y and Z in locations 0, 1, 2 and 3.


## So, What's The Problem?

Problems happen when computers try to read multiple bytes. Some data types contain multiple bytes, like long integers or floating-point numbers. A single byte has only 256 values, so it can store 0 - 255.

When you read multi-byte data, where does the biggest byte appear?

Again, endian-ness does not matter if you have a single byte.

Now suppose we have our 4 bytes (W X Y Z) stored the same way on a big-and little-endian machine. That is, memory location 0 is W on both machines, memory location 1 is X, etc.


## Interpreting Data

Now let's do an example with multi-byte data (finally!).

A `short int` is a 2-byte (16-bit) number, which can range from 0 - 65535 (if unsigned). Let's use it in an example.

```c
  short *s;  // pointer to a short int (2 bytes)
  s = 0;     // point to location 0; *s is the value
```

So, `s` is a pointer to a `short` and is now looking at byte location 0 (which has `W`). What happens when we read the value at `s`?

### Big endian machine


```
    W    X      Y    Z
   0x12 0x34 | 0x56 0x78
    00   01     02   03


    W  X    Y   Z
 | 12  34 | 56  78 |        |        |
   00  01   02  03   04  05   06  07

 |        |        |        |        |
   08  09   0A  0B   0C  0D   0E  0F
```

A short is two bytes, so we read them off:  location `s` is address 0 (`W` or `0x12`) and location `s+1` is address 1 (`X` or `0x34`).

Since the first byte is biggest, the number must be `256 * byte 0 + byte 1` or `256 * W + X` or `0x1234`. I multiplied the first byte by 256 (2^8) because I needed to shift it over 8 bits.

### Little endian machine

```
     X    W      Z    Y
    0x34 0x12 | 0x78 0x56
     00   01     02   03


                    Y   Z    W   X
 |                | 56  78 | 12  34  |
   07  06  05  04   03  02   01  00

 |                |                  |
   0F  0E  0D  0C   0B  0A   09  08
```

I agree a short is 2 bytes, and I'll read them off just like him: location `s` contains 0x12, and location `s+1` contains 0x34. But in my world, the first byte is the littlest! The value of the short is byte 0 + 256 * byte 1, or 256*X + W, or `0x3412`.

Keep in mind that both machines start from location `s` and read memory going upwards. There is no confusion about what location 0 and location 1 mean and there is no confusion that a short is 2 bytes.

But do you see the problem? The big-endian machine thinks `s = 0x1234` and the little-endian machine thinks `s = 0x3412`. The same exact data gives two different numbers. Probably not a good thing.



## Yet another example

Let's do another example with a 4-byte integer.

```c
  int *i;   // pointer to an int (4 bytes on 32-bit machine)
  i = 0;    // points to location zero, so *i is the value there
```

Again we ask: what is the value at `i`?

### Big endian machine

An `int` is 4 bytes, and the first is the largest. I read 4 bytes (W X Y Z) and `W` is the largest. The number is `0x12345678`.

```
     W     X   |  Y     Z
    0x12  0x34   0x56  0x78
```

### Little endian machine

Sure, an `int` is 4 bytes, but the first is smallest. I also read W X Y Z, but `W` belongs way in the back — it's the littlest. The number is `0x78563412`.

```
     Z     Y   |  X     W
    0x78  0x56   0x34  0x12
```

```
  Big and Little Endian Byte Order

  R1   W = 0x12                 18  set data
  R2   X = 0x34                 52
  R3   Y = 0x56                 86
  R4   Z = 0x78                 120

  R5   // data is W, X, Y, Z

  R6   big_short = (W <<8) + X         4,660
  R7   little_short = W + (X << 8)     13,330

  R8   big_int = (W << 24) + (X << 16) + (Y << 8) + Z      305,419,896
  R9   little_int = W + (X << 8) + (Y << 16) + (Z << 24)   2,018,915,346
  R10
  R11  //now, in hex
  R12  hex(big_short)                  0x1234
  R13  hex(little_short)               0x3412
  R14  hex(big_int)                    0x12345678
  R15  hex(little_int)                 0x78563412
```


## The NUXI problem

Issues with byte order are sometimes called _"the NUXI problem"_: `UNIX` stored on a big-endian machine can show up as `NUXI` on a little-endian one.

Suppose we want to store 4 bytes (U, N, I and X) as two shorts: `UN` and `IX`. Each letter is an entire byte. To store the two shorts we would write:

```
  Decimal  Hex   Binary     Symbol
    85     0x55  0101 0101    U
    78     0x4E  0100 1110    N
    73     0x49  0100 1001    I
    88     0x58  0101 1000    X
```

```c
  short *s;   // pointer to set shorts

  short s0;  // 0x8578
  short s2;  // 0x7388

  s = 0;      // point to location 0 (won't work on a real machine!)
  *s = UN;    // store first short: U * 256 + N  where 2^8 = 256

  s = 2;      // point to next location
  *s = IX;    // store second short: I * 256 + X
```

This code is not specific to a machine. If we store `UN` on a machine and ask to read it back, it had better be `UN`! I don't care about endian issues, if we store a value on one machine, we need to get the same value back.

However, if we look at memory one byte at a time (using our `char *` trick), the order could vary.

On a big endian machine, we see

```
    Byte:      U   N   |  I    X
    Location: 0x0 0x01   0x02 0x03

     U    N   |  I    X
    0x00 0x01   0x02 0x03
```

Which make sense. U is the biggest byte in "UN" and is stored first. The same goes for "IX": I is the biggest, and stored first.

On a little-endian machine, we would see

```
    Byte:       N    U   |  X    I
    Location:  0x00 0x01   0x02 0x03

     I    X   |  U    N
    0x03 0x02   0x01 0x00
```

And this makes sense also. `N` is the littlest byte in `UN` and is stored  first. Again, even though the bytes are stored `backwards` in memory, the little-endian machine knows it is little endian, and interprets them correctly when reading the values back.

Also, note that we can specify hex numbers such as `x = 0x1234` on any machine. Even a little-endian machine knows what you mean when you write `0x1234`, and won't force you to swap the values yourself (you specify the hex number to write, and it figures out the details and swaps the bytes in memory, under the covers. Tricky).

This scenario is called the "NUXI" problem because the byte sequence UNIX is interpreted as NUXI on the other type of machine. Again, this is only a problem if you exchange data -- each machine is internally consistent.


## Exchanging Data Between Endian Machines

Computers are connected - gone are the days when a machine only had to worry about reading its own data. Big and little-endian machines need to talk and get along. How do they do this?

### Solution 1: Use a Common Format

The easiest approach is to agree to a common format for sending data over the network. The standard network order is actually big-endian, but some people get uppity that little-endian didn't win... we'll just call it "network order".

To convert data to network order, machines call a function `hton` (host-to-network). On a big-endian machine this won't actually do anything, but we won't talk about that here (the little-endians might get mad).

But it is important to use hton before sending data, even if you are big-endian. Your program may be so popular it is compiled on different machines, and you want your code to be portable (don't you?).

Similarly, there is a function ntoh (network to host) used to read data off the network. You need this to make sure you are correctly interpreting the network data into the host's format. You need to know the type of data you are receiving to decode it properly, and the conversion functions are:

### Solution 2: Use a Byte Order Mark (BOM)

The other approach is to include a magic number, such as 0xFEFF, before every piece of data. If you read the magic number and it is 0xFEFF, it means the data is in the same format as your machine, and all is well.

If you read the magic number and it is 0xFFFE (it is backwards), it means the data was written in a format different from your own. You'll have to translate it.

A few points to note. First, the number isn't really magic, but programmers often use the term to describe the choice of an arbitrary number (the BOM could have been any sequence of different bytes). It's called a byte-order mark because it indicates the byte order the data was stored in.

Second, the BOM adds overhead to all data that is transmitted. Even if you are only sending 2 bytes of data, you need to include a 2-byte BOM. Ouch!

Unicode uses a BOM when storing multi-byte data (some Unicode character encodings can have 2, 3 or even 4-bytes per character). XML avoids this mess by storing data in UTF-8 by default, which stores Unicode information one byte at a time. And why is this cool?


## Importance of endianness

Suppose you are storing integer values to a file, and you send the file to a machine that uses the opposite endianness as it reads in the value. This causes problems because of endianness; you will read in reversed values that will not make sense.

Supoose you are sending numbers over the network. Again, if you send a value from a machine of one endianness to a machine of the opposite endianness, you'll have problems. This is even worse over the network because you might not be able to determine the endianness of the machine that sent you the data.

Each byte-order system has its advantages.

- Little-endian machines let you read the lowest-byte first, without reading   the others. You can check whether a number is odd or even (last bit is 0)   very easily, which is cool if you're into that kind of thing.

- Big-endian systems store data in memory the same way we humans think about   data (left-to-right), which makes low-level debugging easier.



----

[understand-endian]: http://betterexplained.com/articles/understanding-big-and-little-endian-byte-order/ "Understanding Big and Little Endian Byte Order"

[little-vs-big]: http://techforb.blogspot.com/2007/10/little-endian-vs-big-endian.html "little endian vs big endian"

[endianc]: http://www.ibm.com/developerworks/aix/library/au-endianc/index.html "C Code"

[big-and-little-endian]: http://www.cs.umd.edu/class/spring2003/cmsc311/Notes/Data/endian.html "Big and Little Endian"
