<!--
  gdb.md
  Jeff Holmes
-->


# GNU Debugger (gdb)

25 - Debugging with GDB
GNU/Linux Application Programming

[How to Use gdb](gdb.html)
Brad Vander Zanden

[Recursion Debugging Example](http://www.cs.utk.edu/~plank/plank/classes/cs140/Notes/Recursion/debug.html)
Jim Plank


## Overview

`gdb` is the GNU debugger that can be very helpful in finding problems with
your programs. It allows you to do a number of useful things including:

- Controlling the execution of your program: by placing breakpoints and single
stepping your program.

- Printing the values of variables in your program

- Determining where a _segmentation fault_ or _bus error_ occurs in your program.

In order to use gdb you need to compile your files with the `-g `option.

**NOTE:** Debugging with optimization enabled can yield odd results.
The optimizer may move code around or remove code altogether.


## A Sample GDB Session

1. Compile your files with the `-g` option.

```c
	// compile to create object file
	$ gcc -g -c student1.c

	// linking
	$ gcc -o student1 student1.o

	// compile and link in one statement
	$ gcc -g student1.c -o student1
```

2. Run `gdb`.

```bash
	$ gdb testapp
	$ gdb names_list
```


## Command Summary

Here are some of the most frequently needed GDB commands:

```
	backtrace (bt): Display the program stack.

	where: Display call stack information.

	break [file:]function

	  Set a breakpoint at function (in file).

	edit [file:]function

	  Look at the program line where it is presently stopped.

	help [name]

	  Show information about GDB command name, or general information about using GDB.

	info breakpoints: Show information about user-settable breakpoints.

	info functions: List all function names.

	info locals: Show local variables of current stack frame.

	info variables: All global and static variable names.

	info set: Show all GDB settings.

	info set variable = value

	  Assign a value to a variable.

	info source(s): Information about the current source file(s) in the program.

	list (l) [file:]function

	  List specified function or line.

	next (n) [count]

	  Execute next program line (after stopping); _step over_ any function calls
	  in the line.

	step (s) [count]

	  Execute next program line (after stopping); _step into_ any function calls
	  in the line.

	cont (c)

	  Continue running your program (after stopping, e.g. at a breakpoint).

	print (p) [expr]

	  Display the value of a variable or expression each time the program stops.

	run (r) [arglist]

	  Start your program (with arglist, if specified).

	quit (q): Exit from GDB.
```


### Running a Program

You start running a program using gdb's `run` command or `r` for short.

If you have a program that requires command line arguments, you list them
after the run command.

```
	(gdb) r
	(gdb) r 3 gradefile
```

## List

You can list the lines in your program using the `list` command, or `l` for short.

By default, it will list 10 lines of your program, centered about the next line
to be executed.

```
	l first_line_number, last_line_number

	// list lines 10-23 of the current file
	l 10,23

	// look at lines in add_name.c
	l add_name.c:6,12
```

The `names_list` program consists of the two files `list.c` and `add_name.c`.

The file that was listed defaults to `list.c` because that was the first file
we gave to gcc.


## Run

You start running a program using the `run` command or `r` for short.

```
	(gdb) r
```

If the program requires any command line arguments , you list them after the
run command.

```
	(gdb) r 3 gradefile
```

**NOTE:** If your program is _seg faulting_ then the `run` command will probably
suffice to start your debugging session. When the program seg faults, `gdb` will
tell you the file, the procedure, and the line number where the seg fault
occurred. We will explore debugging later.


## Break

Frequently, you will want to stop the program at some point during its execution
and start single-stepping it. You can cause the program to stop at a certain
line or when a certain function is called by creating breakpoints.

A _breakpoint_ is created using the `break` command or `b` for short. It takes
either a line number or a function as an argument.

### Table 25.1

```
	b 15                  // set a breakpoint at line 15 in the current file
	b add_name_to_list    // set a breakpoint on the function add_name_to_list
	b add_name.c:8        // set a breakpoint at line 8 in add_name.c
```

When you place a breakpoint on a function, `gdb` will stop the program any time
the function is called, no matter where in the program it is called.


## Delete

When a breakpoint is created gdb will assign it a number. If you want to later
delete the breakpoint you can do so by typing `delete` and the breakpoint number.

Alternatively you can type `clear` and either the line number or the function
to which the breakpoint is attached.

```
	delete 1
	clear add_name_to_list
	clear add_name.c:8
	clear 79
```

We can view the available breakpoints using the `info` command.

```
	(gdb) info breakpoints
```

### Conditional Breakpoints

Often a program will have to go through many iterations of a loop before it
reachs the point in the program where something breaks. It can be irritating to
have to keep typing continue until you finally reach the point where the error occurs.

For example, suppose that a bug appeared in `bad_names_list` at the point when
I entered "mary". I could cause the program to stop only when "mary" has been
input by entering the following breakpoint and condition:

```
	(gdb) b 18
	Breakpoint 3 at 0x10750: file list1.c, line 18.

	(gdb) condition 3 (strcmp(name, "mary") == 0)
```

The `condition` keyword says to place a conditional breakpoint on breakpoint 3
with the indicated condition. Now try running the program.

```
	(gdb) r
	The program being debugged has been started already.
	Start it from the beginning? (y or n) y

	Starting program: /home/bvz/courses/140/spring-2005/labs/gdb/bad_names_list
	enter a name: brad
	enter a name: sue
	enter a name: mary

	Breakpoint 3, main () at list1.c:18
	18              add_name_to_list(name_list, name);
```

The following example tells gdb to break at the `foo` function if the `name`
argument is equal to "mary".

```
	(gdb) b 18
	Breakpoint 3 at 0x10750: file list1.c, line 18.

	(gdb) break foo if strcmp(name, "mary") == 0
```

## Print

You can print the values of variables in gdb using the `print` command or `p`
for short and the `display` command.

```
	(gdb) r
	The program being debugged has been started already.
	Start it from the beginning? (y or n) y

	Starting program:labs/gdb/names_list
	enter a name: brad

	Breakpoint 1, main () at list.c:18
	18              add_name_to_list(name_list, name);
	(gdb)
```

Now let's print the contents of `name_list` and `name`.

```
	(gdb) p name_list
	$13 = (LIST *) 0x20af0

	(gdb) p name
	$14 = "brad\000\001\005l\000\000\000\003???,\000\000\000\004"
	(gdb)

	(gdb) display name
```

Note that `name_list` is a pointer which points to memory starting at 0x20af0
and `name` is a 20 character array where only the first five characters are
meaningful ("brad\000"). We can see that "brad" was properly read into name.

We can print the contents of the node pointed to by `name_list` by dereferencing
the pointer.

```
	(gdb) p *name_list
	$15 = {name = 0x20af8 "", next = 0x0}
	(gdb)
```

Remember that our list has a sentinel node so everything is as it should be.
The sentinel node's next field is the null pointer, thus making it an empty list.

Now type `continue`, or `c`, and enter another name.

```
	(gdb) c
	Continuing.
	enter a name: nels

	Breakpoint 1, main () at list.c:18
	18              add_name_to_list(name_list, name);
	(gdb)
```

At this point "brad" should have been added to the list but "nels" should not
yet have been added to the list. However, "nels" should be contained in name.

```
	(gdb) p name   // "nels" is in name as it should be
	$16 = "nels\000\001\005l\000\000\000\003???,\000\000\000\004"

	(gdb) p *name_list   // the sentinel node points to something
	$17 = {name = 0x20af8 "", next = 0x20b00}

	(gdb) p *name_list->next     // and that something is "brad"
	$18 = {name = 0x20b10 "brad", next = 0x0}
	(gdb)
```

Note that brad's node does not point to anything, which is correct. Also note
how I was able to "chase" a pointer by typing

```
	p *name_list->next
```

Let's try one more iteration of the program.

```
	(gdb) c
	Continuing.
	enter a name: pat

	Breakpoint 1, main () at list.c:18
	18              add_name_to_list(name_list, name);

	(gdb) p *name_list->next
	$19 = {name = 0x20b30 "nels", next = 0x20b00}  // the first node is "nels"

	(gdb) p *name_list->next->next
	$20 = {name = 0x20b10 "brad", next = 0x0}      // the second node is "brad"

	(gdb) p name    // "pat" is in name
	$21 = "pat\000\000\001\005l\000\000\000\003???,\000\000\000\004"
	(gdb)
```

Notice that I can get arbitrarily far in the list by typing a series of next's.

### Changing Data

It is also possible to change the data in an operating program. We use the
`set` command to change data.

```
	(gdb) set name = "pat"

	(gdb) set stack->stack[9] = 999
	(gdb) p *stack
```

## Repeating a Command

Here is a helpful shortcut you can use while single-stepping.

`gdb` remembers the last command that you entered and if you hit return, it will
repeat that command. This feature is handy when you want to single-step through
a number of statements.

For example, type `s` to execute the `strdup` command. Now hit return twice.
Notice that it single-steps for you.


## Single-Stepping

```
	next (n)        Execute next line, step over functions
	step (s)        Execute next line, step into functions
	cont (c)        Continue execution
```

Once you have gotten a program to stop, you will often want to single step
through the program — executing one statement at a time — rather than continuing
to the next breakpoint.

- step (s for short)

  The `step` command executes each statement and, if it encounters a function
  call, it will step into the function, thus allowing you to follow the
  flow-of-control into subroutines.

- next (n for short)

  The `next` command also executes each statement but if it encounters a
  function call it will execute the function as an _atomic statement_.
  In other words, it will execute all the statements in the function and in any
  functions that that function might call. It will seem as though you typed
  `continue` with a breakpoint set at the next statement.

The one exception to this statement is that if there is a breakpoint nested in
the function, then it will break when it reaches that breakpoint.

## Step — Execute next line, step into functions

```
(gdb) s
add_name_to_list (names=0x20af0, name_to_add=0xffbff6a0 'brad') at add_name.c:5
5           LIST *new_node = (LIST *)malloc(sizeof(struct list));
(gdb)
```

We have stepped into the function `add_name_to_list` and are about to execute
the first statement. Before executing this statement try printing `new_node`.
Since it has not been initialized, it will have a random value. When I printed
the value of new_node I got the following result:

```
	(gdb) p new_node
	$1 = (LIST *) 0x300
```

Now execute the first statement in `add_name_to_list` by typing `s` and then
print out the new value of `new_node` and the contents of the memory pointed
to by `new_node`.

```
	(gdb) s
	7           new_node->name = strdup(name_to_add);

	(gdb) p new_node
	$2 = (LIST *) 0x20b00    // your address will probably be different

	(gdb) p *new_node
	$3 = { name = 0x20b08 '', next = 0x0 }

	(gdb)
```

**NOTE:** We do not step into `malloc` because it is a system-defined function
rather than a user-defined function. The debugger is usually smart enough not
to step into system-defined functions because they have not been compiled with
the `-g` flag and hence could not be interpreted. Also note that the contents of
the memory allocated by `malloc` have reasonable values. `malloc` does no
initialization so it is purely luck that these memory locations have reasonable
values. Often they will point to garbage.

Type `s` to execute the `strdup` command. Now hit return twice. Notice that it
single-steps for you.

**NOTE:** The debugger remembers the last command that you entered and if you
hit return, it will repeat that command.

We are now at the end of the function so let us make sure that the new node was
prepended to the list and that it has the appropriate memory contents:

```
	(gdb) p *new_node    // the strdup and assignment to next worked properly
	$5 = { name = 0x20b10 'brad', next = 0x0 }

	(gdb) p *names       // print the sentinel node for the names list
	$6 = { name = 0x20af8 '', next = 0x20b00 }  // it points to our new node

	(gdb) p *names->next  // now look at the first node in the list
	$8 = { name = 0x20b10 'brad', next = 0x0 }
```

## Next — Execute next line, step over functions

Everything looks fine so let's continue by typing `c`. This  time we will single
step by using `next` rather than `step`.

```
	(gdb) c
	Continuing.
	enter a name: nels

	...

	(gdb) n
	15          for (i = 0; i < NUM_NAMES; i++) {
	(gdb)
```

Notice that we have executed `add_name_to_list` without stepping into it and
have now gone to the top of the loop.

```
	(gdb) p *name_list    // print sentinel node so we can find the first node in list
	$9 = { name = 0x20af8 '', next = 0x20b20 }

	(gdb) p *name_list->;next  // the first node should be the new node
	$10 = { name = 0x20b30 'nels', next = 0x20b00 } // it is.

	(gdb) p *name_list->next->next   // now check the second node
	$11 = { name = 0x20b10 'brad', next = 0x0 }  // it's also ok
```

Note that the `next` field for the new node points to the previous node we
prepended. If you do not believe me, check the memory address, which is `0x20b00`,
with the memory address that was given to `new_node` the first time we visited
`add_name_to_list`. They are the same.

Now, instead of typing `continue`, try typing `n` a couple more times until
`scanf` gets executed.

```
	(gdb) n
	16              printf('enter a name: ');

	(gdb) n
	17              scanf('%s', name);

	(gdb) n
	enter a name: sue

	Undefined command: 'ue'. Try help.
```

**NOTE:** Depending on the version of `gdb` that you are using you may not get
the above error message and everything might work fine. However, some versions
of `gdb` will give you the above error message.

The problem is that `gdb` will only give _THE FIRST CHARACTER_ to `scanf` and
then it will  try to interpret the rest of the character string. To circumvent
this problem, place a breakpoint after the statements which read keyboard input
and _continue_ through them.

Doing so will allow all the keyboard input to be read in before control is
returned to you. Notice that I cleverly told you to place your breakpoint at
line 18, which is the first statement after the `scanf`.

## Backtrace

You can find where you are in a program using the `backtrace` command or `bt`
for short.

The `backtrace` command will show you the current stack of functions that are
active and the parameters passed.

```
	(gdb) b add_name.c:8
	Breakpoint 2 at 0x10830: file add_name.c, line 8.

	(gdb) r
	The program being debugged has been started already.
	Start it from the beginning? (y or n) y

	Starting program: labs/gdb/names_list
	enter a name: brad
`
	Breakpoint 1, main () at list.c:18
	18              add_name_to_list(name_list, name);

	(gdb) c
	Continuing.

	Breakpoint 2, add_name_to_list (names=0x20af0, name_to_add=0xffbff6a0 "brad")
		at add_name.c:8
	8           new_node->next = names->next;
	(gdb)

	(gdb) bt
	#0  add_name_to_list (names=0x20af0, name_to_add=0xffbff6a0 "brad") at add_name.c:8
	#1  0x0001078c in main () at list.c:18
	(gdb)
```

It tells you that you are currently in `add_name_to_list` at line 8 in `add_name.c`
and that `add_name_to_list` was called from main at line 18 in `list.c`.

The `backtrace` command is typically the first command you will type when
debugging a seg fault or bus error because you will want some idea of where you
are in the program.

## Up and Down

You can use the `up` and `down` commands to move around in the stack.
These commands are necessary if you want to print a variable that is local to
another function.

For example, if I try to print main's name variable while I'm stopped at the
current breakpoint, I will get the following error message:

```
	(gdb) p name
	No symbol "name" in current context.
	(gdb)
```

However, if I type `up 1` I will be moved up to the stack record for main.
On some versions of gdb I can then print the value of name:

```
	(gdb) up 1
	#1  0x00010764 in main () at list1.c:18
	18              add_name_to_list(name_list, name);

	(gdb) p name
	$27 = "brad\000\001\005l\000\000\000\003???,\000\000\000\004"
```

Unfortunately some versions of gdb, including the one on our cetus machines,
sometimes get it wrong:

```
	(gdb) p name
	$26 = "???0\000\001\ap?4 \210????????"
```

That is irritating and should not happen.


## Debugging Programs

```
	(gdb) b 18
	Breakpoint 2 at 0x10750: file list1.c, line 18.
	(gdb) r
	Starting program: bad_names_list
	enter a name: BradleyTannerVanderZandenNelsBlakeVanderZanden

	Breakpoint 2, main () at list1.c:18
	18              add_name_to_list(name_list, name);
	(gdb) p name
	$10 = 'BradleyTannerVanderZ'
```

Hopefully the fact that the name you entered has been truncated should be a big
red flag that something is wrong with name. At this point you might check the
declaration for `name` and find that it is only 20 characters.

`scanf` does not know that `name` is only 20 characters so it reads in characters
past the memory allocated for `name`.

From the diagrams of memory that we have drawn in class you should have an
inkling that the memory for the integer variable `i`, which precedes `name` in
the declaration list, has been clobbered.

Sure enough, if you try printing the value of `i` you will find that it is not 0
as you would expect.

```
	(gdb) p i
	$11 = 1933732961
```

However, suppose the truncation problem does not help you. You might then try
typing `n` a couple times.

```
	(gdb) n
	main () at list1.c:15
	15          for (i = 0; i &lt; NUM_NAMES; i++) {

	(gdb) n
	21          for (name_list_ptr = name_list->next;
	(gdb)
```

You can see that the loop has exited unexpectedly early.

Presumably you will now check the value of `i` and find that it is much too large.

Hopefully this information, combined with the suspiciously truncated version of
`name` will give you the clues you need to find the problem.

Using the information that `i` has been overwritten, can you figure out why
'Blake' changed to 'Blbke'?

It is because `i` has been incremented by 1 and the portion of the string that
overwrote `i` was `'Blake'`. Adding 1 to the ASCII character 'a' causes the
character 'b' to be created.


**NOTE:** This error should show you why it is dangerous to declare fixed length
arrays that will contain character strings.

It is difficult to know in advance how large they will be. Of course, you have
to declare the character array to be of some size so you should make it very
large, like maybe 1000 characters.

Fixed length errors are one of the holes that hackers use to attack operating
systems. They create a string that is too long for a fixed length array and th
extra characters can end up overwriting instructions in the OS, thus causing
the OS to do something it is not supposed to do.




## Recursion Debugging Example

This is a simple example of using `gdb` to print the stack.

Below we debug `rec2.c` using `gdb` and we set breakpoints at  lines 7 and 16.

Each time we hit a breakpoint, we use the `where` command to print the stack.

> Try it out.

You should see how the output matches the lecture notes.

### rec2.c

```
/*     1 */   a(int i)
/*     2 */   {
/*     3 */       int j;
/*     4 */
/*     5 */       j = i*5;
/*     6 */       printf('In procedure a: i = %d, j = %d\n', i, j);
/*     7 */       if (i > 0) a(i-1);
/*     8 */       printf('Later In procedure a: i = %d, j = %d\n', i, j);
/*     9 */   }
/*    10 */
/*    11 */   main()
/*    12 */   {
/*    13 */       int i;
/*    14 */
/*    15 */       i = 16;
/*    16 */       a(3);
/*    17 */       printf('main: i = %d\n', i);
/*    18 */   }
```


### Sample GDB Session

```c
	UNIX> gcc -g -c rec2.c
	UNIX> gcc -o rec2 rec2.o

	// compile and link in one statement
	UNIX> gcc -g rec2.c -o rec2

	UNIX> gdb rec2
```

```
	(gdb) break 7
	Breakpoint 1 at 0x15f1: file rec2.c, line 7.

	(gdb) break 16
	Breakpoint 2 at 0x163a: file rec2.c, line 16.

	(gdb) run
	Starting program: rec2

	Breakpoint 2, main () at rec2.c:16
	16        a(3);

	(gdb) print i
	$1 = 16

	(gdb) where
	#0  main () at rec2.c:16

	(gdb) cont
	Continuing.
	In procedure a: i = 3, j = 15

	Breakpoint 1, a (i=3) at rec2.c:7
	7         if (i > 0) a(i-1);

	(gdb) where
	#0  a (i=3) at rec2.c:7
	#1  0x1641 in main () at rec2.c:16

	(gdb) print j
	$2 = 15

	(gdb) cont
	Continuing.
	In procedure a: i = 2, j = 10

	Breakpoint 1, a (i=0) at rec2.c:7
	7         if (i > 0) a(i-1);

	(gdb) where
	#0  a (i=0) at rec2.c:7
	#1  0x1601 in a (i=1) at rec2.c:7
	#2  0x1601 in a (i=2) at rec2.c:7
	#3  0x1601 in a (i=3) at rec2.c:7
	#4  0x1641 in main () at rec2.c:16

	(gdb) cont
	Continuing.
	Later In procedure a: i = 0, j = 0
	Later In procedure a: i = 1, j = 5
	Later In procedure a: i = 2, j = 10
	Later In procedure a: i = 3, j = 15
	main: 16

	Program exited with code 011.

	(gdb) quit
```

