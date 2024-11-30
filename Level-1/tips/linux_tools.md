## Linux Tools

- awk
- bat (batmab and prettybat)
- duf
- ffmpeg
- fkill
- fzf
- hstr
- httpie
- iPerf
- js
- mktemp
- od (view binary file)
- rich
- ripgrep
- screen
- Shellcheck
- tmux
- vnstat


Bash is a command interpreter for Linux and MacOS systems. Here are some tools and tips that will help in the daily battle against bugs and deadlines [1].

ffmpeg is a very useful utility for CV projects [5]. 

speedtest-cli is a command-line tool for testing internet bandwidth using speedtest.net. 

iPerf is a great way to test your LAN speed (rather than your Internet speed [6]. 


## coreutils

### awk

awk is a pattern scanning and text processing language which is also considered a programming language specifically designed for processing text.

### column

The column utility helps create columns in text output and even generate whole tables.

The output is automatically formatted into neatly aligned columns.

```bash
  echo -e "one two three\n1 2 3\n111111 222222 333322" | column -t
```

### fold

fold can wrap input lines to specific lengths. 

The length can be defined in bytes or spaces. 

Suppose we have an input line that is six characters long. We want to limit each line to only five characters and wrap the remainder. Using fold we can do this with the following command:

```bash
  echo "123456" | fold -w 5
```

### httpie

HTTPie is a command-line HTTP client to make CLI interaction with web services as human-friendly as possible. 

HTTPie is designed for testing, debugging, and generally interacting with APIs and HTTP servers. 

The http and https commands allow for creating and sending arbitrary HTTP requests using a simple and natural syntax and provide formatted and colorized output.

### jq

[jq](https://jqlang.github.io/jq/)

`jq` is a lightweight and flexible command-line JSON processor.

jq is like sed for JSON data - you can use it to slice and filter and map and transform structured data with the same ease that sed, awk, grep and friends let you play with text.

jq is written in portable C, and it has zero runtime dependencies. 

You can download a single binary, scp it to a far away machine of the same type, and expect it to work.

jq can mangle the data format that you have into the one that you want with very little effort, and the program to do so is often shorter and simpler than you'd expect.

### Lucidchart

[Lucidchart](https://www.lucidchart.com/pages/)

Lucidchart is a diagraming tool that also has shared space for collaboration and the ability to make notes next to diagrams.

### Multipass

[Multipass](https://multipass.run) is a VM platform (perhaps better than VirtualBox and VMWare).

Multipass can be used on Ubuntu, Windows 10, and macOS (including Apple M1) with exactly the same command set.

[Use Linux Virtual Machines with Multipass](https://medium.com/codex/use-linux-virtual-machines-with-multipass-4e2b620cc6)


### pwdx

The pwdx utility will allow you to get the current working directory of a running process. We pass it the PID and it tells us where that process is working from.

Suppose we wanted to find out where the cron process was working from on our machine. 

First we just need to figure out the PID by searching for it using ps like so:

```bash
  ps aux | grep cron
```

Here we see that the PID of cron is 612. 

Now all we need to do is determine the working directory of that process by passing it to pwdx:

```bash
  sudo pwdx 612
```

The pwdx utility can be a very valuable troubleshooting tool for chasing down directory scoping issues. A quick check with pwdx and you can figure out exactly where a process thinks it should be running from.


### rich

rich makes it easy to add colour and style to terminal output. It can also render pretty tables, progress bars, markdown, syntax highlighted source code, tracebacks, and more — out of the box.

### Screen

[Screen](https://linuxize.com/post/how-to-use-linux-screen/)

Screen is a GNU linux utility that lets you launch and use multiple shell sessions from a single ssh session. The process started with screen can be detached from session and then reattached at a later time. So your experiments can be run in the background, without the need to worry about session closing, or terminal crashing.


### sg

The `sg` comman allows you to directly execute a command using the permissions of another group you specify.

To execute ls from the admin group, you would pass in the following:

```bash
  sg admin ls
```

The sg command will switch the ls command to run using the permissions of the admin group. 

Once the command exits you will be returned to the normal group permissions you had prior to execution.

The sg command is helpful for testing out new group permissions or quickly switching contexts to run a program from another group.

### xxd

The `xxd` utility is one of the multitude of ways to perform hexadecimal dumps on Linux. 

The advantage of `xxd` is that you can both dump and restore hex using this utility. There are also a lot of configurable flags and you can perform patches on binary files as well.



## tmux

`tmux` is a terminal multiplexer that allows you to access a tmux terminal using multiple virtual terminals.

tmux can be used when you want to detach processes from their controlling terminals which allows remote sessions to remain active without being visible.

tmux takes advantage of a client-server model which allows you to attach terminals to a tmux session which means: 

- You can run several terminals at once concurrently off a single tmux session without spawning any new terminal sessions.

- Sudden disconnects from a cloud server running tmux will not kill the processes running inside the tmux session.

tmux also includes a window pane design which allows running more than one terminal on a single screen.

### Starting Your First Tmux Session

To start your first Tmux session, simply type `tmux` in the console:

```bash
  tmux

  # get a list of currently running sessions
  tmux ls
```

This will open a new session, create a new window, and start a shell in that window.

You will notice a status line at the bottom of the screen that shows information about the current session.

You can now run your first Tmux command.

```
  # get a list of all commands
  Ctrl+b ?
```

### Creating Named Tmux Sessions

By default, Tmux sessions are named numerically. Named sessions are useful when you run multiple Tmux sessions. 


```bash
  # create a new named session
  tmux new -s session_name
```

It is always a good idea to choose a descriptive session name.

### Detach from Tmux Session

Detach from the Tmux session and return to your normal shell: 

```
  Ctrl+b d
```

The program running in the Tmux session will continue to run after you detach from the session.

### Reattach to Tmux Session

To attach to a session first, you need to find the name of the session. 

```bash
  # attach to session 0
  tmux attach-session -t 0
```

### tmux Commands

```
    Ctrl+b c  Create a new window (with shell)
    Ctrl+b d  Detach from tmux session and return to normal shell
    Ctrl+b w  Display an interactive list of windows
    Ctrl+b 0  Switch to window 0 (by number)
    Ctrl+b ,  Rename the current window
    
    Ctrl+b %  Split current pane horizontally into two panes
    Ctrl+b "  Split current pane vertically into two panes
    Ctrl+b o  Go to the next pane
    Ctrl+b ;  Toggle between the current and previous pane
    Ctrl+b {  Move pane left
    Ctrl+b }  Move pane right
    Ctrl+b x  Close the current pane
    
    Ctrl+b arrow  Switch between panes
```

### Customizing Tmux

When Tmux is started, it reads its configuration parameters from `~/.tmux.conf` if the file is present.

### Resize Panes

The size of each pane can be changed: CTRL+b:

The bottom bar will change from green to yellow:

- resize-pane -D : Moves the boundary line for the pane downward.
- resize-pane -U : Moves the boundary line for the pane upward.
- resize-pane -R : Moves the boundary line for the pane right.
- resize-pane -L : Moves the boundary line for the pane left.

We can specify a specific number of cells to move the boundary line.

resize-pane -U 10 : Moves the boundary line up 10 cells.

We can specify a different cell than the one we are currently working in. 

Resize Cell 2 (lower right):

resize-pane –t 2 --R 5 : Moves the boundary line 5 cells to the right.

### tmux Tutorials

[Getting started with Tmux](https://linuxize.com/post/getting-started-with-tmux/)

[How to Use Tmux + Cheat Sheet](https://www.hostinger.com/tutorials/tmux-beginners-guide-and-cheat-sheet/)



## Moreutils

Moreutils is a package of additional command-line tools designed to fill in the gaps left by coreutils [5]. 

Here is a list of all commands in the package [5]:

chronic: Runs a command quietly, only outputting if the command fails. This is particularly useful for cron jobs where you only want to be notified when something goes wrong.

combine: Merges lines from two files using Boolean operations like AND, OR, and XOR, offering more control over file merging tasks.

errno: Allows you to look up errno names and descriptions, helping you quickly understand error codes without needing to consult documentation.

ifdata: Retrieves network interface information without requiring you to parse the output of ifconfig, making network management more straightforward.

ifne: Runs a program only if the standard input is not empty. This is useful for conditional execution based on input.

isutf8: Checks if a file or standard input is in UTF-8 format, ensuring data encoding consistency across your projects.

lckdo: Executes a program while holding a lock, preventing other processes from running the same command simultaneously, which is crucial in multi-process environments.

mispipe: Pipes two commands together, returning the exit

parallel: Executes multiple commands concurrently, taking full advantage of multi-core processors to speed up operations like batch processing or testing.

pee: Similar to tee, but instead of writing to multiple files, it sends the standard input to multiple commands simultaneously, allowing you to process input through several commands at once.

sponge: Soaks up the standard input before writing it to a file, preventing issues with overwriting files that are still being read, a common problem when chaining commands.

ts: Adds a timestamp to each line of input, which is particularly useful for logging or monitoring the timing of events in real-time.

vidir: Opens a directory in your text editor, allowing you to rename files as if they were lines in a text file, which is much faster than renaming files one by one.

vipe: Inserts a text editor into a pipeline, enabling you to manually edit the content of a command’s output before passing it on to the next command.
zrun: Automatically uncompresses arguments to a command, simplifying the process of working with compressed files.



## References

[1]: [20 Linux Tricks for the Pro Developers](https://medium.com/codex/linux-tricks-for-the-pro-developers-48edadc7017e)

[2]: [10 Practical Uses of AWK Command for Text Processing](https://betterprogramming.pub/10-practical-use-of-awk-command-in-linux-unix-26fbd92f1112)

[3]: [Display Rich Text In The Console Using rich](https://towardsdatascience.com/get-rich-using-python-af66176ece8f?source=linkShare-d5796c2c39d5-1641842633)

[4]: [6 More Unique Linux Utilities You Forgot About](https://betterprogramming.pub/6-more-unique-linux-utilities-you-forgot-about-1215ac0c58da)

[5]: [8 Advanced Linux Command Line Tools](https://itnext.io/8-advanced-linux-command-line-tools-9d81258c3165)

[6]: [3 handy command-line internet speed tests](https://opensource.com/article/20/1/internet-speed-tests)


[speedtest-cli](https://github.com/sivel/speedtest-cli)

