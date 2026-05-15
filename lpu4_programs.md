| IPC Method | Speed | Complexity | Scope | Typical Use Case |
| --- | --- | --- | --- | --- |
| Shared Memory | Very High | High | Same system only | Fast data exchange |
| Message Queues | Medium | Medium | Same system only | Command passing |
| Pipes | Medium | Low | Same system only | Producer-consumer |
| Sockets | Variable | High | Local + Network | Distributed apps |

```c
h = init_CS("xxx");              // Critical section
h = init_semaphore(20,"xxx");    // Semaphore with initial value 20
h = init_event("xxx");           // Event object
h = init_condition("xxx");       // Condition variable
h = init_message_buffer(100,"xxx"); // Message buffer of size 100
```
⚙️ Example: Using pipe() in UNIX
```c
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main() {
    int data_processed;
    int file_pipes[2];
    const char some_data[] = "123";
    pid_t fork_result;

    if (pipe(file_pipes) == 0) {
        fork_result = fork();
        if (fork_result == (pid_t)-1) {
            fprintf(stderr, "fork failure");
            exit(EXIT_FAILURE);
        }
        if (fork_result == (pid_t)0) {
            close(0);
            dup(file_pipes[0]);
            close(file_pipes[0]);
            close(file_pipes[1]);
            execlp("od", "od", "-c", (char *)0);
            exit(EXIT_FAILURE);
        } else {
            close(file_pipes[0]);
            data_processed = write(file_pipes[1], some_data, strlen(some_data));
            close(file_pipes[1]);
            printf("%d - wrote %d bytes\n", (int)getpid(), data_processed);
        }
    }
    exit(EXIT_SUCCESS);
}

```
✍️ Simplified FIFO Writer (fifo_writer.c)

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#define FIFO_NAME "/tmp/my_fifo"

int main() {
    int fd;
    const char *msg = "Hello from writer!\n";

    // Create FIFO if it doesn’t exist
    if (access(FIFO_NAME, F_OK) == -1) {
        if (mkfifo(FIFO_NAME, 0666) != 0) {
            perror("mkfifo");
            exit(EXIT_FAILURE);
        }
    }

    // Open FIFO for writing
    fd = open(FIFO_NAME, O_WRONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    // Write message
    write(fd, msg, sizeof(msg));
    close(fd);

    printf("Writer finished.\n");
    return 0;
}

```
✍️ Simplified FIFO Reader (fifo_reader.c)
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#define FIFO_NAME "/tmp/my_fifo"

int main() {
    int fd;
    char buffer[100];

    // Open FIFO for reading
    fd = open(FIFO_NAME, O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    // Read message
    read(fd, buffer, sizeof(buffer));
    close(fd);

    printf("Reader got: %s\n", buffer);
    return 0;
}

```
✍️ Simplified Win32 Example

```c
#include <windows.h>
#include <iostream>

int main(int argc, char *argv[]) {
    HANDLE rh, wh;
    SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };

    // Create anonymous pipe
    if (!CreatePipe(&rh, &wh, &sa, 0)) {
        std::cerr << "Pipe creation failed\n";
        return 1;
    }

    // Setup child 1 (writer)
    STARTUPINFO si1 = { sizeof(si1) };
    PROCESS_INFORMATION pi1;
    si1.hStdOutput = wh; // p1 writes into pipe
    si1.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si1.dwFlags    = STARTF_USESTDHANDLES;
    CreateProcess(argv[1], NULL, NULL, NULL, TRUE, 0, NULL, NULL, &si1, &pi1);
    CloseHandle(wh); // parent no longer needs write end

    // Setup child 2 (reader)
    STARTUPINFO si2 = { sizeof(si2) };
    PROCESS_INFORMATION pi2;
    si2.hStdInput  = rh; // p2 reads from pipe
    si2.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si2.dwFlags    = STARTF_USESTDHANDLES;
    CreateProcess(argv[2], NULL, NULL, NULL, TRUE, 0, NULL, NULL, &si2, &pi2);

    // Wait for both children
    WaitForSingleObject(pi1.hProcess, INFINITE);
    WaitForSingleObject(pi2.hProcess, INFINITE);

    CloseHandle(pi1.hProcess);
    CloseHandle(pi2.hProcess);
    CloseHandle(rh);

    return 0;
}

```
✍️ Simplified Example 1: Reading from a Command popen() pclose()
```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *fp;
    char buffer[100];

    // Run "ls" and read its output
    fp = popen("ls", "r");
    if (fp == NULL) {
        perror("popen failed");
        exit(1);
    }

    // Read output line by line
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        printf("Got: %s", buffer);
    }

    pclose(fp);
    return 0;
}

```
✍️ Simplified Example 2: Writing to a Command popen() pclose()
```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *fp;

    // Run "sort" and write data into it
    fp = popen("sort", "w");
    if (fp == NULL) {
        perror("popen failed");
        exit(1);
    }

    // Send unsorted data
    fprintf(fp, "banana\n");
    fprintf(fp, "apple\n");
    fprintf(fp, "cherry\n");

    pclose(fp);
    return 0;
}
```

✍️ Simplified Client/Server Example MSG-Queue
Server (Receiver)
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/msg.h>

struct msg {
    long type;
    char text[100];
};

int main() {
    struct msg message;
    int msgid = msgget(1234, 0666 | IPC_CREAT);

    while (1) {
        msgrcv(msgid, &message, sizeof(message.text), 0, 0);
        printf("Received: %s", message.text);
        if (strncmp(message.text, "end", 3) == 0) break;
    }

    msgctl(msgid, IPC_RMID, NULL); // delete queue
    return 0;
}
```

Client (Sender)
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/msg.h>

struct msg {
    long type;
    char text[100];
};

int main() {
    struct msg message;
    int msgid = msgget(1234, 0666 | IPC_CREAT);

    while (1) {
        printf("Enter text: ");
        fgets(message.text, sizeof(message.text), stdin);
        message.type = 1;
        msgsnd(msgid, &message, strlen(message.text)+1, 0);

        if (strncmp(message.text, "end", 3) == 0) break;
    }
    return 0;
}
```

✍️ Simplified Example - Semaphore

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sem.h>
#include <sys/ipc.h>

union semun {
    int val;
};

int main() {
    int sem_id;
    struct sembuf sem_op;

    // Create semaphore
    sem_id = semget((key_t)1234, 1, 0666 | IPC_CREAT);

    // Initialize semaphore to 1 (resource available)
    union semun arg;
    arg.val = 1;
    semctl(sem_id, 0, SETVAL, arg);

    // P (wait) operation
    sem_op.sem_num = 0;
    sem_op.sem_op = -1;   // decrement
    sem_op.sem_flg = SEM_UNDO;
    semop(sem_id, &sem_op, 1);

    // Critical section
    printf("Process %d in critical section\n", getpid());
    sleep(2);

    // V (signal) operation
    sem_op.sem_op = 1;    // increment
    semop(sem_id, &sem_op, 1);

    printf("Process %d left critical section\n", getpid());

    // Delete semaphore
    semctl(sem_id, 0, IPC_RMID, arg);

    return 0;
}
```
