PROGRAM = nn
LIBS    = -lm
SOURCES = main.c nn.c
FLAGS   = -Wall -std=c99
CC      = gcc

all:
	${CC} ${FLAGS} ${SOURCES} ${LIBS} -o ${PROGRAM}

clean:
	rm -f ${PROGRAM}
