COM = lamuta compiler.lm
ASM = $(SRC:.lm=.ll)
OBJ = $(SRC:.lm=.bc)
EXE = $(SRC:.lm=)

all: $(EXE)

$(ASM): $(SRC)
	$(COM) $(SRC) > $(ASM)
$(OBJ): $(ASM)
	llvm-as $(ASM) -o $(OBJ)
$(EXE): $(OBJ)
	clang $(OBJ) -o $(EXE)

clean:
	rm -f $(ASM) $(OBJ) $(EXE)

.PHONY: all clean run
