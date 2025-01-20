COM = lamuta main.lm
SRC ?= test.lm  # デフォルトのソースファイル
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

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(ASM) $(OBJ) $(EXE)

# 明示的に指定するPHONYターゲット
.PHONY: all clean run
