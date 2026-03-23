CC = icx
CFLAGS = -Wall -Wextra -Iinclude/  -qmkl -O3 -ipo 
# -g -fsanitize=undefined,address -fno-omit-frame-pointer
# -O3 -ipo 

TARGETS = heart_sharp_2DFC
LIBS = q_patch_lib num_linalg_lib s_patch_lib fc_lib curve_seq_lib c_patch_lib r_cartesian_mesh_lib fc2D_lib



OBJS = $(addprefix out/,$(LIBS:=.o)) $(addprefix out/,$(TARGETS:=.o)) 
BINS = $(addprefix bin/,$(TARGETS))

# find <dir> is the command to find files in a directory
# ! -name .gitignore tells find to ignore the .gitignore
# -type f only finds files
# -delete deletes all the files found
CLEAN_COMMAND = find out/ ! -name .gitignore -type f -delete && \
find bin/ ! -name .gitignore -type f -delete

all: $(BINS)

$(BINS): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

out/%.o: src/%.c
	$(CC) $(CFLAGS) -c $^ -o $@

out/%.o: examples/%.c
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	$(CLEAN_COMMAND)

# This special rule tells Make that "all", "clean", and "test" are rules
# that don't build a file.
.PHONY: all clean
# Tells Make not to delete the .o files after the executable is built
.PRECIOUS: out/%.o