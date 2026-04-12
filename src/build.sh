 #!/bin/sh

gcc main.c -o sagan_test.out -O3 -fsanitize=address -g && ./sagan_test.out
