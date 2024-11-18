.globl argmax

.text
# =================================================================
# FUNCTION: Maximum Element First Index Finder
#
# Scans an integer array to find its maximum value and returns the
# position of its first occurrence. In cases where multiple elements
# share the maximum value, returns the smallest index.
#
# Arguments:
#   a0 (int *): Pointer to the first element of the array
#   a1 (int):  Number of elements in the array
#
# Returns:
#   a0 (int):  Position of the first maximum element (0-based index)
#
# Preconditions:
#   - Array must contain at least one element
#
# Error Cases:
#   - Terminates program with exit code 36 if array length < 1
# =================================================================
argmax:
    li t6, 1
    blt a1, t6, handle_error   #i>size時跳出

    lw t0, 0(a0) 

    li t1, 0
    li t2, 0
loop_start:
    addi t2,t2,1
    bge t2,a1 return
    slli t4,t2,2
    add t3,t4,a0
    lw t5,0(t3) #t3是array[i]的值
    blt t5,t0 loop_start
    mv t0,t5
    mv t1,t2
    j loop_start

    
    # TODO: Add your own implementation

handle_error:
    li a0, 36
    j exit
return:
    mv a0,t1
    jr ra
