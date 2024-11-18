.globl dot

.text
# =======================================================
# FUNCTION: Strided Dot Product Calculator
#
# Calculates sum(arr0[i * stride0] * arr1[i * stride1])
# where i ranges from 0 to (element_count - 1)
#
# Args:
#   a0 (int *): Pointer to first input array
#   a1 (int *): Pointer to second input array
#   a2 (int):   Number of elements to process
#   a3 (int):   Skip distance in first array
#   a4 (int):   Skip distance in second array
#
# Returns:
#   a0 (int):   Resulting dot product value
#
# Preconditions:
#   - Element count must be positive (>= 1)
#   - Both strides must be positive (>= 1)
#
# Error Handling:
#   - Exits with code 36 if element count < 1
#   - Exits with code 37 if any stride < 1
# =======================================================
dot:
   
    li t0, 1
    blt a2, t0, error_terminate  
    blt a3, t0, error_terminate   
    blt a4, t0, error_terminate  

    addi sp, sp,-8
    sw t0, 0(sp)
    sw ra, 4(sp)	
    li t0, 0          # t0 = result
    li t1, 0          # t1 = index i
    slli t2, a3, 2    # t2 = a3 * 4 
    slli t3, a4, 2    # t3 = a4 * 4 

loop_start:
    bge t1, a2, loop_end


    lw t4, 0(a0)     # t4 = arr1[i]
    lw t5, 0(a1)     # t5 = arr2[i]
    addi sp,sp,-16
   
    sw a0,0(sp)
    sw a1,4(sp)
    sw a2,8(sp)
    sw a5,12(sp)	
    mv a1,t4
    mv a2,t5

    jal mul_function
    #mul a0, a1, a2

    add t0, t0, a0   # result += t6
    lw a0,0(sp)
    lw a1,4(sp)
    lw a2,8(sp)
    lw a5,12(sp)
    addi sp,sp,16

    add a0, a0, t2   
    add a1, a1, t3   


    addi t1, t1, 1
    j loop_start

loop_end:

    mv a0, t0
    lw t0, 0(sp)
    lw ra, 4(sp)
    addi sp, sp, 8

    jr ra

error_terminate:
    blt a2, t0, set_error_36
    li a0, 37
    j exit

set_error_36:
    li a0, 36
    j exit












mul_function:
    addi sp, sp, -16      
    sw s0, 0(sp)      
    sw s1, 4(sp)      
    sw s2, 8(sp)       
    sw s3, 12(sp)      

    li s0, 0          

mul_loop:
    andi s3, a2, 1        
    beqz s3, skip_add1    
    add s0, s0, a1       

skip_add1:
    slli a1, a1, 1        
    srli a2, a2, 1
    bnez a2, mul_loop       

    mv a0, s0          

    lw s0, 0(sp)       
    lw s1, 4(sp)       
    lw s2, 8(sp)        
    lw s3, 12(sp)      
    addi sp, sp, 16     

    ret





multiply: 
    
    li a0, 0 
multiply_loop:
    andi a5, a2, 1     
    beqz a5, skip_add  
    add a0, a0, a1     

skip_add:
    slli a1, a1, 1      
    srli a2, a2, 1      
    bnez a2, multiply_loop 
    ret    
