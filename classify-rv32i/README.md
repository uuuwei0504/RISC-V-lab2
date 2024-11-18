# Assignment 2: RISC-V Assembly and Instruction Pipeline

## Project Overview
This project focuses on implementing fundamental functions using the RISC-V assembly language. It includes:
- ReLU (Rectified Linear Unit) for non-linear activation in neural networks.
- ArgMax function to determine the index of the maximum value.
- Basic matrix operations such as dot product and matrix multiplication.

These implementations are designed for educational purposes, specifically in understanding the RISC-V instruction set and its application in deep learning.

---

## Features
1. **ReLU Function**: Implements the formula:
   $$
   f(x) = \max(0, x)
   $$
   - Outputs `x` if `x > 0`.
   - Outputs `0` if `x ≤ 0`.

2. **ArgMax Function**: Identifies the index of the maximum value in an array.

3. **Matrix Operations**:
   - Dot Product: Computes the scalar product of two vectors.
   - Matrix Multiplication: Multiplies two matrices efficiently.

---

# Relu function:
ReLU (Rectified Linear Unit) is a widely used activation function in deep learning. Its primary role is to introduce non-linearity, enabling neural networks to learn more complex patterns.

$$
f(x) = \max(0, x)
$$

This means:

- When the input `x > 0`, the output is `f(x) = x`.
- When the input `x ≤ 0`, the output is `f(x) = 0`.


### RISC-V

```asm
.globl relu

.text
# ==============================================================================
# FUNCTION: Array ReLU Activation
#
# Applies ReLU (Rectified Linear Unit) operation in-place:
# For each element x in array: x = max(0, x)
#
# Arguments:
#   a0: Pointer to integer array to be modified
#   a1: Number of elements in array
#
# Returns:
#   None - Original array is modified directly
#
# Validation:
#   Requires non-empty array (length ≥ 1)
#   Terminates (code 36) if validation fails
#
# Example:
#   Input:  [-2, 0, 3, -1, 5]
#   Result: [ 0, 0, 3,  0, 5]
# ==============================================================================
relu:
    li t0, 1             
    blt a1, t0, error     
    li t1, 0       

loop_start:
    bge t1, a1, return      
    slli t2, t1, 2       
    add t3, a0, t2      
    lw t4, 0(t3)         

    blez t4, replace      
    j next                
replace:
    sw x0, 0(t3)
next:
    addi t1, t1, 1       
    j loop_start          
error:
    li a0, 36            
    j exit
return:
    jr ra 

```
## result:
![image](https://hackmd.io/_uploads/H128YVOzJl.png)

The primary purpose of this program is to process each element in an array as follows:  
- If the element's value is greater than 0, it remains unchanged.  
- If the element's value is less than or equal to 0, it is set to 0.  

Additionally, the program includes input length validation to prevent invalid operations.


----
# dot function



The dot product of two vectors is a scalar value obtained by multiplying corresponding elements of two vectors and summing up the results.

## Formula:

$$
\text{Dot Product} = \sum_{i=1}^{n} \left( A[i] \cdot B[i] \right)
$$

**Where:**
- \( A \) and \( B \) are vectors.
- \( n \) is the length of the vectors.



## RISC-V
```asm
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

```

## result:

![image](https://hackmd.io/_uploads/rktOYEOfkg.png)


## Steps to Implement Dot Product

Below are the main steps to implement the dot product:



### Step 1: Input Validation
In the implementation, the first step is to validate the input:
- Check whether the input is valid (e.g., whether the two vectors have the same length).
- If the lengths do not match or the data is invalid, throw an error or handle the error appropriately.



### Step 2: Initialization
- Initialize a variable to store the result, e.g., `result = 0`.
- Use a loop variable `i` to iterate through each element of the vectors.



### Step 3: Compute the Dot Product
Inside the loop:
- Retrieve the corresponding elements from the two vectors, `A[i]` and `B[i]`.
- Multiply these elements and add the product to `result`.



### Step 4: Return the Result
- After completing the loop, return the accumulated `result` as the final dot product.






# ArgMax Function Introduction

The **ArgMax** function is widely used in mathematics, machine learning, and data analysis to find the index of the maximum value in a list or array.


## Definition
The ArgMax function is defined as:
$$
\text{ArgMax}(A) = \text{arg} \, \max_{i} \, A[i]
$$

Where:
- $A$ is a list or array of numbers.
- $\text{ArgMax}(A)$ returns the index $i$ where the value $A[i]$ is the largest in the array.




## Example
If \( A = [1, 3, 7, 2, 5] \), then:
\[
\text{ArgMax}(A) = 2
\]
because \( A[2] = 7 \) is the maximum value.



## Key Characteristics
- **Input**: A list or array of numerical values.
- **Output**: The index of the maximum value in the input array.
- **When multiple maximum values exist**: Typically, the index of the first occurrence is returned.

## RISC-V
```asm
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

```

## result:

![image](https://hackmd.io/_uploads/SJIIn4_z1l.png)



---

# Shift-and-Add Multiplier Explanation

My multiplier is based on the **Shift-and-Add method**, which is a classic approach for implementing binary multiplication in low-level hardware. The following explains its working principle:

---

## 1. Basic Concept

Binary multiplication can be seen as a combination of repeated addition and shifting:

- If a bit in the multiplier is `1`, the multiplicand is added to the result.
- After checking each bit of the multiplier:
  - The multiplicand is shifted left (equivalent to multiplying by 2).
  - The multiplier is shifted right (equivalent to dividing by 2), until all bits in the multiplier are processed.

---

## 2. Operational Steps

### **Initialize the Accumulator**
- Use an accumulator (e.g., `s0`) to store the final multiplication result.
- The initial value of the accumulator is set to 0.

### **Check the Multiplier Bits**
- Extract the least significant bit (LSB) of the multiplier using an AND operation.
- If the bit is `1`, add the current multiplicand to the accumulator. 
- If the bit is `0`, skip the addition step.

### **Perform the Shifting**
- After processing each bit:
  - Shift the multiplicand left by one position (representing multiplication by 2).
  - Shift the multiplier right by one position (representing division by 2 and removing the processed bit).

### **Repeat Until Completion**
- Repeat the above steps until all bits of the multiplier have been processed.

---

This method iteratively builds the result in the accumulator. The final value in the accumulator represents the multiplication result.


## RISC-V

```asm
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
    
```
## Implementation of Multiplication using Custom Function

This section demonstrates how to replace the direct `mul t0, t4, t5` instruction with a custom multiplication function (`mul_function`). This approach leverages stack operations and a subroutine to compute the multiplication result manually. Below is the detailed explanation of the process.

---

## Code Overview

```asm
addi sp, sp, -16        # Allocate stack space
sw a0, 0(sp)            # Save a0 on the stack
sw a1, 4(sp)            # Save a1 on the stack
sw a2, 8(sp)            # Save a2 on the stack
sw a5, 12(sp)           # Save a5 on the stack

mv a1, t4               # Move multiplier (t4) to a1
mv a2, t5               # Move multiplicand (t5) to a2

jal mul_function        # Call the multiplication function
# mul a0, a1, a2        # (Replaced with custom implementation)

add t0, t0, a0          # Accumulate the result: t0 += a0

lw a0, 0(sp)            # Restore a0 from the stack
lw a1, 4(sp)            # Restore a1 from the stack
lw a2, 8(sp)            # Restore a2 from the stack
lw a5, 12(sp)           # Restore a5 from the stack
addi sp, sp, 16         # Free stack space
```
---

# matmul function
## Matrix Multiplication Introduction

Matrix multiplication is a linear algebra operation that calculates the product of two matrices. Suppose we have two matrices $M0$ and $M1$:

- $M0$: Dimensions $rows_0 \times cols_0$
- $M1$: Dimensions $rows_1 \times cols_1$

---

## **Conditions for Matrix Multiplication**

To perform matrix multiplication, the following conditions must be satisfied:
1. $cols_0 = rows_1$: The number of columns in $M0$ must equal the number of rows in $M1$.
2. The resulting matrix $D$ will have dimensions $rows_0 \times cols_1$.

---

## **Computation Process**

The core of matrix multiplication is based on **row-by-column dot product operations**.

### **Result Matrix Elements**
Each element of the resulting matrix $D[i][j]$ is calculated as the dot product of the $i$-th row of $M0$ and the $j$-th column of $M1$:
$$
D[i][j] = \sum_{k=0}^{cols_0-1} M0[i][k] \cdot M1[k][j]
$$
Where:
- $i$: Row index of $M0$
- $j$: Column index of $M1$
- $k$: Intermediate index for dot product computation

---

## **How to Iterate Through Matrices**

1. **Outer Loop**:
   - Iterate over each row of $M0$ (indexed by $i$).
2. **Middle Loop**:
   - Iterate over each column of $M1$ (indexed by $j$).
3. **Inner Loop**:
   - Perform the dot product of the $i$-th row of $M0$ with the $j$-th column of $M1$.
   - Accumulate the products $M0[i][k] \cdot M1[k][j]$ into $D[i][j]$.

---

## **Result Storage**
- After computing the dot product for a specific pair of $i$ and $j$, store the result in the corresponding position $D[i][j]$ of the result matrix $D$.

## RISC-V

```asm
.globl matmul

.text
# =======================================================
# FUNCTION: Matrix Multiplication Implementation
#
# Performs operation: D = M0 × M1
# Where:
#   - M0 is a (rows0 × cols0) matrix
#   - M1 is a (rows1 × cols1) matrix
#   - D is a (rows0 × cols1) result matrix
#
# Arguments:
#   First Matrix (M0):
#     a0: Memory address of first element
#     a1: Row count
#     a2: Column count
#
#   Second Matrix (M1):
#     a3: Memory address of first element
#     a4: Row count
#     a5: Column count
#
#   Output Matrix (D):
#     a6: Memory address for result storage
#
# Validation (in sequence):
#   1. Validates M0: Ensures positive dimensions
#   2. Validates M1: Ensures positive dimensions
#   3. Validates multiplication compatibility: M0_cols = M1_rows
#   All failures trigger program exit with code 38
#
# Output:
#   None explicit - Result matrix D populated in-place
# =======================================================
matmul:
    # Error checks
    li t0 1
    blt a1, t0, error
    blt a2, t0, error
    blt a4, t0, error
    blt a5, t0, error
    bne a2, a4, error

    # Prologue
    addi sp, sp, -28
    sw ra, 0(sp)
    
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    sw s4, 20(sp)
    sw s5, 24(sp)
    
    li s0, 0 # outer loop counter
    li s1, 0 # inner loop counter
    mv s2, a6 # incrementing result matrix pointer
    mv s3, a0 # incrementing matrix A pointer, increments durring outer loop
    mv s4, a3 # incrementing matrix B pointer, increments during inner loop 
    
outer_loop_start:
    #s0 is going to be the loop counter for the rows in A
    li s1, 0
    mv s4, a3
    blt s0, a1, inner_loop_start

    j outer_loop_end
    
inner_loop_start:
# HELPER FUNCTION: Dot product of 2 int arrays
# Arguments:
#   a0 (int*) is the pointer to the start of arr0
#   a1 (int*) is the pointer to the start of arr1
#   a2 (int)  is the number of elements to use = number of columns of A, or number of rows of B
#   a3 (int)  is the stride of arr0 = for A, stride = 1
#   a4 (int)  is the stride of arr1 = for B, stride = len(rows) - 1
# Returns:
#   a0 (int)  is the dot product of arr0 and arr1
    beq s1, a5, inner_loop_end

    addi sp, sp, -24
    sw a0, 0(sp)
    sw a1, 4(sp)
    sw a2, 8(sp)
    sw a3, 12(sp)
    sw a4, 16(sp)
    sw a5, 20(sp)
    
    mv a0, s3 # setting pointer for matrix A into the correct argument value
    mv a1, s4 # setting pointer for Matrix B into the correct argument value
    mv a2, a2 # setting the number of elements to use to the columns of A
    li a3, 1 # stride for matrix A
    mv a4, a5 # stride for matrix B
    
    jal dot
    
    mv t0, a0 # storing result of the dot product into t0
    
    lw a0, 0(sp)
    lw a1, 4(sp)
    lw a2, 8(sp)
    lw a3, 12(sp)
    lw a4, 16(sp)
    lw a5, 20(sp)
    addi sp, sp, 24
    
    sw t0, 0(s2)
    addi s2, s2, 4 # Incrememtning pointer for result matrix
    
    li t1, 4
    add s4, s4, t1 # incrememtning the column on Matrix B
    
