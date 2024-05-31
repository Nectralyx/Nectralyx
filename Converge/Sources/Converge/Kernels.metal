
#include <metal_stdlib>
using namespace metal;

uint flatten(uint2 index, uint width) {
    return index.y * width + index.x;
}
kernel void add_arrays(device const float*arr1 [[ buffer(0) ]],
                       device const float*arr2 [[ buffer(1) ]],
                       device float*result [[ buffer(2) ]],
                       device const uint& rows,
                       device const uint& cols,
                       uint2 index [[thread_position_in_grid]]
                       ) {
    uint row = index.y;
    
    result[row] = arr1[row] + arr2[row];
}


kernel void matrixMultiply(const device float* x [[buffer(0)]],
                           const device float* y [[buffer(1)]],
                           device float* result [[buffer(2)]],
                           device const uint& rowsX, device const uint& colsX, device const uint& colsY,
                           uint2 grid [[thread_position_in_grid]]) {
    uint row = grid.y;
    uint col = grid.x;
    
    if (row < rowsX && col < colsY) {
        float sum = 0.0;
        for (uint k = 0; k < colsX; ++k) {
            sum += x[row * colsX + k] * y[k * colsY + col];
        }
        result[row * colsY + col] = sum;
    }
}




// Function to add two matrices represented as flattened arrays
void addMatrices(const device float* matrixA, const device float* matrixB, device float* resultMatrix, uint rows, uint columns) {
    // Calculate the total number of elements in the matrices
    uint elementCount = rows * columns;
    
    // Perform matrix addition element-wise
    for (uint i = 0; i < elementCount; ++i) {
        resultMatrix[i] = matrixA[i] + matrixB[i];
    }
}

kernel void concatenateWeightMatrices(const device float* matricesBuffers [[ buffer(0) ]],
                                      device float* resultMatrix [[ buffer(1) ]],
                                      constant uint* matrixOffsets [[ buffer(2) ]],
                                      const device uint& numMatrices,
                                      const device uint& numRows,
                                      const device uint& numCols,
                                      uint2 grid [[thread_position_in_grid]]) {
    // Calculate the global index
    uint globalID = (grid.y * numCols) + grid.x;

    // Check if the global index is within the bounds of the result matrix
    if (globalID < numRows * numCols) {
        // Calculate row and column indices
        uint row = globalID / numCols;
        uint col = globalID % numCols;

        // Calculate the index of the matrix containing the current element
        uint matrixIndex = 0;
        uint rowOffset = 0;
        while (rowOffset + matrixOffsets[matrixIndex] <= row) {
            rowOffset += matrixOffsets[matrixIndex];
            matrixIndex++;
        }

        // Calculate the row index within the current matrix
        uint rowIndexInMatrix = row - rowOffset;

        // Calculate the starting index of the current matrix in the flattened buffer
        uint matrixStartIndex = 0;
        for (uint i = 0; i < matrixIndex; ++i) {
            matrixStartIndex += matrixOffsets[i] * numCols;
        }
        
        // Copy the element from the corresponding matrix to the result matrix
        resultMatrix[globalID] = matricesBuffers[matrixStartIndex + (rowIndexInMatrix * numCols) + col];
    }
}

kernel void concatenateArrays(const device float* array1 [[ buffer(0) ]],
                              const device float* array2 [[ buffer(1) ]],
                              const device float* array3 [[ buffer(2) ]],
                              const device float* array4 [[ buffer(3) ]],
                              device float* concatenatedArray [[ buffer(4) ]],
                              const device uint& arraySize,
                              uint index [[ thread_position_in_grid ]]) {
    uint array1Length = arraySize;
    uint array2Length = arraySize;
    uint array3Length = arraySize;
    uint array4Length = arraySize;

    if (index < array1Length) {
        concatenatedArray[index] = array1[index];
    } else if (index < array1Length + array2Length) {
        concatenatedArray[index] = array2[index - array1Length];
    } else if (index < array1Length + array2Length + array3Length) {
        concatenatedArray[index] = array3[index - (array1Length + array2Length)];
    } else if (index < array1Length + array2Length + array3Length + array4Length) {
        concatenatedArray[index] = array4[index - (array1Length + array2Length + array3Length)];
    }
}

kernel void splitArray(const device float* concatenatedArray [[ buffer(0) ]],
                       device float* array1 [[ buffer(1) ]],
                       device float* array2 [[ buffer(2) ]],
                       device float* array3 [[ buffer(3) ]],
                       device float* array4 [[ buffer(4) ]],
                       const device uint& arrayLength,
                       uint2 index [[ thread_position_in_grid ]]) {
    uint split = index.x;
    uint element = index.y;
    uint numElements = arrayLength / 4;
    if (split == 0) {
        array1[element] = concatenatedArray[split * numElements + element];
    } else if (split == 1) {
        array2[element] = concatenatedArray[split * numElements + element];
    } else if (split == 2) {
        array3[element] = concatenatedArray[split * numElements + element];
    } else {
        array4[element] = concatenatedArray[split * numElements + element];
    }
}

kernel void elementMultiply(const device float* matrixA [[ buffer(0) ]],
                            const device float* matrixB [[ buffer(1) ]],
                            device float* result [[ buffer(2) ]],
                            uint2 index [[thread_position_in_grid]]) {
    uint row = index.y;
    result[row] = matrixA[row] * matrixB[row];
}
