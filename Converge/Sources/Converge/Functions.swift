//
//  File.swift
//  
//
//  Created by Morgan Keay on 2024-04-16.
//

import Foundation
import Metal

// Activations

func Linear(_ x: Double) -> Double {
    return x
}

func ReLU(_ x: Double) -> Double {
    return max(0, x)
}

// Define the derivative of the ReLU function
func ReLUDerivative(_ x: Double) -> Double {
    return x >= 0 ? 1.0 : 0.0
}

func Sigmoid(_ x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}
func SigmoidDerivative(_ x: Double) -> Double {
    let s = Sigmoid(x)
    return s * (1.0 - s)
}
func Tanh(_ x: Double) -> Double {
    return (exp(x) - exp(-x) / exp(x) + exp(-x))
}

func TanhDerivative(_ x: Double) -> Double {
    let t = tanh(x)
    return 1.0 - t * t
}

func LeakyReLU(_ x: Double, alpha: Double = 0.01) -> Double {
    return x > 0 ? 1.0 : alpha * x
}

func LeakyReLUDerivative(_ x: Double, alpha: Double = 0.01) -> Double {
    return x > 0 ?  1.0 : alpha
}

func modifiedReLU(_ x: Double, threshold: Double = 0.0, leak: Double = 1.5) -> Double {
    if x >= threshold {
        return x
    } else {
        return leak * x
    }
}

func modifiedReLUDerivative(_ x: Double, threshold: Double = 0.0, leak: Double = 1.5) -> Double {
    if x >= threshold {
        return 1.0
    } else {
        return leak
    }
}

// Define dictionary of activations
let activationFunctions: [String: (Double) -> Double] = [
        "ReLU": ReLU,
        "Sigmoid": Sigmoid,
        "Tanh": Tanh,
        "LeakyReLU": { LeakyReLU($0) }, // To include the alpha parameter
        "ModifiedReLU": { modifiedReLU($0) },
        "Linear": Linear
]

let derivativeFunctions: [String: (Double) -> Double] = [
    "ReLU": ReLUDerivative,
    "Sigmoid": SigmoidDerivative,
    "Tanh": TanhDerivative,
    "LeakyReLU": { LeakyReLUDerivative($0) }, // To include the alpha parameter
    "ModifiedReLU": { modifiedReLUDerivative($0) }
]

// Operations

func add(_ vector1: [Double], _ vector2: [Double]) -> [Double] {
    guard vector1.count == vector2.count else {
        fatalError("Could not add vectors: add()")
    }
    var result: [Double] = []
    for i in 0..<vector1.count {
        result.append(vector1[i] + vector2[i])
    }
    return result
}

func matrixMultiply(_ x: [[Double]], _ y: [[Double]]) -> [[Double]]? {
    guard x[0].count == y.count else {
        // Matrices can't be multiplied, return nil or throw an error
        print("Could not multiply matrices: MatrixMultiply()")
        return nil
    }

    var output: [[Double]] = []
    for i in 0..<x.count {
        var row: [Double] = []
        for j in 0..<y[0].count {
            let dotProduct = zip(x[i], y.map { $0[j] }).map(*).reduce(0, +)
            row.append(dotProduct)
        }
        output.append(row)
    }

    return output
}

func outerProduct(_ matrixA: [[Double]], _ matrixB: [[Double]]) -> [[Double]] {
    let rowsA = matrixA.count
    let colsA = matrixA[0].count
    let rowsB = matrixB.count
    let colsB = matrixB[0].count
    
    var result = Array(repeating: Array(repeating: 0.0, count: colsB * colsA), count: rowsA * rowsB)
    
    for i in 0..<rowsA {
        for j in 0..<colsA {
            for k in 0..<rowsB {
                for l in 0..<colsB {
                    result[i * rowsB + k][j * colsB + l] = matrixA[i][j] * matrixB[k][l]
                }
            }
        }
    }
    
    return result
}

func elementMultiply(_ x: [[Double]], _ y: [[Double]]) -> [[Double]]? {
    guard x.count == y.count && x[0].count == y[0].count && !x.isEmpty else {
        print("Could not multiply matrices: elementMultiply()")
        return nil
    }
    var output: [[Double]] = Array(repeating: Array(repeating: 0.0, count: x[0].count), count: x.count)
    for row in 0..<x.count {
        for col in 0..<x[0].count {
            output[row][col] = x[row][col] * y[row][col]
        }
    }
    return output
}

func matrixAdd(_ x: [[Double]], _ y: [[Double]]) -> [[Double]]? {
    guard x.count == y.count && x[0].count == y[0].count && !x.isEmpty else {
        print("Could not add matrices: matrixAdd()")
        return nil
    }
    var output: [[Double]] = Array(repeating: Array(repeating: 0.0, count: x[0].count), count: x.count)
    for row in 0..<x.count {
        for col in 0..<x[0].count {
            output[row][col] = x[row][col] + y[row][col]
        }
    }
    return output
}

func concatenateWeightMatrices(_ matrices: [[[Double]]]) -> [[Double]] {
    let numCols = matrices[0][0].count
    let numRows = matrices.reduce(0, { $0 + $1.count })

    var result = Array(repeating: Array(repeating: 0.0, count: numCols), count: numRows)

    var rowOffset = 0
    for matrix in matrices {
        let numRowsInMatrix = matrix.count

        for i in 0..<numRowsInMatrix {
            for j in 0..<numCols {
                result[rowOffset + i][j] = matrix[i][j]
            }
        }

        rowOffset += numRowsInMatrix
    }

    return result
}

func splitConcatenatedArrays<T>(_ concatenatedArray: [[T]], numberOfRows N: Int) -> (inputGate: [[T]], outputGate: [[T]], forgetGate: [[T]], updateGate: [[T]])? {
    let totalElements = concatenatedArray.count
    guard totalElements % N == 0 else {
        print("Could not split arrays: splitConcatenatedArrays()")
        return nil
    }
    let numberOfArrays = totalElements / N
    var splitArrays = [[[T]]]()
    
    for i in 0..<numberOfArrays {
        let startIndex = i * N
        let endIndex = startIndex + N
        let subArray = Array(concatenatedArray[startIndex..<endIndex])
        splitArrays.append(subArray)
    }
    
    return (splitArrays[0], splitArrays[1], splitArrays[2], splitArrays[3])
}


// Optimizations

func dropoutMask(x: Int, y: Int, _ dropoutProb: Double) -> [[Double]] {
    var mask: [[Double]] = []
    for _ in 0..<x {
        var output: [Double] = []
        for _ in 0..<y {
            output.append(Double.random(in: 0.0..<1.0) < dropoutProb ? 1.0 : 0.0)
        }
        mask.append(output)
    }
    return mask
}
func l2Norm(_ matrix: [[Double]]) -> Double {
    let flattenedVector = matrix.flatMap { $0 }
    var sumOfSquares: Double = 0.0
    for element in flattenedVector {
        sumOfSquares += element * element
    }
    return sqrt(sumOfSquares)
}

func largeMeanSquaredError(predictions: [[Double]], targets: [[Double]]) -> Double? {
    // Check if predictions and targets have the same shape
    guard predictions.count == targets.count else {
        return nil // Return nil if lengths don't match
    }
    
    // Calculate the total number of elements
    let totalElements = predictions.count * predictions[0].count
    
    // Calculate the sum of squared differences
    var sumSquaredDifferences: Double = 0.0
    for i in 0..<predictions.count {
        for j in 0..<predictions[i].count {
            sumSquaredDifferences += pow(predictions[i][j] - targets[i][j], 2)
        }
    }
    
    // Calculate the mean squared error
    let mse = sumSquaredDifferences / Double(totalElements)
    
    return mse
}

// Processing

func transpose<T>(_ matrix: [[T]]) -> [[T]] {
    guard let firstRow = matrix.first, !firstRow.isEmpty else {
        return [[T]]()
    }
    
    let rowCount = firstRow.count
    var transposedMatrix = [[T]](repeating: [T](), count: rowCount)
    
    for row in matrix {
        for (index, element) in row.enumerated() {
            transposedMatrix[index].append(element)
        }
    }
    
    return transposedMatrix
}

func scaleMatrix(_ x: [[Double]], by: Double) -> [[Double]] {
    return x.map{ $0.map{ $0 * by } }
}

func convertToSwiftArray(_ buffer: MTLBuffer, rows: Int, columns: Int) -> [[Double]] {
    // Access the contents of the Metal buffer as a pointer to Float
    let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
    
    // Calculate the number of elements in the buffer
    let elementCount = rows * columns
    
    // Create a buffer pointer from the raw pointer
    let bufferPointer = UnsafeBufferPointer(start: pointer, count: elementCount)
    
    // Convert the buffer pointer to an array of Float
    let floatArray = Array(bufferPointer)
    
    // Convert the array of Float to a 2D array of Double
    let doubleArray = floatArray.map { Double($0) }
    let result = stride(from: 0, to: doubleArray.count, by: columns).map { Array(doubleArray[$0..<$0+columns]) }
    
    return result
}

func softmax(_ x: [Double]) -> [Double] {
    let exps = x.map { exp($0) }
    let sumExp = exps.reduce(0, +)
    return exps.map { $0 / sumExp }
}
