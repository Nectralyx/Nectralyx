//
//  File.swift
//  
//
//  Created by Morgan Keay on 2024-04-16.
//

import Foundation
import Metal

func MTLAddition(_ x: [[Double]], _ y: [[Double]]) -> [[Double]] {
    guard x.count == y.count && x[0].count == y[0].count else {
        print("Could not add matrices: MTLAddition()")
        return []
    }
    let a: [[Float]] = x.map{ $0.map{ Float($0) } }
    let b: [[Float]] = y.map{ $0.map{ Float($0) } }
    
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let library = device.makeDefaultLibrary()!
    let function = library.makeFunction(name: "add_arrays")!
    let state = try! device.makeComputePipelineState(function: function)
    let buffer = queue.makeCommandBuffer()!
    let xBuff = device.makeBuffer(bytes: a.flatMap{ $0 }, length: MemoryLayout<Float>.stride * a.count * a[0].count)!
    let yBuff = device.makeBuffer(bytes: b.flatMap{ $0 }, length: MemoryLayout<Float>.stride * a.count * a[0].count)!
    let resultBuff = device.makeBuffer(length: MemoryLayout<Float>.stride * a.count * a[0].count)!
    
    let rows = UInt32(a.count)
    let cols = UInt32(a[0].count)
    let commander = buffer.makeComputeCommandEncoder()!
    commander.setComputePipelineState(state)
    commander.setBuffer(xBuff, offset: 0, index: 0)
    commander.setBuffer(yBuff, offset: 0, index: 1)
    commander.setBuffer(resultBuff, offset: 0, index: 2)
    //commander.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 3)
    //commander.setBytes(&cols, length: MemoryLayout<UInt32>.size, index: 4)
    
    let exec = state.threadExecutionWidth
    let groups = MTLSize(width: 1, height: a[0].count * a.count, depth: 1)
    
    commander.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: exec, height: exec, depth: 1))
    commander.endEncoding()
    buffer.commit()
    buffer.waitUntilCompleted()
    
    let output = convertToSwiftArray(resultBuff, rows: Int(rows), columns: Int(cols))
    return output
}

func MTLMatMul(_ x: [[Double]], _ y: [[Double]]) -> [[Double]] {
    let a: [[Float]] = x.map{ $0.map{ Float($0) } }
    let b: [[Float]] = y.map{ $0.map{ Float($0) } }
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let library = device.makeDefaultLibrary()!
    let function = library.makeFunction(name: "matrixMultiply")!
    let state = try! device.makeComputePipelineState(function: function)
    let buffer = queue.makeCommandBuffer()!
    var rowsX = UInt32(a.count)
    var colsX = UInt32(a[0].count)
    var colsY = UInt32(b[0].count)
    
    let xBuff = device.makeBuffer(bytes: a.flatMap{ $0 }, length: MemoryLayout<Float>.stride * a.count * a[0].count)!
    let yBuff = device.makeBuffer(bytes: b.flatMap{ $0 }, length: MemoryLayout<Float>.stride * b.count * b[0].count)
    let resultBuff = device.makeBuffer(length: MemoryLayout<Float>.stride * a.count * b[0].count)
    
    let commander = buffer.makeComputeCommandEncoder()!
    commander.setComputePipelineState(state)
    commander.setBuffer(xBuff, offset: 0, index: 0)
    commander.setBuffer(yBuff, offset: 0, index: 1)
    commander.setBuffer(resultBuff, offset: 0, index: 2)
    commander.setBytes(&rowsX, length: MemoryLayout<UInt32>.size, index: 3)
    commander.setBytes(&colsX, length: MemoryLayout<UInt32>.size, index: 4)
    commander.setBytes(&colsY, length: MemoryLayout<UInt32>.size, index: 5)
    
    let execWidth = state.threadExecutionWidth
    let groups = MTLSize(width: (Int(rowsX) + execWidth - 1) / execWidth, height: (Int(colsY) + execWidth - 1) / execWidth, depth: 1)
    
    commander.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: execWidth, height: execWidth, depth: 1))
    commander.endEncoding()
    buffer.commit()
    buffer.waitUntilCompleted()

    let output = convertToSwiftArray(resultBuff!, rows: Int(rowsX), columns: Int(colsY))
    return output
}

func MTLConcatenate(_ matrices: [[[Double]]]) -> [[Double]] {
    let a = matrices.map{ $0.map{ $0.map{ Float($0) } } }
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let library = device.makeDefaultLibrary()!
    let function = library.makeFunction(name: "concatenateArrays")!
    let state = try! device.makeComputePipelineState(function: function)
    let buffer = queue.makeCommandBuffer()!
    let numRows = UInt32(a[0].count)
    let numCols = UInt32(a[0][0].count)
    var matSize = UInt32(Int(numCols) * Int(numRows))
    let ar1Buff = device.makeBuffer(bytes: a[0].flatMap{ $0 }, length: MemoryLayout<Float>.stride * a[0].count * a[0][0].count)!
    let ar2Buff = device.makeBuffer(bytes: a[1].flatMap{ $0 }, length: MemoryLayout<Float>.stride * a[0].count * a[0][0].count)!
    let ar3Buff = device.makeBuffer(bytes: a[2].flatMap{ $0 }, length: MemoryLayout<Float>.stride * a[0].count * a[0][0].count)!
    let ar4Buff = device.makeBuffer(bytes: a[3].flatMap{ $0 }, length: MemoryLayout<Float>.stride * a[0].count * a[0][0].count)!
    let resultBuff = device.makeBuffer(length: MemoryLayout<Float>.stride * a.count * a[0].count * a[0][0].count)!

    let commander = buffer.makeComputeCommandEncoder()!
    commander.setComputePipelineState(state)
    commander.setBuffer(ar1Buff, offset: 0, index: 0)
    commander.setBuffer(ar2Buff, offset: 0, index: 1)
    commander.setBuffer(ar3Buff, offset: 0, index: 2)
    commander.setBuffer(ar4Buff, offset: 0, index: 3)
    commander.setBuffer(resultBuff, offset: 0, index: 4)
    commander.setBytes(&matSize, length: MemoryLayout<UInt32>.size, index: 5)
    
    let execWidth = state.threadExecutionWidth
    let groups = MTLSize(width: Int(matSize) * 4, height: 1, depth: 1)
    
    commander.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: execWidth, height: 1, depth: 1))
    commander.endEncoding()
    buffer.commit()
    buffer.waitUntilCompleted()

    let resultMatrixPointer = resultBuff.contents().bindMemory(to: Float.self, capacity: a.count * a[0].count * a[0][0].count)
    
    func convertToSwiftArrayConcatenate(_ buffer: MTLBuffer, rows: Int, columns: Int) -> [[Double]] {
        // Access the contents of the Metal buffer as a pointer to Float
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        
        // Calculate the number of elements in the buffer
        let elementCount = rows * columns * 4
        
        // Create a buffer pointer from the raw pointer
        let bufferPointer = UnsafeBufferPointer(start: pointer, count: elementCount)
        
        // Convert the buffer pointer to an array of Float
        let floatArray = Array(bufferPointer)
        
        // Convert the array of Float to a 2D array of Double
        let doubleArray = floatArray.map { Double($0) }
        let result = stride(from: 0, to: doubleArray.count, by: columns).map { Array(doubleArray[$0..<$0+columns]) }
        
        return result
    }
    return convertToSwiftArrayConcatenate(resultBuff, rows: Int(numRows), columns: Int(numCols))
}

func MTLSplitArrays(_ matrix: [[Double]]) -> (inputGate: [[Double]], outputGate: [[Double]], forgetGate: [[Double]], updateGate: [[Double]]) {
    let a = matrix.map{ $0.map{ Float($0) } }
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let library = device.makeDefaultLibrary()!
    let function = library.makeFunction(name: "splitArray")!
    let state = try! device.makeComputePipelineState(function: function)
    let buffer = queue.makeCommandBuffer()!
    
    let numRows = UInt32(a.count)
    let numCols = UInt32(a[0].count)
    var matSize = UInt32(Int(numCols) * Int(numRows))
    let ar1Buff = device.makeBuffer(length: MemoryLayout<Float>.stride * a.count * a[0].count / 4)!
    let ar2Buff = device.makeBuffer(length: MemoryLayout<Float>.stride * a.count * a[0].count / 4)!
    let ar3Buff = device.makeBuffer(length: MemoryLayout<Float>.stride * a.count * a[0].count / 4)!
    let ar4Buff = device.makeBuffer(length: MemoryLayout<Float>.stride * a.count * a[0].count / 4)!
    
    let inputBuff = device.makeBuffer(bytes: a.flatMap{ $0 }, length: MemoryLayout<Float>.stride * a.count * a[0].count)!

    let commander = buffer.makeComputeCommandEncoder()!
    commander.setComputePipelineState(state)
    commander.setBuffer(ar1Buff, offset: 0, index: 1)
    commander.setBuffer(ar2Buff, offset: 0, index: 2)
    commander.setBuffer(ar3Buff, offset: 0, index: 3)
    commander.setBuffer(ar4Buff, offset: 0, index: 4)
    commander.setBuffer(inputBuff, offset: 0, index: 0)
    commander.setBytes(&matSize, length: MemoryLayout<UInt32>.size, index: 5)
    
    let execWidth = state.threadExecutionWidth
    let groups = MTLSize(width: (Int(numRows) + execWidth - 1) / execWidth, height: (Int(numCols) + execWidth - 1) / execWidth, depth: 1)
    
    commander.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: execWidth, height: execWidth, depth: 1))
    commander.endEncoding()
    buffer.commit()
    buffer.waitUntilCompleted()

    let out1 = convertToSwiftArray(ar1Buff, rows: a.count / 4, columns: a[0].count)
    let out2 = convertToSwiftArray(ar2Buff, rows: a.count / 4, columns: a[0].count)
    let out3 = convertToSwiftArray(ar3Buff, rows: a.count / 4, columns: a[0].count)
    let out4 = convertToSwiftArray(ar4Buff, rows: a.count / 4, columns: a[0].count)
    return (out1, out2, out3, out4)
}

func MTLElementMultiply(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
    guard a.count == b.count && a[0].count == b[0].count else {
        print("Could not multiply arrays: MTLElementMultiply")
        return []
    }
    
    var x = a.map{ $0.map{ Float($0) } }
    var y = b.map{ $0.map{ Float($0) } }
    let device = MTLCreateSystemDefaultDevice()!
    let queue = device.makeCommandQueue()!
    let library = device.makeDefaultLibrary()!
    let function = library.makeFunction(name: "elementMultiply")!
    let state = try! device.makeComputePipelineState(function: function)
    let buffer = queue.makeCommandBuffer()!
    var numRows = UInt32(x.count)
    var numCols = UInt32(x[0].count)
    var matSize = UInt32(Int(numCols) * Int(numRows))
    let ar1Buff = device.makeBuffer(bytes: x.flatMap{ $0 }, length: MemoryLayout<Float>.stride * x.count * x[0].count)!
    let ar2Buff = device.makeBuffer(bytes: y.flatMap{ $0 }, length: MemoryLayout<Float>.stride * x.count * x[0].count)!
    
    let resultBuff = device.makeBuffer(length: MemoryLayout<Float>.stride * x.count * x[0].count)!

    let commander = buffer.makeComputeCommandEncoder()!
    commander.setComputePipelineState(state)
    commander.setBuffer(ar1Buff, offset: 0, index: 0)
    commander.setBuffer(ar2Buff, offset: 0, index: 1)
    commander.setBuffer(resultBuff, offset: 0, index: 2)
    
    let exec = state.threadExecutionWidth
    let groups = MTLSize(width: 1, height: x[0].count * x.count, depth: 1)
    
    commander.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: exec, height: exec, depth: 1))
    commander.endEncoding()
    buffer.commit()
    buffer.waitUntilCompleted()
    
    return convertToSwiftArray(resultBuff, rows: Int(numRows), columns: Int(numCols))
}
