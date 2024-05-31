//
//  File.swift
//  
//
//  Created by Morgan Keay on 2024-04-16.
//

import Foundation

class EmbeddingLayer: Codable {
    let dimensionSize: Int
    let vocabularySize: Int
    var weights: [[Double]]
    var sport: Bool
    init(dimensionSize: Int, vocabularySize: Int, sport: Bool = false) {
        var Iweights: [[Double]] = []
        for _ in 0..<vocabularySize {
            var word: [Double] = []
            for _ in 0..<dimensionSize {
                word.append(Double.random(in: -1...1))
            }
            Iweights.append(word)
        }
        self.weights = Iweights
        self.dimensionSize = dimensionSize
        self.vocabularySize = vocabularySize
        self.sport = sport
    }
    
    func embed(_ words: [[Double]]) -> [[Double]] {
        var output: [[Double]] = []
        for word in 0..<words.count {
            if let index = words[word].firstIndex(where: { $0 != 0 }) {
                output.append(weights[index])
            } else {
                output.append(Array(repeating: -1.0, count: dimensionSize))
            }
        }
        return output
    }
    func getIndices(_ words: [[Double]]) -> [Int] {
        var output: [Int] = []
        for word in 0..<words.count {
            if let index = words[word].firstIndex(where: {$0 != 0 }) {
                output.append(index)
            }
        }
        return output
    }
    func backward(input: [[Double]], target: [[Double]], learningRate: Double, 𝛅_l: [[Double]], dropout: Double = 0, gradientClip: Double = 0) {
        var new = 𝛅_l
        if dropout > 0 {
            let mask = dropoutMask(x: 𝛅_l.count, y: 𝛅_l[0].count, dropout)
            for i in 0..<𝛅_l.count {
                for j in 0..<𝛅_l[0].count {
                    new[i][j] *= mask[i][j]
                }
            }
        }
        var 𝛅W = sport ? MTLMatMul(transpose(input), new) : matrixMultiply(transpose(input), new)!
       // var 𝛅W = matrixMultiply(transpose(input), new)!
        if gradientClip != 0 {
            let clipThreshold = gradientClip
            let L2Norm = l2Norm(𝛅W)
            if L2Norm > clipThreshold {
                let scale = clipThreshold / L2Norm
                𝛅W = scaleMatrix(𝛅W, by: scale)
            }
        }
        weights = sport ? MTLAddition(weights, 𝛅W.map{ $0.map{ -$0 * learningRate } }) : matrixAdd(weights, 𝛅W.map{ $0.map{ -$0 * learningRate } })!
        //weights = matrixAdd(weights, 𝛅W.map{ $0.map{ -$0 * learningRate } })!
    }
}

class DenseLayer: Codable {
    var weights: [[Double]] = []
    var biases: [Double] = []
    var activation: String = ""
    var inputSize: Int
    var hiddenSize: Int
    var sport: Bool
    init(inputSize: Int, hiddenSize: Int, activation: String, initialization: String, sport: Bool = false) {
        self.weights = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        self.biases = Array(repeating: 0.0, count: hiddenSize)
        self.activation = activation
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.sport = sport
        
        if initialization == "he" {
            initializeDenseHe()
        } else {
            initializeDenseXavier()
        }
    }
    
    func forward(input: [[Double]]) -> [[Double]] {
        //let a = dotProduct(input, weights)!
        //let a = MTLMatMul(input, weights) // <- Metal
        let a = sport ? MTLMatMul(input, weights) : matrixMultiply(input, weights)!
        var b: [[Double]] = []
        for row in a {
            b.append(zip(row, biases).map{ $0 + $1 })
        }
        let result = b.map{ $0.map{ activationFunctions[activation]!($0) } }
        return result
    }
    
    func backward(input: [[Double]], prediction: [[Double]], target: [[Double]], learningRate: Double, 𝛅_l: [[Double]] = [], loss: String = "L2", dropout: Double = 0, gradientClip: Double = 0) -> [[Double]] {
        var outputError: [[Double]] = prediction
        if 𝛅_l.isEmpty {
            if loss == "L2" {
                for i in 0..<prediction.count {
                    for j in 0..<prediction[0].count {
                        outputError[i][j] = prediction[i][j] - target[i][j]
                    }
                }
            } else if loss == "SequenceCrossEntropy" {
                for i in 0..<prediction.count {
                    let softMax = softmax(prediction[i])
                    for j in 0..<prediction[0].count {
                        outputError[i][j] = softMax[j] - target[i][j]
                    }
                    
                }
                
            }
        } else {
            outputError = 𝛅_l
        }
        if dropout > 0 {
            let mask = dropoutMask(x: 𝛅_l.isEmpty ? prediction.count : 𝛅_l.count, y: 𝛅_l.isEmpty ? prediction[0].count : 𝛅_l[0].count, dropout)
            let value = 𝛅_l.isEmpty ? prediction : 𝛅_l
            for i in 0..<value.count {
                for j in 0..<value[0].count {
                    outputError[i][j] *= mask[i][j]
                }
            }
        }
        var 𝛅W = sport ? MTLMatMul(transpose(input), outputError) : matrixMultiply(transpose(input), outputError)!
        var 𝛅x = sport ? MTLMatMul(outputError, transpose(weights)) : matrixMultiply(outputError, transpose(weights))!
        
        //var 𝛅W = matrixMultiply(transpose(input), outputError)!
        //let 𝛅x = matrixMultiply(outputError, transpose(weights))!
        //var 𝛅W = MTLMatMul(transpose(input), outputError) // <-Metal
        //var 𝛅x = MTLMatMul(outputError, transpose(weights)) // <-Metal
        var 𝛅b: [Double] = outputError.reduce(into: [Double](repeating: 0.0, count: outputError[0].count)) { result, error in
               for i in 0..<error.count {
                   result[i] += error[i]
               }
           }
        if gradientClip != 0 {
            let clipThreshold = gradientClip
            let L2Norm = l2Norm(𝛅W)
            if L2Norm > clipThreshold {
                let scale = clipThreshold / L2Norm
                𝛅W = scaleMatrix(𝛅W, by: scale)
                𝛅b = 𝛅b.map{ $0 * scale }
            }
        }
        weights = sport ? MTLAddition(weights, 𝛅W.map{ $0.map{ -$0 * learningRate} }) : matrixAdd(weights, 𝛅W.map{ $0.map{ -$0 * learningRate } })!
        //weights = matrixAdd(weights, 𝛅W.map{ $0.map{ -$0 * learningRate } })!
        //weights = MTLAddition(weights, 𝛅W.map{ $0.map{ -$0 * learningRate } }) //<- Metal
        for bias in 0..<biases.count {
            biases[bias] -= (𝛅b[bias] * learningRate)
        }
        
        return 𝛅x
    }
    
    func initializeDenseHe() {
        let heScale = sqrt(2.0 / Double(inputSize))
        for neuron in 0..<inputSize {
            for weight in 0..<hiddenSize {
                weights[neuron][weight] = Double.random(in: -heScale...heScale)
            }
        }
        for weight in 0..<hiddenSize {
            biases[weight] = Double.random(in: -heScale...heScale)
        }
    }

    func initializeDenseXavier() {
        let xavierScale = sqrt(1.0 / Double(inputSize + hiddenSize))
        for neuron in 0..<inputSize {
            for weight in 0..<hiddenSize {
                weights[neuron][weight] = Double.random(in: -xavierScale...xavierScale)
            }
        }
        for weight in 0..<hiddenSize {
            biases[weight] = Double.random(in: -xavierScale...xavierScale)
        }
    }
}

class LSTMLayer: Codable {
    let inputSize: Int
    let hiddenSize: Int
    let sport: Bool
    //Weights v
    var inputWeights: [[Double]] = []
    var outputWeights: [[Double]] = []
    var forgetWeights: [[Double]] = []
    var updateWeights: [[Double]] = []
    //Biases v
    var inputBiases: [Double] = []
    var outputBiases: [Double] = []
    var forgetBiases: [Double] = []
    var updateBiases: [Double] = []
    //States v
    var inputRecurrentStates: [[Double]] = []
    var outputRecurrentStates: [[Double]] = []
    var forgetRecurrentStates: [[Double]] = []
    var updateRecurrentStates: [[Double]] = []
    
    var hiddenStates: [[Double]] = []
    var cellStates: [[Double]] = []
    
    init(inputSize: Int, steps: Int, hiddenSize: Int, activation: String, sport: Bool = false) {
        inputRecurrentStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize)
        outputRecurrentStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize)
        forgetRecurrentStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize)
        updateRecurrentStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize)
        
        hiddenStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: steps)
        cellStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: steps)
        for iinput in 0..<hiddenSize { // inputSize
            var input: [Double] = []
            for hidden in 0..<inputSize { // hiddenSize
                input.append(0.0)
            }
            inputWeights.append(input)
            outputWeights.append(input)
            forgetWeights.append(input)
            updateWeights.append(input)
            
            for inner in 0..<hiddenSize {
                inputRecurrentStates[iinput][inner] = Double.random(in: -0.001...0.001)
                outputRecurrentStates[iinput][inner] = Double.random(in: -0.001...0.001)
                forgetRecurrentStates[iinput][inner] = Double.random(in: -0.001...0.001)
                updateRecurrentStates[iinput][inner] = Double.random(in: -0.001...0.001)
            }
        }
        for hidden in 0..<hiddenSize {
            inputBiases.append(0.0)
            outputBiases.append(0.0)
            forgetBiases.append(0.0)
            updateBiases.append(0.0)
        }
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.sport = sport
        
        if activation == "Xavier" {
            initializeLSTMXavier()
        } else {
            initializeLSTMHe()
        }
    }
    
    func forward(input: [[Double]]) -> ([[Double]], [[Double]]) {
        let recurrentStates = (inputGate: inputRecurrentStates, outputGate: outputRecurrentStates, forgetGate: forgetRecurrentStates, updateGate: updateRecurrentStates)
        
        for step in 0..<input.count {
            let i_t = input[step].map{ [$0] }
            let b_i = inputBiases.map{ [$0] }
            let b_f = forgetBiases.map{ [$0] }
            let b_o = outputBiases.map{ [$0] }
            let b_c = updateBiases.map{ [$0] }
            let h_tl = hiddenStates[step > 0 ? step - 1 : step].map{ [$0] }
            let c_tl = cellStates[step > 0 ? step - 1 : step].map{ [$0] }
            //Forget Gate v
            let a = sport ? MTLMatMul(forgetWeights, i_t) : matrixMultiply(forgetWeights, i_t)!
            let b = sport ? MTLMatMul(recurrentStates.forgetGate, h_tl) : matrixMultiply(recurrentStates.forgetGate, h_tl)!
            let c = sport ? MTLAddition(a, b) : matrixAdd(a, b)!
            let d = sport ? MTLAddition(c, b_f) : matrixAdd(c, b_f)!
            let forgetGate = d.map{ $0.map{ Sigmoid($0) } }
            
            //Input Gate v
            let e = sport ? MTLMatMul(inputWeights, i_t) : matrixMultiply(inputWeights, i_t)!
            let f = sport ? MTLMatMul(recurrentStates.inputGate, h_tl) : matrixMultiply(recurrentStates.inputGate, h_tl)!
            let g = sport ? MTLAddition(e, f) : matrixAdd(e, f)!
            let h = sport ? MTLAddition(g, b_i) : matrixAdd(g, b_i)!
            let inputGate = h.map{ $0.map{ Sigmoid($0) } }
            
            //Output Gate v
            let i = sport ? MTLMatMul(outputWeights, i_t) : matrixMultiply(outputWeights, i_t)!
            let j = sport ? MTLMatMul(recurrentStates.outputGate, h_tl) : matrixMultiply(recurrentStates.outputGate, h_tl)!
            let k = sport ? MTLAddition(i, j) : matrixAdd(i, j)!
            let l = sport ? MTLAddition(k, b_o) : matrixAdd(k, b_o)!
            let outputGate = l.map{ $0.map{ Sigmoid($0) } }
            
            //Update Gate v
            let m = sport ? MTLMatMul(updateWeights, i_t) : matrixMultiply(updateWeights, i_t)!
            let n = sport ? MTLMatMul(recurrentStates.updateGate, h_tl) : matrixMultiply(recurrentStates.updateGate, h_tl)!
            let o = sport ? MTLAddition(m, n) : matrixAdd(m, n)!
            let p = sport ? MTLAddition(o, b_c) : matrixAdd(o, b_c)!
            let updateGate = p.map{ $0.map{ Tanh($0) } }
            
            //Cell State v
            let q = sport ? MTLElementMultiply(forgetGate, c_tl) : elementMultiply(forgetGate, c_tl)!
            let r = sport ? MTLElementMultiply(inputGate, updateGate) : elementMultiply(inputGate, updateGate)!
            let cellState = sport ? MTLAddition(q, r) : matrixAdd(q, r)!
            
            //Hidden State v
            let s = cellState.map{ $0.map{ Tanh($0) } }
            let hiddenState = sport ? MTLElementMultiply(outputGate, s) : elementMultiply(outputGate, s)!

            hiddenStates[step] = hiddenState.flatMap{ $0 }
            cellStates[step] = cellState.flatMap{ $0 }
        }
        return(hiddenStates, cellStates)
    }
    func backward(input: [[Double]], prediction: [[Double]], learningRate: Double, 𝛅_l: [[Double]] = [], target: [[Double]], gradientClip: Double = 0, loss: String = "L2", dropout: Double = 0) -> [[Double]] {
        var outputError: [[Double]] = prediction
        guard prediction.count == target.count && prediction[0].count == target[0].count else {
            print("Mismatched Sizes: Prediction: \(prediction.count)x\(prediction[0].count), Target: \(target.count)x\(target[0].count)")
            return []
        }
        if loss == "L2" {
            for i in 0..<prediction.count {
                for j in 0..<prediction[0].count {
                    outputError[i][j] = prediction[i][j] - target[i][j]
                }
            }
        } else if loss == "SequenceCrossEntropy" {
            for i in 0..<prediction.count {
                let softMax = softmax(prediction[i])
                for j in 0..<prediction[0].count {
                    outputError[i][j] = softMax[j] - target[i][j]
                }
            }
        }
        
        if dropout > 0 {
            let mask = dropoutMask(x: prediction.count, y: prediction[0].count, dropout)
            for i in 0..<prediction.count {
                for j in 0..<prediction[0].count {
                    outputError[i][j] *= mask[i][j]
                }
            }
        }
       // print("Output Error: \(outputError)")

        var 𝛅x: [[Double]] = []
        let (inputGates, outputGates, forgetGates, updateGates, cellStates, advhiddenStates) = advancedForward(input: input)
        var 𝛅W: [[Double]] = Array(repeating: Array(repeating: 0.0, count: inputSize), count: hiddenSize * 4)
        var 𝛅U: [[Double]] = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize * 4)
        var 𝛅b: [[Double]] = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: 4)
        var gates: [[[Double]]] = Array(repeating: Array(repeating: Array(repeating: 0.0, count: 1), count: hiddenSize * 4), count: prediction.count)
        for step in (0..<prediction.count).reversed() {
            let x_t = [input[step]]
            let inputGate = inputGates[step]
            let outputGate = outputGates[step]
            let forgetGate = forgetGates[step]
            let updateGate = updateGates[step]
            let cellState = cellStates[step]
            let 𝛅cell_tl = Array(repeating: 0.0, count: hiddenSize).map{ [$0] }
            let forget_tl = Array(repeating: 0.0, count: hiddenSize).map{ [$0] }
            //let 𝚫_t = outputError[step]
            let 𝚫_t = 𝛅_l.isEmpty ? outputError[step] : 𝛅_l[step]
            var 𝚫out_t = Array(repeating: 0.0, count: hiddenSize)
            let 𝛅out_t = add(𝚫_t, 𝚫out_t).map{ [$0] }

            let aa = sport ? MTLElementMultiply(𝛅out_t, outputGate) : elementMultiply(𝛅out_t, outputGate)!
            let ab = cellState.map{ $0.map{ (1 - Tanh($0)) } }
            let ac = sport ? MTLElementMultiply(𝛅cell_tl, forget_tl) : elementMultiply(𝛅cell_tl, forget_tl)!
            let ad = sport ? MTLElementMultiply(aa, ab) : elementMultiply(aa, ab)!
            let 𝛅state_t = sport ? MTLAddition(ad, ac) : matrixAdd(ad, ac)!
            //Cell State ^
            let ae = updateGate.map{ $0.map{ 1 - $0 * $0 } }
            let af = sport ? MTLElementMultiply(𝛅state_t, inputGate) : elementMultiply(𝛅state_t, inputGate)!
            let 𝛅a_t = sport ? MTLElementMultiply(af, ae) : elementMultiply(af, ae)!
            //print("𝛅a_t: \(𝛅a_t)")
            let ag = inputGate.map{ $0.map{ 1 - $0} }
            let ah = sport ? MTLElementMultiply(𝛅state_t, updateGate) : elementMultiply(𝛅state_t, updateGate)!
            let ai = sport ? MTLElementMultiply(ah, inputGate) : elementMultiply(ah, inputGate)!
            let 𝛅i_t = sport ? MTLElementMultiply(ai, ag) : elementMultiply(ai, ag)!
            //print("𝛅i_t: \(𝛅i_t)")
            let aj = forgetGate.map{ $0.map{ 1 - $0} }
            let ak = sport ? MTLElementMultiply(𝛅state_t, step > 0 ? cellStates[step - 1] : Array(repeating: Array(repeating: 0.0, count: cellState[0].count), count: cellState.count)) : elementMultiply(𝛅state_t, step > 0 ? cellStates[step - 1] : Array(repeating: Array(repeating: 0.0, count: cellState[0].count), count: cellState.count))!
            let al = sport ? MTLElementMultiply(ak, forgetGate) : elementMultiply(ak, forgetGate)!
            let 𝛅f_t = sport ? MTLElementMultiply(al, aj) : elementMultiply(al, aj)!
            //print("𝛅f_t: \(𝛅f_t)")
            let am = cellState.map{ $0.map{ Tanh($0) } }
            let an = outputGate.map{ $0.map{ 1 - $0 } }
            let ao = sport ? MTLElementMultiply(𝛅out_t, am) : elementMultiply(𝛅out_t, am)!
            let ap = sport ? MTLElementMultiply(ao, outputGate) : elementMultiply(ao, outputGate)!
            let 𝛅o_t = sport ? MTLElementMultiply(ap, an) : elementMultiply(ap, an)!
            //print("𝛅o_t: \(𝛅o_t)")
            let W = sport ? MTLConcatenate([inputWeights, outputWeights, forgetWeights, updateWeights]) : concatenateWeightMatrices([inputWeights, outputWeights, forgetWeights, updateWeights])
            let G = sport ? MTLConcatenate([𝛅i_t, 𝛅o_t, 𝛅f_t, 𝛅a_t]) : concatenateWeightMatrices([𝛅i_t, 𝛅o_t, 𝛅f_t, 𝛅a_t])
            let 𝛅x_t = sport ? MTLMatMul(transpose(W), G) : matrixMultiply(transpose(W), G)!
            //print("𝛅x_t: \(𝛅x_t)")
            let U = sport ? MTLConcatenate([inputRecurrentStates, outputRecurrentStates, forgetRecurrentStates, updateRecurrentStates]) : concatenateWeightMatrices([inputRecurrentStates, outputRecurrentStates, forgetRecurrentStates, updateRecurrentStates])
            𝚫out_t = sport ? MTLMatMul(transpose(U), G).flatMap{ $0 } : matrixMultiply(transpose(U), G)!.flatMap{ $0 }
            𝛅x.append(𝛅x_t.flatMap{ $0 })
            gates[step] = G
        }
        for step in 0..<prediction.count {
            let x_t = [input[step]]
            𝛅W = sport ? MTLAddition(𝛅W, MTLMatMul(gates[step], x_t)) : matrixAdd(𝛅W, outerProduct(gates[step], x_t))!
            let deconstructed = sport ? MTLSplitArrays(gates[step]) : splitConcatenatedArrays(gates[step], numberOfRows: hiddenSize)!
            𝛅b = sport ? MTLAddition(𝛅b, [deconstructed.inputGate.flatMap{ $0 }, deconstructed.outputGate.flatMap{ $0 }, deconstructed.forgetGate.flatMap{ $0 }, deconstructed.updateGate.flatMap{ $0 }]) : matrixAdd(𝛅b, [deconstructed.inputGate.flatMap{ $0 }, deconstructed.outputGate.flatMap{ $0 }, deconstructed.forgetGate.flatMap{ $0 }, deconstructed.updateGate.flatMap{ $0 }])!
        }
        for step in 0..<prediction.count - 1 {
            let out_t = advhiddenStates[step].flatMap{ $0 }
            𝛅U = sport ? MTLAddition(𝛅U, MTLMatMul(gates[step + 1], [out_t])) : matrixAdd(𝛅U, outerProduct(gates[step + 1], [out_t]))!
        }
        if gradientClip != 0 {
            let clipThreshold = gradientClip
            let L2Norm = l2Norm(𝛅W)
            if L2Norm > clipThreshold {
                let scale = clipThreshold / L2Norm
                𝛅W = scaleMatrix(𝛅W, by: scale)
                𝛅U = scaleMatrix(𝛅U, by: scale)
                𝛅b = scaleMatrix(𝛅b, by: scale)
            }
        }
        let biasUpdates = sport ? MTLAddition([inputBiases, outputBiases, forgetBiases, updateBiases], 𝛅b.map{ $0.map{ -$0 * learningRate } }) : matrixAdd([inputBiases, outputBiases, forgetBiases, updateBiases], 𝛅b.map{ $0.map{ -$0 * learningRate } })!
        inputBiases = biasUpdates[0]
        outputBiases = biasUpdates[1]
        forgetBiases = biasUpdates[2]
        updateBiases = biasUpdates[3]
        
        let weightUpdates = sport ? MTLSplitArrays(𝛅W) : splitConcatenatedArrays(𝛅W, numberOfRows: hiddenSize)!
        inputWeights = sport ? MTLAddition(inputWeights, weightUpdates.inputGate.map{ $0.map{ -$0 * learningRate } }) : matrixAdd(inputWeights, weightUpdates.inputGate.map{ $0.map{ -$0 * learningRate } })!
        outputWeights = sport ? MTLAddition(outputWeights, weightUpdates.outputGate.map{ $0.map{ -$0 * learningRate } }) : matrixAdd(outputWeights, weightUpdates.outputGate.map{ $0.map{ -$0 * learningRate } })!
        forgetWeights = sport ? MTLAddition(forgetWeights, weightUpdates.forgetGate.map{ $0.map{ -$0 * learningRate } }) : matrixAdd(forgetWeights, weightUpdates.forgetGate.map{ $0.map{ -$0 * learningRate } })!
        updateWeights = sport ? MTLAddition(updateWeights, weightUpdates.updateGate.map{ $0.map{ -$0 * learningRate } }) : matrixAdd(updateWeights, weightUpdates.updateGate.map{ $0.map{ -$0 * learningRate } })!
        
        let recurrentUpdates = sport ? MTLSplitArrays(𝛅U) : splitConcatenatedArrays(𝛅U, numberOfRows: hiddenSize)!
        inputRecurrentStates = sport ? MTLAddition(inputRecurrentStates, recurrentUpdates.inputGate.map{ $0.map{ -$0 * learningRate} }) : matrixAdd(inputRecurrentStates, recurrentUpdates.inputGate.map{ $0.map{ -$0 * learningRate } })!
        outputRecurrentStates = sport ? MTLAddition(outputRecurrentStates, recurrentUpdates.outputGate.map{ $0.map{ -$0 * learningRate} }) : matrixAdd(outputRecurrentStates, recurrentUpdates.outputGate.map{ $0.map{ -$0 * learningRate } })!
        forgetRecurrentStates = sport ? MTLAddition(forgetRecurrentStates, recurrentUpdates.forgetGate.map{ $0.map{ -$0 * learningRate} }) : matrixAdd(forgetRecurrentStates, recurrentUpdates.forgetGate.map{ $0.map{ -$0 * learningRate } })!
        updateRecurrentStates = sport ? MTLAddition(updateRecurrentStates, recurrentUpdates.updateGate.map{ $0.map{ -$0 * learningRate} }) : matrixAdd(updateRecurrentStates, recurrentUpdates.forgetGate.map{ $0.map{ -$0 * learningRate } })!
        return 𝛅x
    }
    
    func advancedForward(input: [[Double]]) -> (inputGates: [[[Double]]], outputGates: [[[Double]]], forgetGates: [[[Double]]], updateGates: [[[Double]]], cellStates: [[[Double]]], hiddenStates: [[[Double]]]) {
        var newinputGates: [[[Double]]] = []
        var newoutputGates: [[[Double]]] = []
        var newforgetGates: [[[Double]]] = []
        var newupdateGates: [[[Double]]] = []
        var newcellStates: [[[Double]]] = []
        var newhiddenStates: [[[Double]]] = []
        let recurrentStates = (inputGate: inputRecurrentStates, outputGate: outputRecurrentStates, forgetGate: forgetRecurrentStates, updateGate: updateRecurrentStates)
        
        for step in 0..<input.count {
            let i_t = input[step].map{ [$0] }
            let b_i = inputBiases.map{ [$0] }
            let b_f = forgetBiases.map{ [$0] }
            let b_o = outputBiases.map{ [$0] }
            let b_c = updateBiases.map{ [$0] }
            let h_tl = hiddenStates[step > 0 ? step - 1 : step].map{ [$0] }
            let c_tl = cellStates[step > 0 ? step - 1 : step].map{ [$0] }
            //Forget Gate v
            let a = sport ? MTLMatMul(forgetWeights, i_t) : matrixMultiply(forgetWeights, i_t)!
            let b = sport ? MTLMatMul(recurrentStates.forgetGate, h_tl) : matrixMultiply(recurrentStates.forgetGate, h_tl)!
            let c = sport ? MTLAddition(a, b) : matrixAdd(a, b)!
            let d = sport ? MTLAddition(c, b_f) : matrixAdd(c, b_f)!
            let forgetGate = d.map{ $0.map{ Sigmoid($0) } }
            
            //Input Gate v
            let e = sport ? MTLMatMul(inputWeights, i_t) : matrixMultiply(inputWeights, i_t)!
            let f = sport ? MTLMatMul(recurrentStates.inputGate, h_tl) : matrixMultiply(recurrentStates.inputGate, h_tl)!
            let g = sport ? MTLAddition(e, f) : matrixAdd(e, f)!
            let h = sport ? MTLAddition(g, b_i) : matrixAdd(g, b_i)!
            let inputGate = h.map{ $0.map{ Sigmoid($0) } }
            
            //Output Gate v
            let i = sport ? MTLMatMul(outputWeights, i_t) : matrixMultiply(outputWeights, i_t)!
            let j = sport ? MTLMatMul(recurrentStates.outputGate, h_tl) : matrixMultiply(recurrentStates.outputGate, h_tl)!
            let k = sport ? MTLAddition(i, j) : matrixAdd(i, j)!
            let l = sport ? MTLAddition(k, b_o) : matrixAdd(k, b_o)!
            let outputGate = l.map{ $0.map{ Sigmoid($0) } }
            
            //Update Gate v
            let m = sport ? MTLMatMul(updateWeights, i_t) : matrixMultiply(updateWeights, i_t)!
            let n = sport ? MTLMatMul(recurrentStates.updateGate, h_tl) : matrixMultiply(recurrentStates.updateGate, h_tl)!
            let o = sport ? MTLAddition(m, n) : matrixAdd(m, n)!
            let p = sport ? MTLAddition(o, b_c) : matrixAdd(o, b_c)!
            let updateGate = p.map{ $0.map{ Tanh($0) } }
            
            //Cell State v
            let q = sport ? MTLElementMultiply(forgetGate, c_tl) : elementMultiply(forgetGate, c_tl)!
            let r = sport ? MTLElementMultiply(inputGate, updateGate) : elementMultiply(inputGate, updateGate)!
            let cellState = sport ? MTLAddition(q, r) : matrixAdd(q, r)!
            
            //Hidden State v
            let s = cellState.map{ $0.map{ Tanh($0) } }
            let hiddenState = sport ? MTLElementMultiply(outputGate, s) : elementMultiply(outputGate, s)!

            hiddenStates[step] = hiddenState.flatMap{ $0 }
            cellStates[step] = cellState.flatMap{ $0 }
            newcellStates.append(cellState)
            newhiddenStates.append(hiddenState)
            newinputGates.append(inputGate)
            newoutputGates.append(outputGate)
            newforgetGates.append(forgetGate)
            newupdateGates.append(updateGate)
        }
        return (newinputGates, newoutputGates, newforgetGates, newupdateGates, newcellStates, newhiddenStates)
    }
    /*
    func backward(outputError: [[Double]], sigmoidDelta: [[Double]], tanhDelta: [[Double]], hiddenStates: [[Double]], cellStates: [[Double]], gates: [(inputGate: [[Double]], outputGate: [[Double]], forgetGate: [[Double]], updateGate: [[Double]])], input: [[Double]], learningRate: Double) -> ([[Double]], [[Double]], [[Double]], [[Double]]) {
        var inputWeightGradients: [[Double]] = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        var outputWeightGradients: [[Double]] = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        var forgetWeightGradients: [[Double]] = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        var updateWeightGradients: [[Double]] = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        
        
        var inputBiasGradients: [Double] = Array(repeating: 0.0, count: hiddenSize)
        var outputBiasGradients: [Double] = Array(repeating: 0.0, count: hiddenSize)
        var forgetBiasGradients: [Double] = Array(repeating: 0.0, count: hiddenSize)
        var updateBiasGradients: [Double] = Array(repeating: 0.0, count: hiddenSize)
        
        var prevLayerDelta: [Double] = Array(repeating: 0.0, count: hiddenSize)
        var prevCellDelta: [Double] = Array(repeating: 0.0, count: hiddenSize)
        
        var outputHiddenStates: [[Double]] = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        var outputCellStates: [[Double]] = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: inputSize)
        
        for timeStep in (0..<hiddenStates.count).reversed() {
            let x = input[timeStep]
            let hiddenState = hiddenStates[timeStep]
            let cellState = cellStates[timeStep]
            
            let outputGateDerivative = sigmoidDelta[timeStep]
            
            let cellStateDerivative = tanhDelta[timeStep]
            
            // Getting Gates
            //let (inputGate, outputGate, forgetGate, updateGate) = gates[timeStep]
            
            var inputGates: [[Double]] = []
            var outputGates: [[Double]] = []
            var forgetGates: [[Double]] = []
            var updateGates: [[Double]] = []
            for unitIndex in 0..<hiddenSize {

                let inputGate = sigmoidMap(add(dotProductBroadcast(inputWeights, hiddenStates[unitIndex]), dotProductFlat(x, Array(repeating: inputBiases[unitIndex], count: inputSize))))

                let forgetGate = sigmoidMap(add(dotProductBroadcast(forgetWeights, hiddenStates[unitIndex]), dotProductFlat(x, Array(repeating: forgetBiases[unitIndex], count: inputSize))))

                let outputGate = sigmoidMap(add(dotProductBroadcast(outputWeights, hiddenStates[unitIndex]), dotProductFlat(x, Array(repeating: outputBiases[unitIndex], count: inputSize))))

                let updateGate = tanhMap(add(dotProductBroadcast(updateWeights, hiddenStates[unitIndex]), dotProductFlat(x, Array(repeating: updateBiases[unitIndex], count: inputSize))))

                inputGates.append(inputGate)
                outputGates.append(outputGate)
                forgetGates.append(forgetGate)
                updateGates.append(updateGate)
            }
            
            
            // Compute derivatives of gates
            let broadcastedPrevCellDelta = Array(repeating: prevCellDelta, count: cellState.count / prevCellDelta.count).flatMap{ $0 } + Array(repeating: prevCellDelta.last!, count: cellState.count % prevCellDelta.count)
            let inputGateDerivative = unevenAdd(multiply(broadcastedPrevCellDelta, cellState), unevenMultiply(outputGateDerivative, tanhMap(cellState)))
            
            let forgetGateDerivative = multiply(broadcastedPrevCellDelta, cellState)
            let updateGateDerivative = unevenMultiply(outputGateDerivative, tanhMap(cellState))
            // Update Gradients for Biases
            for unitIndex in 0..<hiddenSize {
                inputBiasGradients[unitIndex] += inputGateDerivative[unitIndex] * SigmoidDerivative(inputGates[unitIndex][timeStep]) //<unitIndex/TimeStep vvv>
                outputBiasGradients[unitIndex] += outputGateDerivative[unitIndex] * SigmoidDerivative(outputGates[unitIndex][timeStep])
                forgetBiasGradients[unitIndex] += forgetGateDerivative[unitIndex] * SigmoidDerivative(forgetGates[unitIndex][timeStep])
                updateBiasGradients[unitIndex] += updateGateDerivative[unitIndex] * TanhDerivative(updateGates[unitIndex][timeStep])
            }
            // Update Gradients for Weights
            for unitIndex in 0..<hiddenSize {
                for inputIndex in 0..<inputSize {
                    inputWeightGradients[inputIndex][unitIndex] += inputGateDerivative[unitIndex] * SigmoidDerivative(inputGates[unitIndex][timeStep]) * input[timeStep][inputIndex]
                    outputWeightGradients[inputIndex][unitIndex] += outputGateDerivative[unitIndex] * SigmoidDerivative(outputGates[unitIndex][timeStep]) * input[timeStep][inputIndex]
                    forgetWeightGradients[inputIndex][unitIndex] += forgetGateDerivative[unitIndex] * SigmoidDerivative(forgetGates[unitIndex][timeStep]) * input[timeStep][inputIndex]
                    updateWeightGradients[inputIndex][unitIndex] += updateGateDerivative[unitIndex] * TanhDerivative(updateGates[unitIndex][timeStep]) * input[timeStep][inputIndex]
                }
            }
            // Compute deltas for the next timestep
            let inputDelta = dotProductReverse(broadcastedPrevCellDelta, inputWeights)
            let outputDelta = dotProductReverse(broadcastedPrevCellDelta, outputWeights)
            let forgetDelta = dotProductReverse(broadcastedPrevCellDelta, forgetWeights)
            let updateDelta = dotProductReverse(broadcastedPrevCellDelta, updateWeights)
            
            // Update the previous delta for the next time step
            prevCellDelta = add(
                add(
                unevenMultiply(inputDelta, inputGateDerivative),
                unevenMultiply(outputDelta, outputGateDerivative)
                ),
                add(
                unevenMultiply(forgetDelta, forgetGateDerivative),
                unevenMultiply(updateDelta, updateGateDerivative)
                )
            )
            prevLayerDelta = prevCellDelta
            
            outputHiddenStates[timeStep] = prevLayerDelta
            outputCellStates[timeStep] = prevCellDelta
        }
        
        let batchSize = Double(hiddenStates.count)
        for unitIndex in 0..<hiddenSize {
            inputBiases[unitIndex] += inputBiasGradients[unitIndex] / batchSize * learningRate
            outputBiases[unitIndex] += outputBiasGradients[unitIndex] / batchSize * learningRate
            forgetBiases[unitIndex] += forgetBiasGradients[unitIndex] / batchSize * learningRate
            updateBiases[unitIndex] += updateBiasGradients[unitIndex] / batchSize * learningRate
            
            for inputIndex in 0..<inputSize {
                inputWeights[inputIndex][unitIndex] += inputWeightGradients[inputIndex][unitIndex] / batchSize * learningRate
                outputWeights[inputIndex][unitIndex] += outputWeightGradients[inputIndex][unitIndex] / batchSize * learningRate
                forgetWeights[inputIndex][unitIndex] += forgetWeightGradients[inputIndex][unitIndex] / batchSize * learningRate
                updateWeights[inputIndex][unitIndex] += updateWeightGradients[inputIndex][unitIndex] / batchSize * learningRate
            }
        }
        
        return (inputWeightGradients, outputWeightGradients, outputHiddenStates, outputCellStates)
    } */
    func forgetStates() {
        inputRecurrentStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize)
        outputRecurrentStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize)
        forgetRecurrentStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize)
        updateRecurrentStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenSize)
        
        hiddenStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: hiddenStates.count)
        cellStates = Array(repeating: Array(repeating: 0.0, count: hiddenSize), count: cellStates.count)
    }
    func initializeLSTMHe() {
        let heScale = sqrt(2.0 / Double(inputSize))
        for input in 0..<inputSize {
            for hidden in 0..<hiddenSize {
                inputWeights[input][hidden] = Double.random(in: -heScale...heScale)
                outputWeights[input][hidden] = Double.random(in: -heScale...heScale)
                forgetWeights[input][hidden] = Double.random(in: -heScale...heScale)
                updateWeights[input][hidden] = Double.random(in: -heScale...heScale)
            }
        }
        for hidden in 0..<hiddenSize {
            inputBiases[hidden] = Double.random(in: -heScale...heScale)
            outputBiases[hidden] = Double.random(in: -heScale...heScale)
            forgetBiases[hidden] = Double.random(in: -heScale...heScale)
            updateBiases[hidden] = Double.random(in: -heScale...heScale)
        }
    }
    func initializeLSTMXavier() {
        let xavierScale = sqrt(1.0 / Double(inputSize + hiddenSize))
        for input in 0..<hiddenSize { // inputSize
            for hidden in 0..<inputSize { // hiddenSize
                inputWeights[input][hidden] = Double.random(in: -xavierScale...xavierScale)
                outputWeights[input][hidden] = Double.random(in: -xavierScale...xavierScale)
                forgetWeights[input][hidden] = Double.random(in: -xavierScale...xavierScale)
                updateWeights[input][hidden] = Double.random(in: -xavierScale...xavierScale)
            }
        }
        for hidden in 0..<hiddenSize {
            inputBiases[hidden] = Double.random(in: -xavierScale...xavierScale)
            outputBiases[hidden] = Double.random(in: -xavierScale...xavierScale)
            forgetBiases[hidden] = Double.random(in: -xavierScale...xavierScale)
            updateBiases[hidden] = Double.random(in: -xavierScale...xavierScale)
        }
    }
}
