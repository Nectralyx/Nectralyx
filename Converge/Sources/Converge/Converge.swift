// The Swift Programming Language
// https://docs.swift.org/swift-book

@available(macOS 10.15.0, *)
public class Luminal: Codable {
    var LSTM: LSTMLayer
    var DenseLayers: [DenseLayer]
    var learningRate: Double
    init(LSTM: LSTMLayer, DenseLayers: [DenseLayer], learningRate: Double) {
        self.LSTM = LSTM
        self.DenseLayers = DenseLayers
        self.learningRate = learningRate
        var compatible = true
        if DenseLayers.isEmpty {
            compatible = false
        }
        if LSTM.hiddenSize != DenseLayers.first?.inputSize {
            compatible = false
        }
        for layer in 0..<DenseLayers.count - 1 {
            let currentLayer = DenseLayers[layer]
            let nextLayer = DenseLayers[layer + 1]
            
            if currentLayer.hiddenSize != nextLayer.inputSize {
                compatible = false
            }
        }
        guard compatible else {
            fatalError("Neural Network could not be initialized because layers do not line up: Luminal")
        }
        print("""

            ///||###########///
           ///|||##########///
          ///#|||#########///
         ///##|||########///
        ///###|||#######///
       ///####|||######///
      ///#####|||#####///
     ///######|||####///
    ///#######|||###///
   ///########|||##///
  ///#########|||#///
 ///##########|||///
///###########||///


""")
    }
    
    func forward(input: [[Double]], prescape: Bool = false) -> [[[Double]]] {
        let first = LSTM.forward(input: input).0
        var final: [[[Double]]] = []
        var allOutputs: [[[Double]]] = []
        var nextInput: [[Double]] = []
        if DenseLayers.count > 1 {
            for dense in 0..<DenseLayers.count {
                if dense == 0 {
                    allOutputs.append(DenseLayers[dense].forward(input: first))
                    nextInput = DenseLayers[dense].forward(input: first)
                } else if dense == DenseLayers.count - 1 {
                    final.append(DenseLayers[dense].forward(input: nextInput))
                    allOutputs.append(DenseLayers[dense].forward(input: nextInput))
                } else {
                    allOutputs.append(DenseLayers[dense].forward(input: nextInput))
                    nextInput = DenseLayers[dense].forward(input: nextInput)
                }
            }
        } else {
            nextInput = first
            final.append(DenseLayers[0].forward(input: nextInput))
            allOutputs.append(DenseLayers[0].forward(input: nextInput))
        }
        return prescape ? allOutputs : final
    }
    func backward(input: [[Double]], target: [[Double]], gradientClipping: Double = 0) {
        let prediction = forward(input: input)[0]
        guard target.count == prediction.count && target[0].count == prediction[0].count else {
            print("Targets are not the same size as predictions: Luminal.backward()")
            return
        }
        let firstDenseInput = LSTM.forward(input: input).0
        let firstSet = forward(input: input, prescape: true)
        var inputGradient: [[Double]] = []
        if DenseLayers.count > 1 {
             inputGradient = DenseLayers.last!.backward(input: firstSet.dropLast().last!, prediction: prediction, target: target, learningRate: learningRate)
            
            for dense in (0..<DenseLayers.count - 1).reversed() {
                inputGradient = DenseLayers[dense].backward(input: dense == 0 ? firstDenseInput : firstSet[dense - 1], prediction: prediction, target: target, learningRate: learningRate, 𝛅_l: inputGradient)
            }
        } else {
            inputGradient = DenseLayers.last!.backward(input: firstDenseInput, prediction: prediction, target: target, learningRate: learningRate)
        }
        let _ = LSTM.backward(input: input, prediction: prediction, learningRate: learningRate, 𝛅_l: inputGradient, target: target, gradientClip: gradientClipping)
    }
    @available(iOS 13.0.0, *)
    func train(
        input: [[[Double]]],
        targets: [[[Double]]],
        validationInputs: [[[Double]]] = [],
        validationTargets: [[[Double]]] = [],
        epochs: Int = 0,
        printTrain: Bool = false,
        printValidation: Bool = true,
        learnValidation: Bool = false,
        lrOptimizer: Bool = true,
        gradientClipping: Double = 0
        ) async {
        guard input.count == targets.count && validationInputs.count == validationTargets.count else {
            print("Training sequence could not be run because dataset is not correct: Luminal.train()")
            return
        }
        var initialEpochs = epochs
        if epochs == 0 {
            initialEpochs = 1000
        }
            
            for epoch in 0...initialEpochs {
            var additional = ""
            var totalLoss = 0.0
            var validationLoss = 0.0
            var validationOutputs: [[[Double]]] = []
            var trainingOutputs: [[[Double]]] = []
            for (input, target) in zip(input, targets) {
                let output = forward(input: input)[0]
                backward(input: input, target: target, gradientClipping: gradientClipping)
                totalLoss += largeMeanSquaredError(predictions: output, targets: target)!
                trainingOutputs.append(output)
            }
            if !validationInputs.isEmpty {
                for (input, target) in zip(validationInputs, validationTargets) {
                    let output = forward(input: input)[0]
                    if learnValidation {
                        backward(input: input, target: target, gradientClipping: gradientClipping)
                    }
                    validationOutputs.append(output)
                    totalLoss += largeMeanSquaredError(predictions: output, targets: target)!
                    validationLoss += largeMeanSquaredError(predictions: output, targets: target)!
                }
            }
                if lrOptimizer {
                    //learningRate = cosineAnnealingLearningRate(currentEpoch: epoch, totalEpochs: initialEpochs, initialLR: learningRate)
                    learningRate -= learningRate / Double(initialEpochs)
                }
            if epoch == initialEpochs - 1 {
                if epochs == 0 {
                    if validationInputs.isEmpty {
                        if totalLoss < 0.005 {
                            break
                        } else {
                            initialEpochs += 1000
                            additional = "Adding more epochs to improve training \n"
                        }
                    } else {
                        if validationLoss < 0.005 {
                            break
                        } else {
                            initialEpochs += 1000
                            additional = "Adding more epochs to improve training \n"
                        }
                    }
                }
            }
            if printTrain {
                additional.append("Training Outputs: \(trainingOutputs) \n")
            }
            if printValidation {
                additional.append("Validation Outputs: \(validationOutputs) \n")
            }
            print("""
 _________________________________________________________________________________________________________
/ Epoch \(epoch) / \(initialEpochs)
|---------------------------------
|Total Loss: \(totalLoss) | Validation Loss: \(validationLoss)
|
|Outputs:
\(additional)
""")
        }
    }
}
