import XCTest
@testable import Converge

final class ConvergeTests: XCTestCase {
    func testExample() async throws {
        // XCTest Documentation
        // https://developer.apple.com/documentation/xctest

        // Defining Test Cases and Test Methods
        // https://developer.apple.com/documentation/xctest/defining_test_cases_and_test_methods
        
        let dense1 = DenseLayer(inputSize: 50, hiddenSize: 50, activation: "Linear", initialization: "Xavier", sport: false)
        let dense2 = DenseLayer(inputSize: 50, hiddenSize: 50, activation: "Linear", initialization: "Xavier", sport: false)
        let dense3 = DenseLayer(inputSize: 50, hiddenSize: 1, activation: "Linear", initialization: "Xavier", sport: false)
        let lstm = LSTMLayer(inputSize: 1, steps: 5, hiddenSize: 50, activation: "Xavier", sport: false)
        let model = Luminal(LSTM: lstm, DenseLayers: [dense1, dense2, dense3], learningRate: 0.001)
        let inputs = [
            [[0.0], [1], [2], [3], [4]],
            [[1], [2], [3], [4], [5]],
            [[2], [3], [4], [5], [6]]
        ]
        let targets = [
            [[1.0], [2], [3], [4], [5]],
            [[2], [3], [4], [5], [6]],
            [[3], [4], [5], [6], [7]]
        ]
        
        let newInputs = [
            [[], [], []],
            [[], [], []],
            [[], [], []]
        ]
        let newtargets = [
            [[], [], []],
            [[], [], []],
            [[], [], []]
        ]
        
        await model.train(input: inputs, targets: targets, printTrain: true, gradientClipping: 0.001)
    }
    
}


