// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import "opengradient-neuroml/src/OGInference.sol";

contract Test {

    string public resultString;
    TensorLib.Number public resultNumber;

    function run() public {
        string memory modelId = "QmbbzDwqSxZSgkz1EbsNHp2mb67rYeUYHYWJ4wECE24S7A";

        ModelInput memory modelInput = ModelInput(
            new TensorLib.MultiDimensionalNumberTensor[](1),
            new TensorLib.StringTensor[](0));

        TensorLib.Number[] memory numbers = new TensorLib.Number[](2);
        numbers[0] = TensorLib.Number(7286679744720459, 17); // 0.07286679744720459
        numbers[1] = TensorLib.Number(4486280083656311, 16); // 0.4486280083656311
        modelInput.numbers[0] = TensorLib.numberTensor1D("input", numbers);

        ModelOutput memory output = OG_INFERENCE_CONTRACT.runModelInference(
            ModelInferenceRequest(ModelInferenceMode.ZK, modelId, modelInput));

        if (output.is_simulation_result == false) {
            resultNumber = output.numbers[0].values[0];
        } else {
            resultNumber = TensorLib.Number(0, 0);
        }
    }

    function runVanilla() public {
        ModelInput memory modelInput = ModelInput(
            new TensorLib.MultiDimensionalNumberTensor[](1),
            new TensorLib.StringTensor[](0));

        TensorLib.Number[] memory numbers = new TensorLib.Number[](2);
        numbers[0] = TensorLib.Number(7286679744720459, 17); // 0.07286679744720459
        numbers[1] = TensorLib.Number(4486280083656311, 16); // 0.4486280083656311

        modelInput.numbers[0] = TensorLib.numberTensor1D("input", numbers);

        ModelOutput memory output = OG_INFERENCE_CONTRACT.runModelInference(
            ModelInferenceRequest(
                ModelInferenceMode.VANILLA,
                "QmbbzDwqSxZSgkz1EbsNHp2mb67rYeUYHYWJ4wECE24S7A",
                modelInput
        ));

        if (output.is_simulation_result == false) {
            resultNumber = output.numbers[0].values[0];
        } else {
            resultNumber = TensorLib.Number(0, 0);
        }
    }

    function runLlm() public {
        string[] memory stopSequence = new string[](1);
        stopSequence[0] = "<end>";

        LlmResponse memory llmResult = OG_INFERENCE_CONTRACT.runLLMInference(
            LlmInferenceRequest(
                LlmInferenceMode.VANILLA,
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "Hello ser, who are you?\n<start>",
                1000,
                stopSequence,
                0
        ));

        resultString = llmResult.answer;
    }

    function runTee() public {
        string[] memory stopSequence = new string[](1);
        stopSequence[0] = "<end>";

        LlmResponse memory llmResult = OG_INFERENCE_CONTRACT.runLLMInference(
            LlmInferenceRequest(
                LlmInferenceMode.TEE,
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "Hello TEE ser, who are you?\n<start>",
                1000,
                stopSequence,
                0
        ));

        resultString = llmResult.answer;
    }

    function result() public view returns (int128, int128) {
        return (resultNumber.value, resultNumber.decimals);
    }
}
