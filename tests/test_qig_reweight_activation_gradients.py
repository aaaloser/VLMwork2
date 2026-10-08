import unittest
from types import SimpleNamespace

import torch
from torch import nn

from qmllm.methods.qig.quantize import pre_quant


class ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 1, bias=False)
        self.config = SimpleNamespace(
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )

    def forward(self, inputs_embeds, **kwargs):
        self.forward_kwargs = kwargs
        return (self.projection(inputs_embeds).sum(),)


class ToyProcessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyLanguageModel()

    def forward(self, **kwargs):
        return self.model(**kwargs)


class QIGReweightActivationGradientTest(unittest.TestCase):
    def test_reweight_backward_tracks_input_gradients_without_parameter_gradients(self):
        self.assertTrue(
            hasattr(pre_quant, "_prepare_reweight_backward"),
            "QIG must prepare reweighting for activation-only gradients",
        )

        model = ToyProcessModel()
        original_inputs = torch.randn(1, 5, 4)
        mini_inputs = {
            "inputs_embeds": original_inputs,
            "labels": torch.zeros(1, 5, dtype=torch.long),
        }

        prepared_inputs = pre_quant._prepare_reweight_backward(model, mini_inputs)
        outputs = model(**prepared_inputs)
        outputs[0].backward()

        self.assertIsNot(prepared_inputs, mini_inputs)
        self.assertIsNot(prepared_inputs["inputs_embeds"], original_inputs)
        self.assertIsNotNone(prepared_inputs["inputs_embeds"].grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(model.model.config.use_cache)
        self.assertFalse(model.model.config.output_attentions)
        self.assertFalse(model.model.config.output_hidden_states)
        self.assertIs(model.model.forward_kwargs["use_cache"], False)
        self.assertIs(model.model.forward_kwargs["return_dict"], True)


if __name__ == "__main__":
    unittest.main()
