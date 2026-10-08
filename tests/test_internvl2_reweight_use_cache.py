import unittest
from types import SimpleNamespace

import torch
from torch import nn

from qmllm.models.internvl2.internvl2 import InternVL2


class GenerationOptimizedLanguageModel(nn.Module):
    """Small stand-in for the local InternLM2 logits projection behavior."""

    def __init__(self, hidden_size=4, vocab_size=7):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(vocab_size=vocab_size, use_cache=True)
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=None):
        del attention_mask
        if inputs_embeds is not None:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        else:
            batch_size, sequence_length = input_ids.shape

        cache_enabled = self.config.use_cache if use_cache is None else use_cache
        output_length = 1 if cache_enabled else sequence_length
        logits = torch.zeros(batch_size, output_length, self.config.vocab_size)
        return SimpleNamespace(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class FakeInternVLChatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = GenerationOptimizedLanguageModel()
        self.config = SimpleNamespace(use_return_dict=True)
        self.num_image_token = 1


class InternVL2ReweightUseCacheTest(unittest.TestCase):
    def test_labels_disable_cache_so_loss_receives_full_sequence_logits(self):
        model = InternVL2(FakeInternVLChatModel(), tokenizer=None)
        sequence_length = 5
        inputs_embeds = torch.zeros(1, sequence_length, 4)
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        attention_mask = torch.ones(1, sequence_length, dtype=torch.long)

        try:
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        except ValueError as exc:
            self.fail(f"loss calculation received truncated logits: {exc}")

        self.assertEqual(outputs.logits.shape, (1, sequence_length, 7))
        self.assertTrue(torch.isfinite(outputs.loss))


if __name__ == "__main__":
    unittest.main()
