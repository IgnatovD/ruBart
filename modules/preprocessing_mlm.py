from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollator, PreTrainedTokenizer

@dataclass
class DataCollatorForLanguageModeling(DataCollator):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            attention_mask = self.get_attention_mask(inputs)
            return {"input_ids": inputs, "masked_lm_labels": labels, 'attention_mask': attention_mask}
        else:
            return {"input_ids": batch, "labels": batch}

    def get_attention_mask(self, inputs):
        inputs = inputs.masked_fill(inputs != 1, 0)
        return 1 - inputs

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        # are_tensors_same_length : bool
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer.token_to_id("<pad>") is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.token_to_id("<pad>"))

    def get_special_tokens_mask(self, token_ids_0: List) -> List[int]:
        """
          Returns:
            A list of integers in the range [0, 1]: 1 for a special token ('<s>': 0, '</s>' : 2), 0 for a sequence token.
        """
        return list(map(lambda x: int(x in (0, 2)), token_ids_0))

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.token_to_id("<mask>") is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.get_special_tokens_mask(val) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.token_to_id("<pad>") is not None:
            padding_mask = labels.eq(self.tokenizer.token_to_id("<pad>"))
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.token_to_id("<mask>")

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

def get_tokenizer(max_length=128):
    tokenizer = ByteLevelBPETokenizer('tokenizer/bart-vocab.json',
                                      'tokenizer/bart-merges.txt')

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer

def batch_generator(data, batch_size):
    tokenizer = get_tokenizer(max_length=128)
    mlm = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
    for j in range(0, len(data), batch_size):
        preproc = [torch.tensor(tokenizer.encode(sentence).ids) for sentence in data[j: j + batch_size]]
        yield mlm.collate_batch(preproc)