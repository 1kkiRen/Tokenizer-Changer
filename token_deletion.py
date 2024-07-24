import json
from tqdm import tqdm
from tokenizers import models
from transformers import PreTrainedTokenizer


class TokenizerChanger:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.unwanted_tokens = []
        self.none_types = []
        self.target_changes = 0
        self.model_state = json.loads(
            tokenizer.backend_tokenizer.model.__getstate__())

    def delete_tokens(self, unwanted_tokens: list[str] = None):
        self.unwanted_tokens = list(set(unwanted_tokens)) if unwanted_tokens else list(
            set(self.unwanted_tokens))
        for token in tqdm(self.unwanted_tokens, desc="Deleting unwanted words"):
            del self.model_state["vocab"][token]

    def find_least_tokens(self, k_least: int, exclude: list[str] = []):
        self.unwanted_tokens = []
        for k, v in tqdm(dict(reversed(list(self.model_state["vocab"].items()))).items(), desc="Finding unwanted tokens"):
            if len(self.unwanted_tokens) >= k_least:
                break
            if k not in exclude:
                self.unwanted_tokens.append(k)

    def find_tokens(self, unwanted_tokens: list[str]):
        for token in self.model_state["vocab"]:
            for unwanted_token in unwanted_tokens:
                if unwanted_token in token:
                    self.unwanted_tokens.append(token)

    def delete_merges(self, unwanted_tokens: list[str] = None):
        processed_merges = [(''.join(merge).replace(' ', ''), merge)
                            for merge in self.model_state["merges"]]

        unwanted_merges_set = set()

        self.unwanted_tokens = list(set(unwanted_tokens)) if unwanted_tokens else list(
            set(self.unwanted_tokens))

        for processed_merge, original_merge in tqdm(processed_merges, desc="Finding unwanted merges"):
            if any(token in processed_merge for token in self.unwanted_tokens):
                unwanted_merges_set.add(original_merge)

        self.model_state["merges"] = [merge for merge in tqdm(
            self.model_state["merges"], desc="Deleting unwanted merges") if merge not in unwanted_merges_set]

    def find_token_id_gap(self):
        reversed_vocab_values = list(
            reversed(self.model_state['vocab'].values()))
        last_gap = 0
        for i in range(len(self.model_state['vocab']) - 1):
            if reversed_vocab_values[i] - reversed_vocab_values[i + 1] > 1:
                last_gap = reversed_vocab_values[i + 1]

        return last_gap

    def add_tokens(self, tokens: list[str]):
        i = 1
        border_id = self.find_token_id_gap()
        for token in tqdm(tokens, desc="Adding tokens"):
            if token not in self.model_state["vocab"]:
                while border_id + i in self.model_state['vocab'].values():
                    i += 1
                self.model_state["vocab"][token] = border_id + i
                i += 1

    def add_merges(self, merges: list[str]):
        for merge in tqdm(self.model_state["merges"], desc="Adding merges"):
            merges.append(merge)

        self.model_state["merges"] = list(set(merges))

    def delete_inappropriate_merges(self, vocab: list[str]):
        processed_merges = [(''.join(merge).replace(' ', ''), merge)
                            for merge in self.model_state["merges"]]

        unwanted_merges_set = set()

        for processed_merge, original_merge in tqdm(processed_merges, desc="Finding unwanted merges"):
            if not all(token in vocab for token in [processed_merge, original_merge[0], original_merge[1]]):
                unwanted_merges_set.add(original_merge)

        self.model_state["merges"] = [merge for merge in tqdm(
            self.model_state["merges"], desc="Deleting unwanted merges") if merge not in unwanted_merges_set]

    def get_overlapping_tokens(self, vocab: dict):
        overlapping_tokens = []
        for token in tqdm(vocab.keys(), desc="Finding overlapping tokens"):
            if token in self.model_state["vocab"].keys():
                overlapping_tokens.append(token)
        return overlapping_tokens

    def get_overlapping_megres(self, merges: list):
        overlapping_merges = []

        processed_merges_new_tokenizer = [(''.join(merge).replace(' ', ''), merge)
                                          for merge in self.model_state["merges"]]

        processed_merges_old_tokenizer = [(''.join(merge).replace(' ', ''), merge)
                                          for merge in merges]

        for merge in tqdm(processed_merges_new_tokenizer, desc="Finding overlapping merges"):
            if any(merge in processed_merge for processed_merge in processed_merges_old_tokenizer):
                overlapping_merges.append(merge)

        return overlapping_merges

    def format_merges(self):
        for i in tqdm(range(len(self.model_state["merges"])), desc="Formating merges"):
            if type(self.model_state["merges"][i]) != tuple:
                self.model_state["merges"][i] = tuple(
                    map(str, self.model_state["merges"][i].split()))

    def delete_none_types(self):
        for k, v in self.model_state.items():
            if v == None:
                self.none_types.append(k)

        for k in self.none_types:
            del self.model_state[k]

    def delete_k_least_frequent_tokens(self, k: int, exclude: list[str] = []):
        self.find_least_tokens(k, exclude)
        self.delete_tokens()
        self.delete_merges()

    def delete_unwanted_tokens(self, unwanted_tokens: list):
        self.find_tokens(unwanted_tokens)
        self.delete_tokens()
        self.delete_merges()

    def delete_overlaps(self, vocab: dict):
        overlaps = list(set(self.get_overlapping_tokens(vocab)))
        self.delete_tokens(unwanted_tokens=overlaps)
        self.delete_merges()

    def save_tokenizer(self, path: str = "updated_tokenizer"):
        self.format_merges()
        self.delete_none_types()

        model_class = getattr(
            models, self.model_state.pop("type")
        )

        self.tokenizer.backend_tokenizer.model = model_class(
            **self.model_state)

        self.model_state = json.loads(
            self.tokenizer.backend_tokenizer.model.__getstate__())

        self.tokenizer.save_pretrained(path)

    def updated_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer
