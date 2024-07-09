import json
from tqdm import tqdm
from tokenizers import models
from transformers import PreTrainedTokenizer


class TokenizerChanger:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.unwanted_words = []
        self.none_types = []
        self.model_state = json.loads(
            tokenizer.backend_tokenizer.model.__getstate__())

    def delete_tokens(self):
        for word in tqdm(self.unwanted_words, desc="Deleting unwanted words"):
            del self.model_state["vocab"][word]

    def find_least_words(self, k_least: int):
        for k, v in tqdm(self.model_state["vocab"].items(), desc="Finding unwanted words"):
            if v >= (len(self.model_state["vocab"]) - k_least):
                self.unwanted_words.append(k)

    def find_tokens(self, unwanted_words: list[str]):
        for word in self.model_state["vocab"]:
            for unwanted_word in unwanted_words:
                if unwanted_word in word:
                    self.unwanted_words.append(word)

    def delete_merges(self):
        processed_merges = [(''.join(merge).replace(' ', ''), merge)
                            for merge in self.model_state["merges"]]

        unwanted_merges_set = set()

        for processed_merge, original_merge in tqdm(processed_merges, desc="Finding unwanted merges"):
            if any(word in processed_merge for word in self.unwanted_words):
                unwanted_merges_set.add(original_merge)

        self.model_state["merges"] = [merge for merge in tqdm(
            self.model_state["merges"], desc="Deleting unwanted merges") if merge not in unwanted_merges_set]

    def format_merges(self):
        for i in tqdm(range(len(self.model_state["merges"])), desc="Formating merges"):
            self.model_state["merges"][i] = tuple(
                map(str, self.model_state["merges"][i].split()))

    def delete_none_types(self):
        for k, v in self.model_state.items():
            if v == None:
                self.none_types.append(k)

        for k in self.none_types:
            del self.model_state[k]

    def delete_k_least_frequent_tokens(self, k: int):
        self.find_least_words(k)
        self.delete_tokens()
        self.delete_merges()
        self.format_merges()
        self.delete_none_types()

    def delete_unwanted_tokens(self, unwanted_tokens: list):
        self.find_tokens(unwanted_tokens)
        self.delete_tokens()
        self.delete_merges()
        self.format_merges()
        self.delete_none_types()

    def save_tokenizer(self, path: str = "updated_tokenizer"):
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
