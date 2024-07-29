import re
import json
from tqdm import tqdm
from tokenizers import models
from transformers import PreTrainedTokenizerFast
from multiprocessing import Pool, cpu_count


class TokenizerChanger:
    def __init__(self, tokenizer: PreTrainedTokenizerFast = None):
        """Base class for changing tokenizers

        Args:
            tokenizer (PreTrainedTokenizerFast, optional): the tokenizer that will be changed. Defaults to None.
        """
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.unwanted_tokens = []
        self.none_types = []
        self.target_changes = 0
        self.model_state = json.loads(
            tokenizer.backend_tokenizer.model.__getstate__()) if tokenizer else {}

    def __is_tokenizer(self):
        """The tokenizer existance checker

        Raises:
            ValueError: Tokenizer is not loaded
        """
        if not self.tokenizer or not self.model_state:
            raise ValueError("Tokenizer is not loaded")

    def load_tokenizer(self, tokenizer: PreTrainedTokenizerFast):
        """The tokenizer loader function

        Args:
            tokenizer (PreTrainedTokenizerFast): the tokenizer to be loaded
        """
        self.tokenizer = tokenizer
        self.model_state = json.loads(
            tokenizer.backend_tokenizer.model.__getstate__())

    def find_least_tokens(self, k_least: int, exclude: list[str] = []):
        """Finds the k least frequent tokens

        Args:
            k_least (int): number of tokens to find
            exclude (list[str], optional): tokens that will be excluded from the search. Defaults to [].
        """
        self.__is_tokenizer()

        self.unwanted_tokens = []
        for k, v in tqdm(dict(reversed(list(self.model_state["vocab"].items()))).items(), desc="Finding unwanted tokens"):
            if len(self.unwanted_tokens) >= k_least:
                break
            if k not in exclude:
                self.unwanted_tokens.append(k)

    def find_tokens(self, unwanted_tokens: list[str]):
        """Finds the tokens and their occurrences

        Args:
            unwanted_tokens (list[str]): list of tokens to find
        """
        self.__is_tokenizer()

        unwanted_tokens = list(set(unwanted_tokens))

        for token in tqdm(unwanted_tokens, desc="Finding unwanted tokens"):
            if token in self.model_state["vocab"].keys():
                self.unwanted_tokens.append(token)

    def _fill_unwanted_merges(self, batch: list[str]):
        """Fills the unwanted merges process

        Args:
            batch (list[str]): list of merges

        Returns:
            list[str]: list of unwanted merges
        """
        self.__is_tokenizer()
        unwanted_merges = []
        for processed_merge, original_merge in tqdm(batch, desc="Finding unwanted merges"):
            if any(token in processed_merge for token in self.unwanted_tokens):
                unwanted_merges.append(original_merge)
        return unwanted_merges

    def delete_merges(self, unwanted_tokens: list[str] = None):
        """Deletes the unwanted merges

        Args:
            unwanted_tokens (list[str], optional): the merges deletion will be processed exactly for this tokens. Defaults to None.
        """
        self.__is_tokenizer()

        pattern = r"\s+"
        processed_merges = [(re.sub(pattern, "", merge), merge)
                            for merge in self.model_state["merges"]]

        unwanted_merges_set = set()

        self.unwanted_tokens = list(set(unwanted_tokens)) if unwanted_tokens else list(
            set(self.unwanted_tokens))

        try:
            num_chunks = cpu_count()
            chunk_size = len(processed_merges) // num_chunks
            chunks = [processed_merges[i:i + chunk_size]
                      for i in range(0, len(processed_merges), chunk_size)]

            with Pool(num_chunks) as pool:
                unwanted_merges = list(tqdm(pool.imap(
                    self._fill_unwanted_merges, chunks), total=len(chunks), desc="Processing merges"))

            unwanted_merges_set = set()
            for unwanted_merge in tqdm(unwanted_merges, desc="Filling unwanted merges"):
                unwanted_merges_set.update(unwanted_merge)

            self.model_state["merges"] = [merge for merge in tqdm(
                self.model_state["merges"], desc="Deleting unwanted merges") if merge not in unwanted_merges_set]
        except Exception as e:
            if e == ZeroDivisionError:
                unwanted_merges_set = set()
                for processed_merge, original_merge in tqdm(processed_merges, desc="Finding unwanted merges"):
                    if any(token in processed_merge for token in self.unwanted_tokens):
                        unwanted_merges_set.add(original_merge)

                self.model_state["merges"] = [merge for merge in tqdm(
                    self.model_state["merges"], desc="Deleting unwanted merges") if merge not in unwanted_merges_set]

    def find_token_id_gap(self):
        """Finds the token id of the gap

        Returns:
            int: the token id of the gap 
        """
        self.__is_tokenizer()

        reversed_vocab_values = list(
            reversed(self.model_state['vocab'].values()))
        last_gap = 0
        for i in range(len(self.model_state['vocab']) - 1):
            if reversed_vocab_values[i] - reversed_vocab_values[i + 1] > 1:
                last_gap = reversed_vocab_values[i + 1]

        return last_gap

    def add_tokens(self, tokens: list[str]):
        """Adds the tokens to the tokenizer

        Args:
            tokens (list[str]): list of tokens to be added
        """
        self.__is_tokenizer()

        border_id = self.find_token_id_gap()
        vocab_values_set = set(self.model_state['vocab'].values())
        next_id = border_id + 1

        for token in tqdm(tokens, desc="Adding tokens"):
            if token not in self.model_state["vocab"]:
                while next_id in vocab_values_set:
                    next_id += 1
                self.model_state["vocab"][token] = next_id
                vocab_values_set.add(next_id)
                next_id += 1

    def add_merges(self, merges: list[str]):
        """Adds the merges to the tokenizer

        Args:
            merges (list[str]): list of merges to be added
        """
        self.__is_tokenizer()

        for merge in tqdm(self.model_state["merges"], desc="Adding merges"):
            merges.append(merge)

        self.model_state["merges"] = list(set(merges))

    def delete_inappropriate_merges(self, vocab: list[str]):
        """Deletes all merges from tokenizer which contradict the vocab variable

        Args:
            vocab (list[str]): list of tokens
        """
        self.__is_tokenizer()

        processed_merges = [(''.join(merge).replace(' ', ''), merge)
                            for merge in self.model_state["merges"]]

        unwanted_merges_set = set()

        for processed_merge, original_merge in tqdm(processed_merges, desc="Finding unwanted merges"):
            if not all(token in vocab for token in [processed_merge, original_merge[0], original_merge[1]]):
                unwanted_merges_set.add(original_merge)

        self.model_state["merges"] = [merge for merge in tqdm(
            self.model_state["merges"], desc="Deleting unwanted merges") if merge not in unwanted_merges_set]

    def get_overlapping_tokens(self, vocab: dict):
        """Returns the intersection between the tokenizer's vocabulary and the vocab variable

        Args:
            vocab (dict): the vocabulary

        Returns:
            list[str]: the list of overlapping tokens
        """
        self.__is_tokenizer()

        overlapping_tokens = []
        for token in tqdm(vocab.keys(), desc="Finding overlapping tokens"):
            if token in self.model_state["vocab"].keys():
                overlapping_tokens.append(token)
        return overlapping_tokens

    def get_overlapping_megres(self, merges: list):
        """Returns the intersection between the tokenizer's merges and the merges variable

        Args:
            merges (list): the merges

        Returns:
            list: the list of overlapping merges
        """
        self.__is_tokenizer()

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
        """Formats the merges to the tuple format"""
        self.__is_tokenizer()

        for i in tqdm(range(len(self.model_state["merges"])), desc="Formating merges"):
            if type(self.model_state["merges"][i]) != tuple:
                self.model_state["merges"][i] = tuple(
                    map(str, self.model_state["merges"][i].split()))

    def delete_tokens(self, unwanted_tokens: list[str] = [], include_substrings: bool = True):
        """Deletes the unwanted tokens from the tokenizer

        Args:
            unwanted_tokens (list[str]): list of tokens to be deleted. If empty self.unwanted_tokens will be deleted. Defaults to [].
            include_substrings (bool, optional): the flag of deletion the occurences of the tokens in other tokens. Defaults to True.
        """
        self.__is_tokenizer()

        self.unwanted_tokens = list(set(unwanted_tokens)) if unwanted_tokens else list(
            set(self.unwanted_tokens))

        if include_substrings:
            self.find_tokens(self.unwanted_tokens)

        for token in tqdm(list(set(self.unwanted_tokens)), desc="Deleting unwanted words"):
            del self.model_state["vocab"][token]

        self.delete_merges()

        self.unwanted_tokens = []

    def delete_overlaps(self, vocab: dict):
        """Finds and deletes all intersections of the tokenizer's vocabulary and the vocab variable from the tokenizer

        Args:
            vocab (dict): the vocabulary
        """
        overlaps = list(set(self.get_overlapping_tokens(vocab)))
        self.delete_tokens(unwanted_tokens=overlaps, include_substrings=False)

    def _delete_none_types(self):
        """Deletes all None fields from the tokenizer"""
        self.__is_tokenizer()

        for k, v in self.model_state.items():
            if v == None:
                self.none_types.append(k)

        for k in self.none_types:
            del self.model_state[k]

    def delete_k_least_frequent_tokens(self, k: int, exclude: list[str] = []):
        """Deletes k most frequent tokens. The exclude argument stands for tokens that will be ignored during the deletion of least frequent tokens

        Args:
            k (int): number of tokens to delete
            exclude (list[str], optional): tokens to be ignored. Defaults to [].
        """
        self.find_least_tokens(k, exclude)
        self.delete_tokens(include_substrings=False)

    def save_tokenizer(self, path: str = "updated_tokenizer"):
        """Saves the current state of the changed tokenizer. Additionally, it saves tokenizer configs into path folder (./updated_tokenizer by default)

        Args:
            path (str, optional): save to path. Defaults to "updated_tokenizer".
        """
        self.format_merges()
        self._delete_none_types()

        model_class = getattr(
            models, self.model_state.pop("type")
        )

        self.tokenizer.backend_tokenizer.model = model_class(
            **self.model_state)

        self.model_state = json.loads(
            self.tokenizer.backend_tokenizer.model.__getstate__())

        self.tokenizer.save_pretrained(path)

    def updated_tokenizer(self):
        """Returns the updated tokenizer

        Returns:
            PreTrainedTokenizerFast: the updated tokenizer
        """
        self.__is_tokenizer()
        return self.tokenizer
