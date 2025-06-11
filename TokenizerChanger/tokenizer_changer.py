import copy
import json
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from multiprocessing import Pool, cpu_count


class TokenizerChanger:
    def __init__(self, tokenizer: PreTrainedTokenizerFast = None, space_sign: str = "Ġ"):
        """Base class for changing tokenizers

        Args:
            tokenizer (PreTrainedTokenizerFast, optional): the tokenizer that will be changed. Defaults to None.
            space_sign (str, optional): the space sign. Defaults to "Ġ".
        """
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.unwanted_tokens = []
        self.none_types = []
        self.adding_permission = False
        self.original_tokenizer = copy.deepcopy(tokenizer)
        self.none_permission = False
        self.space_sign = space_sign
        self.state = json.loads(
            tokenizer.backend_tokenizer.__getstate__()) if tokenizer else {}
        self.initial_length = len(
            self.state["model"]["vocab"]) if self.state else 0


# ======================== Utils ========================

    def __is_tokenizer(self):
        """The tokenizer existence checker

        Raises:
            ValueError: Tokenizer is not loaded
        """
        if not self.tokenizer or not self.state:
            raise ValueError("Tokenizer is not loaded")

    def set_space_sign(self, space_sign: str):
        """The space sign setter

        Args:
            space_sign (str): the space sign
        """
        self.space_sign = space_sign

    def load_tokenizer(self, tokenizer: PreTrainedTokenizerFast):
        """The tokenizer loader function

        Args:
            tokenizer (PreTrainedTokenizerFast): the tokenizer to be loaded
        """
        self.tokenizer = tokenizer
        self.state = json.loads(tokenizer.backend_tokenizer.__getstate__())
        self.initial_length = len(self.state["model"]["vocab"])

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

    def format_merges(self):
        """Formats the merges to the tuple format"""
        self.__is_tokenizer()

        for i in tqdm(range(len(self.state["model"]["merges"])), desc="Formatting merges"):
            if type(self.state["model"]["merges"][i]) != tuple:
                self.state["model"]["merges"][i] = tuple(
                    map(str, self.state["model"]["merges"][i].split()))

    def _move_special_tokens(self):
        """Moves the special tokens to the end of the vocabulary"""

        for i in tqdm(range(len(self.state["added_tokens"])), desc="Moving special tokens"):
            self.state["added_tokens"][i]["id"] += (
                len(self.state["model"]["vocab"]) - self.initial_length)

        def process_special_tokens(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "special_tokens" and isinstance(value, dict):
                        for k in value.keys():
                            if "ids" in value[k]:
                                for j in tqdm(range(len(value[k]["ids"])), desc="Moving special tokens"):
                                    value[k]["ids"][j] += (
                                    len(self.state["model"]["vocab"]) - self.initial_length)
                    else:
                        process_special_tokens(value)

            elif isinstance(obj, list):
                for item in obj:
                    process_special_tokens(item)

        process_special_tokens(self.state.get("post_processor", {}))

    def _process_and_add_tokens(self, merge: list):
        processed_merge = ''.join(merge).replace(' ', '')
        split_merge = ''.join(merge).split()
        self.add_tokens([processed_merge] + split_merge)


# =================== Find operations ===================

    def find_least_tokens(self, k_least: int, exclude: list[str] = [], consider_excluded_tokens: bool = False):
        """Finds the k least frequent tokens

        Args:
            k_least (int): number of tokens to find
            exclude (list[str], optional): tokens that will be excluded from the search. Defaults to [].
            consider_excluded_tokens (bool, optional): the flag of considering the excluded tokens in the final number of deletions. Defaults to False.
        """
        self.__is_tokenizer()

        self.unwanted_tokens = []

        k_least -= len(exclude) if consider_excluded_tokens else 0
        if k_least < 0:
            raise ValueError("k must be greater than 0")

        for k, v in tqdm(dict(reversed(list(self.state["model"]["vocab"].items()))).items(), desc="Finding unwanted tokens"):
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
            if token in self.state["model"]["vocab"].keys():
                self.unwanted_tokens.append(token)

    def find_token_id_gap(self):
        """Finds the token id of the gap

        Returns:
            int: the token id of the gap 
        """
        self.__is_tokenizer()

        reversed_vocab_values = list(
            reversed(self.state["model"]['vocab'].values()))
        last_gap = 0
        for i in range(len(self.state["model"]['vocab']) - 1):
            if reversed_vocab_values[i] - reversed_vocab_values[i + 1] > 1:
                last_gap = reversed_vocab_values[i + 1]

        return last_gap


# ==================== Add operations ===================

    def add_tokens(self, tokens: list[str]):
        """Adds the tokens to the tokenizer

        Args:
            tokens (list[str]): list of tokens to be added
        """
        self.__is_tokenizer()

        border_id = self.find_token_id_gap()
        vocab_values_set = set(self.state["model"]['vocab'].values())
        next_id = border_id + 1

        for token in tqdm(tokens, desc="Adding tokens"):
            token = token.replace(' ', self.space_sign)
            if token not in self.state["model"]["vocab"]:
                while next_id in vocab_values_set:
                    next_id += 1
                self.state["model"]["vocab"][token] = next_id
                vocab_values_set.add(next_id)
                next_id += 1

    def add_token_suggestion(self, merge: str):
        """Suggests to add the missing token to the tokenizer

        Args:
            merge (str): the merge to be suggested to add
        """
        self.__is_tokenizer()

        if self.none_permission:
            return False

        if not self.adding_permission:
            print(
                f"Merge \"{merge}\" can not be added. Do you want to add its tokens? ([y]es/[n]o/[a]ll/[none])")

            answer = input().lower()

            while answer not in ['y', 'n', 'a', 'none']:
                print("Please, enter the correct answer")
                answer = input().lower()

            if answer == 'y':
                self._process_and_add_tokens(merge)
                return True
            elif answer == 'a':
                self.adding_permission = True
                self._process_and_add_tokens(merge)
                return True
            elif answer == 'n':
                return False
            else:
                self.none_permission = True
                return False

        self._process_and_add_tokens(merge)
        return True

    def add_merges(self, merges: list[str]):
        """Adds the merges to the tokenizer

        Args:
            merges (list[str]): list of merges to be added
        """
        self.__is_tokenizer()

        processed_merges = [("".join(merge), merge)
                            for merge in merges]

        vocab = set(self.state["model"]["vocab"].keys())

        for processed_merge, merge in tqdm(iterable=processed_merges, desc="Adding merges"):
            if all(token in vocab for token in merge) and processed_merge in vocab:
                self.state["model"]["merges"].append(merge)

            elif self.none_permission:
                continue

            else:
                self.add_token_suggestion(merge)

        self.adding_permission = False
        self.none_permission = False


# ================== Delete operations ==================


    def _simple_delete_merges(self, processed_merges: list[str]):
        unwanted_merges_set = set()
        for processed_merge, original_merge in tqdm(processed_merges, desc="Finding unwanted merges"):
            if any(token in processed_merge for token in self.unwanted_tokens):
                unwanted_merges_set.add(tuple(original_merge))

        self.state["model"]["merges"] = [merge for merge in tqdm(
            self.state["model"]["merges"], desc="Deleting unwanted merges") if tuple(merge) not in unwanted_merges_set]

    def delete_merges(self, unwanted_tokens: list[str] = None, n_jobs=1):
        """Deletes the unwanted merges

        Args:
            unwanted_tokens (list[str], optional): the merges deletion will be processed exactly for this tokens. Defaults to None.
            n_jobs (int, optional): number of threads while deleting merges. Defaults to 1.
        """
        self.__is_tokenizer()

        processed_merges = [("".join(merge), merge)
                            for merge in self.state["model"]["merges"]]
        unwanted_merges_set = set()

        self.unwanted_tokens = list(set(unwanted_tokens)) if unwanted_tokens else list(
            set(self.unwanted_tokens))

        if n_jobs > 1:
            try:
                n_jobs = min(n_jobs, cpu_count())

                chunk_size = len(processed_merges) // n_jobs
                chunks = [processed_merges[i:i + chunk_size]
                          for i in range(0, len(processed_merges), chunk_size)]
                with Pool(n_jobs) as pool:
                    unwanted_merges_batches = list(tqdm(pool.map(
                        self._fill_unwanted_merges, chunks), total=len(chunks), desc="Processing merges"))

                unwanted_merges = list(
                    merge for merge_batch in unwanted_merges_batches for merge in merge_batch)
                unwanted_merges_set = set()
                for unwanted_merge in tqdm(unwanted_merges, desc="Filling unwanted merges"):
                    unwanted_merges_set.add(tuple(unwanted_merge))

                self.state["model"]["merges"] = [merge for merge in tqdm(
                    self.state["model"]["merges"], desc="Deleting unwanted merges") if tuple(merge) not in unwanted_merges_set]
                self.state["model"]["vocab"] = {vocab_item: idx for idx, vocab_item in
                                                enumerate(self.state["model"]["vocab"].keys())}
            except Exception as e:
                print("Failed to delete merges with multiprocessing")
                print(e)
                print("Trying to delete merges with single thread")

                try:
                    self._simple_delete_merges(processed_merges)
                except Exception as e:
                    print("Failed to delete merges with single thread")
                    raise e

        elif n_jobs == 1:
            self._simple_delete_merges(processed_merges)
        else:
            raise ValueError("Number of jobs should be greater than 0")

        self.unwanted_tokens = []

    def delete_inappropriate_merges(self, vocab: list[str]):
        """Deletes all merges from tokenizer which contradict the vocab variable

        Args:
            vocab (list[str]): list of tokens
        """
        self.__is_tokenizer()

        processed_merges = [(''.join(merge).replace(' ', ''), merge)
                            for merge in self.state["model"]["merges"]]

        unwanted_merges_set = set()

        for processed_merge, original_merge in tqdm(processed_merges, desc="Finding unwanted merges"):
            if not all(token in vocab for token in [processed_merge, original_merge[0], original_merge[1]]):
                unwanted_merges_set.add(original_merge)

        self.state["model"]["merges"] = [merge for merge in tqdm(
            self.state["model"]["merges"], desc="Deleting unwanted merges") if merge not in unwanted_merges_set]

    def delete_tokens(self, unwanted_tokens: list[str] = [], include_substrings: bool = True, delete_merges: bool = True, n_jobs: int = 1) -> None:
        """Deletes the unwanted tokens from the tokenizer

        Args:
            unwanted_tokens (list[str]): list of tokens to be deleted. If empty self.unwanted_tokens will be deleted. Defaults to [].
            include_substrings (bool, optional): the flag of deletion the occurrences of the tokens in other tokens. Defaults to True.
            delete_merges (bool, optional): the flag of deletion the merges. Defaults to True.
            n_jobs (int, optional): number of threads while deleting merges. Defaults to 1.
        """
        self.__is_tokenizer()

        self.unwanted_tokens = list(set(unwanted_tokens)) if unwanted_tokens else list(
            set(self.unwanted_tokens))

        if include_substrings:
            self.find_tokens(self.unwanted_tokens)

        for token in tqdm(list(set(self.unwanted_tokens)), desc="Deleting unwanted words"):
            try:
                del self.state["model"]["vocab"][token]
            except KeyError:
                raise KeyError(f"Token {token} not found in the vocabulary")

        if delete_merges:
            self.delete_merges(n_jobs=n_jobs)

        self.unwanted_tokens = []

    def delete_overlaps(self, vocab: dict, delete_merges: bool = True):
        """Finds and deletes all intersections of the tokenizer's vocabulary and the vocab variable from the tokenizer

        Args:
            vocab (dict): the vocabulary
            delete_merges (bool, optional): the flag of deletion the merges. Defaults to True.
        """
        overlaps = list(set(self.get_overlapping_tokens(vocab)))
        self.delete_tokens(unwanted_tokens=overlaps,
                           include_substrings=False, delete_merges=delete_merges)

    def _delete_none_types(self):
        """Deletes all None fields from the tokenizer"""
        self.__is_tokenizer()

        self.none_types = []

        for k, v in self.state["model"].items():
            if v == None:
                self.none_types.append(k)

        for k in self.none_types:
            try:
                del self.state["model"][k]
            except KeyError:
                raise KeyError(f"Key {k} not found in the model state")

    def delete_k_least_frequent_tokens(self, k: int, exclude: list[str] = [], delete_merges: bool = True, consider_excluded_tokens: bool = False, n_jobs=1):
        """Deletes k most frequent tokens. The exclude argument stands for tokens that will be ignored during the deletion of least frequent tokens

        Args:
            k (int): number of tokens to delete
            exclude (list[str], optional): tokens to be ignored. Defaults to [].
            delete_merges (bool, optional): the flag of deletion the merges. Defaults to True.
            consider_excluded_tokens (bool, optional): the flag of considering the excluded tokens in the final number of deletions. Defaults to False.
            n_jobs (int, optional): number of threads while deleting merges. Defaults to 1.
        """
        self.find_least_tokens(k_least=k, exclude=exclude,
                               consider_excluded_tokens=consider_excluded_tokens)
        self.delete_tokens(include_substrings=False,
                           delete_merges=delete_merges, n_jobs=n_jobs)


# ==================== Get operations ===================

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
            if token in self.state["model"]["vocab"].keys():
                overlapping_tokens.append(token)

        return overlapping_tokens

    def get_overlapping_merges(self, merges: list):
        """Returns the intersection between the tokenizer's merges and the merges variable

        Args:
            merges (list): the merges

        Returns:
            list: the list of overlapping merges
        """
        self.__is_tokenizer()

        overlapping_merges = []

        processed_merges_new_tokenizer = [(''.join(merge).replace(' ', ''), merge)
                                          for merge in self.state["model"]["merges"]]

        processed_merges_old_tokenizer = [(''.join(merge).replace(' ', ''), merge)
                                          for merge in merges]

        for merge in tqdm(processed_merges_new_tokenizer, desc="Finding overlapping merges"):
            if any(merge in processed_merge for processed_merge in processed_merges_old_tokenizer):
                overlapping_merges.append(merge)

        return overlapping_merges


# ================== Saving operations ==================

    def save_tokenizer(self, path: str = "updated_tokenizer"):
        """Saves the current state of the changed tokenizer. Additionally, it saves tokenizer configs into path folder (./updated_tokenizer by default)

        Args:
            path (str, optional): save to path. Defaults to "updated_tokenizer".
        """

        self.updated_tokenizer()

        self.tokenizer.save_pretrained(path)

    def updated_tokenizer(self):
        """Returns the updated tokenizer

        Returns:
            PreTrainedTokenizerFast: the updated tokenizer
        """
        self.__is_tokenizer()

        if self.initial_length != len(self.state["model"]["vocab"]):
            self._move_special_tokens()

        backend_tokenizer = Tokenizer.from_str(json.dumps(self.state))

        self.tokenizer = self.original_tokenizer.__class__(
            tokenizer_object=backend_tokenizer, **self.original_tokenizer.init_kwargs)

        self.state = json.loads(
            self.tokenizer.backend_tokenizer.__getstate__())

        return self.tokenizer
