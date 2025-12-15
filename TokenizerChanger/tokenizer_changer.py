import copy
import json
from tqdm import tqdm
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from multiprocessing import Pool, cpu_count


class TokenizerChanger:
    def __init__(self, tokenizer: PreTrainedTokenizerFast = None, space_sign: str = "Ġ"):
                """Edit a Hugging Face fast tokenizer by manipulating its underlying tokenizers state.

                This utility operates on the JSON state of ``tokenizer.backend_tokenizer`` (the
                `tokenizers` Rust-backed tokenizer) and then reconstructs a new
                :class:`~transformers.PreTrainedTokenizerFast` instance from the modified state.

                Parameters
                ----------
                tokenizer:
                        The tokenizer to operate on. Must be a *fast* tokenizer
                        (:class:`~transformers.PreTrainedTokenizerFast`).
                space_sign:
                        Replacement marker used when adding tokens that contain spaces. Many BPE
                        tokenizers represent a leading space with a special character (e.g. ``"Ġ"``).

                Notes
                -----
                - Methods in this class mutate ``self.state`` (the tokenizer JSON) and may replace
                    ``self.tokenizer`` when calling :meth:`updated_tokenizer`.
                - Some operations (e.g. :meth:`add_token_suggestion`) can prompt the user via
                    stdin/stdout; this is not suitable for non-interactive environments.
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
        """Validate that a tokenizer is loaded.

        Raises
        ------
        ValueError
            If no tokenizer has been loaded via ``__init__`` or :meth:`load_tokenizer`.
        """
        if not self.tokenizer or not self.state:
            raise ValueError("Tokenizer is not loaded")

    def set_space_sign(self, space_sign: str):
        """Set the marker used to represent spaces when adding tokens.

        Parameters
        ----------
        space_sign:
            The character (or string) used to replace regular spaces in token strings.
        """
        self.space_sign = space_sign

    def load_tokenizer(self, tokenizer: PreTrainedTokenizerFast):
        """Load a tokenizer and initialize internal JSON state.

        Parameters
        ----------
        tokenizer:
            The tokenizer to load (must be a fast tokenizer).
        """
        self.tokenizer = tokenizer
        self.state = json.loads(tokenizer.backend_tokenizer.__getstate__())
        self.initial_length = len(self.state["model"]["vocab"])

    def _fill_unwanted_merges(self, batch: list[str]):
        """Worker helper to collect merges containing unwanted tokens.

        Parameters
        ----------
        batch:
            A batch of merge entries, where each entry is expected to be
            ``(processed_merge, original_merge)``.

        Returns
        -------
        list
            A list of original merges that contain any token from ``self.unwanted_tokens``.
        """
        self.__is_tokenizer()
        unwanted_merges = []
        for processed_merge, original_merge in tqdm(batch, desc="Finding unwanted merges"):
            if any(token in processed_merge for token in self.unwanted_tokens):
                unwanted_merges.append(original_merge)

        return unwanted_merges

    def format_merges(self):
        """Normalize merge entries to tuples of strings.

        Some tokenizer JSON dumps store merges as whitespace-separated strings.
        This converts each merge into a tuple (typically length 2).
        """
        self.__is_tokenizer()

        for i in tqdm(range(len(self.state["model"]["merges"])), desc="Formatting merges"):
            if type(self.state["model"]["merges"][i]) != tuple:
                self.state["model"]["merges"][i] = tuple(
                    map(str, self.state["model"]["merges"][i].split()))

    def _move_special_tokens(self):
        """Shift special token ids to the end of the vocabulary.

        When new tokens are added, ids may need to be adjusted so that special tokens
        keep consistent positions relative to the base vocabulary.
        """

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
        """Derive candidate tokens from a merge and add them to the vocab.

        Parameters
        ----------
        merge:
            The merge represented as a list/sequence of string pieces.

        Notes
        -----
        This method calls :meth:`add_tokens` and therefore mutates ``self.state``.
        """
        processed_merge = ''.join(merge).replace(' ', '')
        split_merge = ''.join(merge).split()
        self.add_tokens([processed_merge] + split_merge)


# =================== Find operations ===================

    def find_least_tokens(self, k_least: int, exclude: list[str] = [], consider_excluded_tokens: bool = False):
        """Select *k* tokens from the tail of the vocabulary ordering.

        Parameters
        ----------
        k_least:
            Number of tokens to select.
        exclude:
            Tokens to ignore.
        consider_excluded_tokens:
            If True, ``k_least`` is reduced by ``len(exclude)`` so that the total number
            of tokens removed/considered stays closer to the user-requested number.

        Notes
        -----
        Despite the name, this method does **not** compute true token frequencies.
        It walks the vocabulary in reverse insertion/id order and picks the first *k*.
        This is only “least frequent” if your tokenizer’s vocab is ordered by frequency.

        Side Effects
        ------------
        Updates ``self.unwanted_tokens``.
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
        """Add existing tokens from ``unwanted_tokens`` to the internal deletion list.

        Parameters
        ----------
        unwanted_tokens:
            Candidate tokens to look up in the tokenizer vocabulary.

        Side Effects
        ------------
        Appends found tokens to ``self.unwanted_tokens``.
        """
        self.__is_tokenizer()

        unwanted_tokens = list(set(unwanted_tokens))

        for token in tqdm(unwanted_tokens, desc="Finding unwanted tokens"):
            if token in self.state["model"]["vocab"].keys():
                self.unwanted_tokens.append(token)

    def find_token_id_gap(self):
        """Find the most recent gap in token ids.

        Returns
        -------
        int
            The id at the start of the last detected gap. ``add_tokens`` uses this
            to pick ids that do not collide with existing ones.
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
        """Add new tokens to the tokenizer vocabulary.

        Parameters
        ----------
        tokens:
            Token strings to add. Any literal spaces will be replaced by ``self.space_sign``.

        Notes
        -----
        This method mutates the underlying JSON vocab (``self.state['model']['vocab']``).
        It does not automatically update merges; call :meth:`updated_tokenizer` to rebuild
        a usable :class:`~transformers.PreTrainedTokenizerFast`.
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
        """Prompt the user to add tokens required by a merge.

        Parameters
        ----------
        merge:
            The merge that cannot be added because required pieces are missing.

        Returns
        -------
        bool
            True if tokens were added (or permission already granted), otherwise False.

        Warnings
        --------
        This method is interactive: it calls ``input()`` and prints to stdout.
        Avoid using it in non-interactive environments.
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
        """Add merge rules to the tokenizer.

        Parameters
        ----------
        merges:
            Iterable of merges. Each merge should be a sequence (typically length 2)
            describing pieces that can be merged.

        Notes
        -----
        If required tokens are missing, this may call :meth:`add_token_suggestion`, which
        can prompt the user.
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
        """Single-process merge deletion helper.

        Parameters
        ----------
        processed_merges:
            Iterable of ``(processed_merge, original_merge)`` pairs.
        """
        unwanted_merges_set = set()
        for processed_merge, original_merge in tqdm(processed_merges, desc="Finding unwanted merges"):
            if any(token in processed_merge for token in self.unwanted_tokens):
                unwanted_merges_set.add(tuple(original_merge))

        self.state["model"]["merges"] = [merge for merge in tqdm(
            self.state["model"]["merges"], desc="Deleting unwanted merges") if tuple(merge) not in unwanted_merges_set]

    def delete_merges(self, unwanted_tokens: list[str] = None, n_jobs=1):
        """Delete merge rules that contain unwanted tokens.

        Parameters
        ----------
        unwanted_tokens:
            If provided, merges are filtered using these tokens. Otherwise uses
            the current ``self.unwanted_tokens``.
        n_jobs:
            Number of worker processes to use. ``1`` uses a single process.

        Notes
        -----
        Multiprocessing can be unreliable on some platforms/environments (especially
        when pickling bound methods). If it fails, the implementation falls back to
        single-process deletion.
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
        """Delete merges that reference tokens not present in a provided vocab.

        Parameters
        ----------
        vocab:
            A list of allowed tokens. Any merge referencing tokens outside this list
            will be removed.
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
        """Delete tokens from the vocabulary and optionally delete affected merges.

        Parameters
        ----------
        unwanted_tokens:
            Tokens to delete. If empty, deletes the current ``self.unwanted_tokens``.
        include_substrings:
            If True, expands the deletion list by searching for tokens present in the
            current vocabulary. (Note: this does not search arbitrary substrings; it
            filters to existing vocab entries.)
        delete_merges:
            If True, also remove merges that reference deleted tokens.
        n_jobs:
            Worker process count used by :meth:`delete_merges`.

        Raises
        ------
        KeyError
            If a requested token is not in the vocabulary.
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
        """Delete tokens that overlap with another vocabulary.

        Parameters
        ----------
        vocab:
            A vocabulary mapping (e.g., token->id). Any token present in both
            vocabularies will be deleted from this tokenizer.
        delete_merges:
            Whether to also remove merges referencing the deleted tokens.
        """
        overlaps = list(set(self.get_overlapping_tokens(vocab)))
        self.delete_tokens(unwanted_tokens=overlaps,
                           include_substrings=False, delete_merges=delete_merges)

    def _delete_none_types(self):
        """Remove ``None``-valued keys from the tokenizer model state.

        Some tokenizer JSON states may include keys set to ``None``; removing them can
        help compatibility with downstream serialization.
        """
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
        """Delete *k* tokens selected by :meth:`find_least_tokens`.

        Parameters
        ----------
        k:
            Number of tokens to delete.
        exclude:
            Tokens to skip during selection.
        delete_merges:
            Whether to also delete merges that reference deleted tokens.
        consider_excluded_tokens:
            Whether to subtract the excluded count from ``k``.
        n_jobs:
            Worker process count used by :meth:`delete_merges`.

        Notes
        -----
        This uses :meth:`find_least_tokens`, which is based on vocab ordering rather than
        true frequency unless your tokenizer’s vocab is frequency-sorted.
        """
        self.find_least_tokens(k_least=k, exclude=exclude,
                               consider_excluded_tokens=consider_excluded_tokens)
        self.delete_tokens(include_substrings=False,
                           delete_merges=delete_merges, n_jobs=n_jobs)


# ==================== Get operations ===================

    def get_overlapping_tokens(self, vocab: dict):
        """Return tokens that exist in both vocabularies.

        Parameters
        ----------
        vocab:
            A mapping representing another vocabulary (e.g., token->id).

        Returns
        -------
        list[str]
            Tokens that appear in both ``vocab`` and this tokenizer's vocab.
        """
        self.__is_tokenizer()

        overlapping_tokens = []
        for token in tqdm(vocab.keys(), desc="Finding overlapping tokens"):
            if token in self.state["model"]["vocab"].keys():
                overlapping_tokens.append(token)

        return overlapping_tokens

    def get_overlapping_merges(self, merges: list):
        """Return merges that overlap with another merge list.

        Parameters
        ----------
        merges:
            A list of merges from another tokenizer.

        Returns
        -------
        list
            Merges from this tokenizer that appear to overlap with the provided list.
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
        """Persist the updated tokenizer to disk.

        Parameters
        ----------
        path:
            Output directory passed to ``tokenizer.save_pretrained``.

        Notes
        -----
        This calls :meth:`updated_tokenizer` first to rebuild the tokenizer instance from
        the current JSON state.
        """

        self.updated_tokenizer()

        self.tokenizer.save_pretrained(path)

    def updated_tokenizer(self):
        """Rebuild and return a tokenizer from the current internal JSON state.

        Returns
        -------
        transformers.PreTrainedTokenizerFast
            A new tokenizer instance that reflects the current ``self.state``.

        Side Effects
        ------------
        Replaces ``self.tokenizer`` and refreshes ``self.state`` from the rebuilt tokenizer.
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
