# Tokenizer-Changer

Python script for manipulating the existing tokenizer.

The solution was tested on Llama3-8B tokenizer.

-----

# Installation

Installation from PyPI:

```bash
pip install tokenizerchanger
```

-----

# Usage

```python
changer = TokenizerChanger(tokenizer, space_sign)
```

Create the object of `TokenizerChanger` class that optionally requires an existing tokenizer and space sign, which differs from one tokenizer to another. The tokenizer could be `PreTrainedTokenizerFast` class from 🤗 `tokenizers` library.

```python
changer.load_tokenizer(tokenizer)
```

If you did not load the tokenizer with `TokenizerChanger` class declaration, you can load it using this function.

``` python
changer.set_space_sign(space_sign)
```

If you did not set the space sign with `TokenizerChanger` class declaration, you can set it using this function. Default space sign is `Ġ`.

## Deletion

```python
changer.delete_tokens(list_of_unwanted_tokens, include_substrings)
```

Deletes the unwanted tokens from the tokenizer. If `include_substrings` is `True`, all token occurrences will be deleted even in other tokens. Defaults to `True`.

```python
changer.delete_k_least_frequent_tokens(k=1000)
changer.delete_k_least_frequent_tokens(k=1000, exclude=list_of_tokens)
```

Deletes k most frequent tokens. The `exclude` argument stands for tokens that will be ignored during the deletion of the least frequent tokens.

```python
changer.delete_overlaps(vocab)
```

Finds and deletes all intersections of the `tokenizer`'s vocabulary and the `vocab` variable from the `tokenizer`. Notice that `vocab` should be a `dict` variable.

```python
changer.delete_inappropriate_merges(vocab)
```

Deletes all merges from `tokenizer` which contradict the `vocab` variable. Notice that `vocab` should be a `list[str]` variable.

## Addition

The idea of creating such functions arose due to the fact that the built-in functions do not add tokens/merges properly, when some tokens are deleted. That is why you can get more tokens after encoding the same text, even if the necessary tokens have been added.

```python
changer.add_tokens(list_of_tokens)
```

Adds the tokens from the list. The indexes will be filled automatically.

```python
changer.add_merges(list_of_merges)
```

Adds the merges from the list.

## "Get" functions

```python
changer.get_overlapping_tokens(vocab)
```

Returns the intersection between the `tokenizer`'s vocabulary and the `vocab` variable. Notice that `vocab` should be a `dict` variable.

```python
changer.get_overlapping_merges(merges)
```

Returns the intersection between the `tokenizer`'s merges and the `merges` variable. Notice that `merges` should be a `list` variable.

## Saving

```python
changer.save_tokenizer(path)
```

Saves the current state of the changed tokenizer. Additionally, it saves tokenizer configs into `path` folder (`./updated_tokenizer` by default).

```python
tokenizer = ch.updated_tokenizer()
```

Return the changed tokenizer.
