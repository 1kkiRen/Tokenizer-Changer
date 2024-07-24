# Tokens-Deletion
Python script for manipulating the existing tokenizer.

The solution was tested on Llama3-8B tokenizer.

-----
# Usage:

```python
changer = TokenizerChanger(tokenizer)
```
Create the object of `TokenizerChanger` class that requires an existing tokenizer that will be changed, e.g. `PreTrainedTokenizerFast` class from 🤗 Tokenizers library.

## Deletion:
```python
changer.delete_k_least_frequent_tokens(k=1000)
changer.delete_k_least_frequent_tokens(k=1000, exclude=list_of_tokens)
```
Deletes k most frequent tokens. The `exclude` argument stands for tokens that will be ignored during the deletion of least frequent tokens.

```python
changer.delete_unwanted_tokens(list_of_unwanted_tokens)
```
Deletes all tokens from `list_of_unwanted_tokens` from the tokenizer.

```python
changer.delete_tokens(list_of_unwanted_tokens)
```
Now, you can delete exactly the list of unwanted tokens, in contrast to the `delete_unwanted_tokens` function, which deletes all tokens from the list and tokens that contain unwanted tokens as a substring.

```python
changer.delete_overlaps(vocab)
```
Finds and deletes all intersections of the `tokenizer`'s vocabulary and the `vocab` variable from the `tokenizer`. Notice that `vocab` should be a `dict` variable.

```python
changer.delete_inappropriate_merges(vocab)
```
Deletes all merges from `tokenizer` which contradict the `vocab` variable. Notice that `vocab` should be a `list[str]` variable.


## Addition:
The idea of creating such functions arose due to the fact that the built-in functions do not add tokens/merges properly, when some tokens are deleted. That is why you can get more tokens after encoding the same text, even if the necessary tokens have been added.

```python
changer.add_tokens(list_of_tokens)
```
Adds the tokens from the list. The indexes will be filled automatically.

```python
changer.add_merges(list_of_merges)
```
Adds the merges from the list.


## "Get" functions:
```python
changer.get_overlapping_tokens(vocab)
```
Returns the intersection between the `tokenizer`'s vocabulary and the `vocab` variable. Notice that `vocab` should be a `dict` variable.

```python
changer.get_overlapping_megres(merges)
```
Returns the intersection between the `tokenizer`'s merges and the `merges` variable. Notice that `merges` should be a `list` variable.


## Saving:
```python
changer.save_tokenizer(path)
```
Saves the current state of the changed tokenizer. Additionally, it saves tokenizer configs into `path` folder (`./updated_tokenizer` by default).

```python
tokenizer = ch.updated_tokenizer()
```
Return the changed tokenizer.
