# Tokens-Deletion
Python script for deletion of the unwanted tokens from the existing tokenizer.

The solution was tested on Llama3-8B tokenizer.

-----
# Usage:

```python
changer = TokenizerChanger(tokenizer)
```
Create the object of ```TokenizerChanger``` class that requires an existing tokenizer that will be changed, e.g. ```PreTrainedTokenizerFast``` class from 🤗 Tokenizers library.

```python
changer.delete_k_least_frequent_tokens(k=1000)
```
This function deletes k most frequent tokens.

```python
changer.delete_unwanted_tokens(list_of_unwanted_tokens)
```
This function deletes all tokens from ```list_of_unwanted_tokens``` from the tokenizer.

```python
changer.save_tokenizer(path)
```
Saves the current state of the changed tokenizer. Additionally, it saves tokenizer configs into ```path``` folder (```./updated_tokenizer``` by default).

```python
tokenizer = ch.updated_tokenizer()
```
Return the changed tokenizer.
