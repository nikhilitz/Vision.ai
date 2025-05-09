# Path: Vision.ai/src/captioning/vocabulary.py

import nltk
from collections import Counter
import torch  # Imported but not strictly used in the provided class code

# Download tokenizer data if not already present
# This might cause issues if running offline. Consider a check or handle in setup script.
# nltk.download('punkt') # Often handled once during setup or in init files, or train/inference scripts

class Vocabulary:
    def __init__(self, freq_threshold=5):
        """
        Initialize the Vocabulary object.
        :param freq_threshold: Minimum number of times a word must appear to be included in vocab
        """
        self.freq_threshold = freq_threshold

        # Define Special tokens - standard convention
        self.pad_token = '<pad>'  # Used for padding sequences to the same length
        self.sos_token = '<sos>'  # Start Of Sentence token
        self.eos_token = '<eos>'  # End Of Sentence token
        self.unk_token = '<unk>'  # Unknown token for rare/unseen words

        # Create mapping dictionaries: token string to integer index and vice versa
        # Assigning specific indices to special tokens (0, 1, 2, 3) - consistent with collate_fn pad_value=0
        self.stoi = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        # Inverse mapping from integer index back to token string
        self.itos = {idx: token for token, idx in self.stoi.items()}

        # Counter to store word frequencies during vocabulary building
        self.word_freq = Counter()

    def __len__(self):
        """Returns the total number of unique tokens in the vocabulary."""
        return len(self.stoi)

    def tokenize(self, text):
        """
        Tokenizes a cleaned text string into words using nltk word tokenizer.
        Assumes input text is already cleaned (lowercased, punctuation removed, etc. by utils.clean_caption).
        """
        if not isinstance(text, str):
            # print(f"Warning: tokenize received non-string input: {type(text)}")
            return []
        # NLTK word_tokenize is sensitive to punctuation unless removed beforehand.
        # We rely on clean_caption in utils.py for removing punctuation before numericalization/building vocab.
        # Lowercasing might be redundant if clean_caption already lowercased, but doesn't hurt.
        return nltk.tokenize.word_tokenize(text.lower())  # Keeping .lower() just in case

    def build_vocabulary(self, sentence_dict):
        """
        Builds the vocabulary from a dictionary of image_id → [list of cleaned captions].
        Only includes words with frequency >= threshold, excluding special tokens.
        :param sentence_dict: Dictionary from load_captions (assuming captions are cleaned strings)
        """
        print("Building vocabulary...")
        # First pass: count word frequencies across all cleaned captions
        if not isinstance(sentence_dict, dict):
            print(f"Error: build_vocabulary received non-dict input: {type(sentence_dict)}")
            return

        processed_sentences_count = 0
        for caption_list in sentence_dict.values():
            if isinstance(caption_list, list):
                for sentence in caption_list:
                    # Ensure the sentence is a non-empty string before tokenizing
                    if isinstance(sentence, str) and sentence:
                        tokens = self.tokenize(sentence)
                        # Only update frequency if tokens were produced
                        if tokens:
                            self.word_freq.update(tokens)
                            processed_sentences_count += 1

        # Second pass: add words meeting frequency threshold to the vocabulary mappings
        # Start assigning indices from the next available integer after special tokens
        idx = len(self.stoi)  # Starts at 4

        # Iterate through words and their frequencies
        # word_freq.items() returns (word, freq) pairs
        for word, freq in self.word_freq.items():
            # If word frequency meets threshold AND it's not already a special token
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx  # Assign new index to word
                self.itos[idx] = word  # Assign word to new index
                idx += 1  # Increment index for the next word

        print(f"Vocabulary built with {len(self)} unique tokens (including special tokens) from {processed_sentences_count} sentences.")
        if self.unk_token in self.stoi:
            # Getting frequency of words *mapped* to <unk> is more involved
            # This counts the frequency of the literal string '<unk>' if it appeared in data
            unk_freq_in_data = self.word_freq.get(self.unk_token, 0)
            # print(f"Note: Frequency of literal '{self.unk_token}' string in original dataset: {unk_freq_in_data}")
        # Optional: print(f"Most common words: {self.word_freq.most_common(20)}")
        # Optional: print(f"Size before threshold: {len(self.word_freq)}. Size after threshold: {len(self)}")

    def numericalize(self, text):
        """
        Converts a cleaned sentence string into a list of token indices.
        Unknown words are mapped to the <unk> token index.
        :param text: A cleaned caption string.
        :return: A list of integer indices.
        """
        if not isinstance(text, str):
            # print(f"Warning: numericalize received non-string input: {type(text)}")
            return []

        tokens = self.tokenize(text)
        if not tokens:
            return []  # Return empty list if tokenization resulted in no tokens

        # Use .get() with a default value to handle tokens not in the vocabulary
        # The default value is the index of the <unk> token
        return [self.stoi.get(token, self.stoi[self.unk_token]) for token in tokens]

    # Optional: Add a method to get the state for safer saving
    def get_state(self):
        return {'stoi': self.stoi, 'itos': self.itos, 'freq_threshold': self.freq_threshold}

    # Optional: Add a method to load state and reconstruct
    @classmethod
    def from_state(cls, state):
        vocab = cls(freq_threshold=state.get('freq_threshold', 5))  # Use threshold from state or default
        vocab.stoi = state['stoi']
        vocab.itos = state['itos']
        # Note: word_freq is NOT saved/loaded as it's only needed during build_vocabulary
        return vocab
