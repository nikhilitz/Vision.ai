import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.captioning.vocabulary import Vocabulary

class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.captions = [
            "a dog is running in the park",
            "a cat is sitting on the mat",
            "the dog is playing"
        ]
        self.vocab = Vocabulary(freq_threshold=2)
        self.vocab.build_vocabulary(self.captions)

    def test_tokenize(self):
        text = "A dog is playing"
        tokens = self.vocab.tokenize(text)
        self.assertEqual(tokens, ['a', 'dog', 'is', 'playing'])

    def test_build_vocabulary(self):
        expected_words = ['<pad>', '<sos>', '<eos>', '<unk>', 'a', 'dog', 'is', 'the']
        for word in expected_words:
            self.assertIn(word, self.vocab.stoi)
        self.assertNotIn('cat', self.vocab.stoi)

    def test_numericalize(self):
        text = "a dog is playing in the park"
        expected_seq = [
            self.vocab.stoi['a'],
            self.vocab.stoi['dog'],
            self.vocab.stoi['is'],
            self.vocab.stoi['<unk>'],  # 'playing'
            self.vocab.stoi['<unk>'],  # 'in'
            self.vocab.stoi['the'],
            self.vocab.stoi['<unk>']   # 'park'
        ]
        num_seq = self.vocab.numericalize(text)
        self.assertEqual(num_seq, expected_seq)

if __name__ == '__main__':
    unittest.main()
