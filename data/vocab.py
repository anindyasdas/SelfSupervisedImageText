import spacy as spacy
import torch
import csv
import re
spacy.prefer_gpu()

class BaseTokenizer(object):
    def __init__(self):
        pass

    def tokenize(self, sentence):
        raise NotImplementedError("Method must be overitten in subclass")


class SpacyTokenizer(BaseTokenizer):
    def __init__(self):
        super(SpacyTokenizer, self).__init__()
        self.spacy_tokenizer = spacy.load('en_core_web_sm', disable=["parser", "tagger", "ner"])
        print('##############loading SpacyTokenizer###########')

    
    def tokenize(self, sentence):
        #spacy_tokenizer = spacy.load('en_core_web_sm', disable=["parser", "tagger", "ner"])

        tokens = self.spacy_tokenizer(sentence)
        tokens = [token.text for token in tokens]
        return tokens


class WhitespaceTokenizer(BaseTokenizer):
    def __init__(self):
        super(WhitespaceTokenizer, self).__init__()
        print('############loading WhitespaceTokenizer###########')

    def tokenize(self, sentence):
        tokens = sentence.split()
        return tokens


class BaseVocabBuilder(object):

    def __init__(self, tokenizer=SpacyTokenizer()):
        self.tokenizer = tokenizer
        self.t2i = {}
        self.i2t = []

        self.eos = '<EOS>'
        self.sos = '<SOS>'
        self.pad = '<PAD>'
        self.unk = '<UNK>'

        self.add_token(self.pad)
        self.add_token(self.sos)
        self.add_token(self.eos)
        self.add_token(self.unk)

    def vocab_size(self):
        return len(self.i2t)

    def eos_pos(self):
        return self.t2i[self.eos]

    def pad_pos(self):
        return self.t2i[self.pad]

    def sos_pos(self):
        return self.t2i[self.sos]

    def add_token(self, token):
        if token not in self.t2i:
            self.i2t.append(token)
            self.t2i[token] = len(self.t2i)

    def encode_sentences(self, sentences, max_length=None):
        """
        Map each word in a sentence to it's id
        :param sentences: list of sentences to encode
        :param max_length: max length of the sentences
        :return: encoded sentences, length of each sentence(excluding padding, including <SOS> and <EOS> token)
        """
        splits = [self.tokenizer.tokenize(sentence) for sentence in sentences]

        lengths = [len(tokens) + 2 for tokens in splits]
        if not max_length:
            max_length = max(lengths)

        batch_positions = []
        for split in splits:
            positions = []
            positions.append(self.t2i[self.sos])
            #positions += [self.t2i[token] for token in split]
            for token in split:
                try:
                    code=self.t2i[token]
                except:
                    code =self.t2i[self.unk]
                positions.append(code)
                
            positions.append(self.t2i[self.eos])
            while len(positions) < max_length:
                positions.append(self.t2i[self.pad])
            batch_positions.append(positions)

        batch_positions = torch.LongTensor(batch_positions)
        lengths = torch.LongTensor(lengths)

        return batch_positions, lengths

    def decode_positions(self, batch_positions):
        batch_sentences = []
        for positions in batch_positions:
            sentence = ''
            for i, position in enumerate(positions):
                if i == 0: continue
                if position == self.sos_pos(): continue
                if position == self.eos_pos(): break
                sentence += self.i2t[int(position)] + ' '
            batch_sentences.append(sentence)
        return batch_sentences

    def load_text(self, filepath):
        """
        Loads all sentences from the file and adds all tokens to the vocabulary.
        :param filepath: file to load the vocab from
        :return:
        """
        sentences = self.read_sentences(filepath)
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            [self.add_token(token) for token in tokens]

    def load_texts(self, listof_filepaths):
        """
        Load tokens from a list of files.
        :param listof_filepaths: a list of files to load the tokens from
        :return:
        """
        for filepath in listof_filepaths:
            self.load_text(filepath)

    def read_sentences(self, filepath):
        """
        Returns a list of sentences. The sentences will be tokenized and the tokens will be added to the vocab in
        load_text(self, filepath).
        Since each dataset might store the captions/text in a different format, this method
        is supposed to be implemented in a specific subclass for each dataset.
        :param filepath: path of the file to read the senteces from
        :return: list of sentences
        """
        raise NotImplementedError("Method must be overitten in subclass")


class ShapesVocabBuilder(BaseVocabBuilder):
    def __init__(self):
        super(ShapesVocabBuilder, self).__init__()

    def read_sentences(self, filepath):
        sentences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0: continue
                sentences.append(line[1])
        return sentences


class BirdsVocabBuilder(BaseVocabBuilder):
    def __init__(self):
        super(BirdsVocabBuilder, self).__init__()

    def read_sentences(self, filepath):
        sentences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', line)
                sentences.append(line.strip())
        return sentences

class BillionVocabBuilder(BaseVocabBuilder):
    def __init__(self):
        super(BillionVocabBuilder, self).__init__()

    def read_sentences(self, filepath):
        sentences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', line)
                sentences.append(line.strip().lower())
        return sentences

class FlowersVocabBuilder(BaseVocabBuilder):
    def __init__(self):
        super(FlowersVocabBuilder, self).__init__()

    def read_sentences(self, filepath):
        sentences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = re.sub(r'[^A-Za-z0-9 ,.?!-]+', ' ', line)
                sentences.append(line.strip())
        return sentences
