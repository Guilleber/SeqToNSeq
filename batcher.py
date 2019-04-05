import numpy as np
import json


class Batcher:
    def __init__(self, path_to_data):
        self.data_parts = ['train']
        self.src_files = [open(path_to_data + part + '.src') for part in self.data_parts]
        self.trg_files = [open(path_to_data + part + '.trg') for part in self.data_parts]
        
        self.vocab_files = [path_to_data + 'word_to_idx.json', path_to_data + 'idx_to_word.json']

        self.blind_token = ''
        self.pad_token = 'PAD'
        self.go_token = 'GO'
        self.end_token = 'EOS'
        self.unk_token = 'UNK'

        self.epoch = 1
        self.offset_new_epoch = 0
        self.begin = 1

        self.max_number_word = 50000

        try:
            with open(self.vocab_files[0], 'r') as f:
                self.word_to_idx = json.load(f)
            with open(self.vocab_files[1], 'r') as f:
                self.idx_to_word = json.load(f)
        except (IOError, ValueError):
            src_data = [[sentence.split() for sentence in f.read().split('\n')] for f in self.src_files]
            trg_data = [[sentence.split() for sentence in f.read().split('\n')] for f in self.trg_files]
            for f in self.src_files:
                f.seek(0)
            for f in self.trg_files:
                f.seek(0)

            vocab = dict()
            for sentence in src_data[0]:
                for word in sentence:
                    if word in vocab.keys():
                        vocab[word] = vocab[word] + 1
                    else:
                        vocab[word] = 1

            for sentence in trg_data[0]:
                for word in sentence:
                    if word in vocab.keys():
                        vocab[word] = vocab[word] + 1
                    else:
                        vocab[word] = 1

            vocab_list = vocab.items()
            vocab_list = sorted(vocab_list, key=lambda x: x[1], reverse=True)
            vocab_list = vocab_list[:self.max_number_word]
            vocab_list.append((self.pad_token, 1))
            vocab_list.append((self.end_token, 1))
            vocab_list.append((self.go_token, 1))
            #vocab_list.append((self.unk_token, 1))
            self.word_to_idx = {word: i for i, (word, _) in enumerate(vocab_list)}
            self.idx_to_word = [word for _, (word, _) in enumerate(vocab_list)]
            with open(self.vocab_files[0], 'w') as f:
                json.dump(self.word_to_idx, f)
                f.close()
            with open(self.vocab_files[1], 'w') as f:
                json.dump(self.idx_to_word, f)
                f.close()

        self.vocab_size = len(self.idx_to_word)


    def next_batch(self, batch_size, part, trg_seq_len=None, src_seq_len=None, dropout=0.0):
        """
        Returns the next minibatch of sentences following the order in the original file. If there is not enough sentences left at the end of the file, this will return at the begining and increase the epoch attribute.

        Outputs have the size (batch_size x seq_len). seq_len is equal to trg_seq_len if given else it is set automaticaly to be the maximum length of the sentences in the batch

        Args:
            batch_size: The size of the output minibatch
            part: The part of the corpora we want the minibatch to be extracted from among ["train", "dev", "test"]
            trg_seq_len: Used to force the output sentences to be of a precise length

        Returns:
            A tupple of 3 elements corresponding respectively to the source, the target and the decoder input which is used for teacher forcing.
        """

        if part == 'train':
            part_idx = 0
        elif part == 'dev':
            part_idx = 1
        elif part == 'test':
            part_idx = 2

        src_sentences = []
        trg_sentences = []

        src_file = self.src_files[part_idx]
        trg_file = self.trg_files[part_idx]
        if not self.begin:
            self.offset_new_epoch = -1
        self.begin = 0
        for i in range(batch_size):
            src_line = src_file.readline()
            trg_line = trg_file.readline()
            if not src_line:
                assert not trg_line
                self.offset_new_epoch = i
                src_file.seek(0)
                trg_file.seek(0)
                src_line = src_file.readline()
                trg_line = trg_file.readline()
                if part_idx == 0:
                    self.epoch += 1
            src_sentences.append(src_line.split() + [self.end_token])
            trg_sentences.append(trg_line.split() + [self.end_token])

        src_sentences = [[self.word_to_idx[word] if word in self.word_to_idx.keys() else self.word_to_idx['<unk>'] for word in sentence] for sentence in src_sentences]
        trg_sentences = [[self.word_to_idx[word] if word in self.word_to_idx.keys() else self.word_to_idx['<unk>'] for word in sentence] for sentence in trg_sentences]

        max_src_len = src_seq_len if src_seq_len else np.max([len(sentence) for sentence in src_sentences])
        max_trg_len = trg_seq_len if trg_seq_len else np.max([len(sentence) for sentence in trg_sentences])

        src_batch = [[self.word_to_idx[self.pad_token] if i < (max_src_len - len(sentence)) else sentence[i - (max_src_len - len(sentence))] for i in range(max_src_len)] for sentence in src_sentences]
        trg_batch = [[sentence[i] if i < len(sentence) else self.word_to_idx[self.pad_token] for i in range(max_trg_len)] for sentence in trg_sentences]
        dec_input_batch = [[self.word_to_idx[self.go_token]] + sentence[:-1] for sentence in trg_batch]

        drop_prob = np.random.rand(len(dec_input_batch), len(dec_input_batch[0]))
        pad_idx = self.word_to_idx[self.pad_token]
        dec_input_batch = [[dec_input_batch[i][j] if drop_prob[i][j] >= dropout else pad_idx for j in range(len(dec_input_batch[i]))] for i in range(len(dec_input_batch))]

        return np.array(src_batch), np.array(trg_batch), np.array(dec_input_batch)


    def full_batch(self, part):
        """
        Similar to next_batch but returns the whole batch instead of a minibatch
        """

        """In need of optimization -> Do not use !!!"""
        if part == 'train':
            part_idx = 0
        elif part == 'dev':
            part_idx = 1
        elif part == 'test':
            part_idx = 2

        src_file = self.src_files[part_idx]
        trg_file = self.trg_files[part_idx]

        src_sentences = src_file.readlines()
        trg_sentences = trg_file.readlines()

        src_sentences = [sentence.split() + [self.end_token] for sentence in src_sentences]
        trg_sentences = [sentence.split() + [self.end_token] for sentence in trg_sentences]

        src_sentences = [[self.word_to_idx[word] for word in sentence] for sentence in src_sentences]
        trg_sentences = [[self.word_to_idx[word] for word in sentence] for sentence in trg_sentences]

        max_src_len = np.max([len(sentence) for sentence in src_sentences])
        max_trg_len = np.max([len(sentence) for sentence in trg_sentences])

        src_batch = [[self.word_to_idx[self.pad_token] if i < (max_src_len - len(sentence)) else sentence[i - (max_src_len - len(sentence))] for i in range(max_src_len)] for sentence in src_sentences]
        trg_batch = [[sentence[i] if i < len(sentence) else self.word_to_idx[self.pad_token] for i in range(max_trg_len)] for sentence in trg_sentences]
        dec_input_batch = [[self.word_to_idx[self.go_token]] + sentence[:-1] for sentence in trg_batch]

        return np.array(src_batch), np.array(trg_batch), np.array(dec_input_batch)


    def full_file(self, max_batch_size, src_file, to_call, max_src_len=None):
        """
        Efficient way to make the entire batch go through some model. This function will feed the to_call function minibatch-by-minibatch.

        Args:
            max_batch_size: The size of the minibatchs that are passed
            src_file: The file which we want to pass through the network
            to_call: A callback function that will be called with each minibatch

        Returns:
            None
        """
        end = False
        while not end:
            src_sentences = []
            for _ in range(max_batch_size):
                src_line = src_file.readline()
                if not src_line:
                    end = True
                else:
                    src_sentences.append(src_line.split() + [self.end_token])

            src_sentences = [[self.word_to_idx[word] if word in self.word_to_idx.keys() else self.word_to_idx['<unk>'] for word in sentence] for sentence in src_sentences]

            if not max_src_len:
                max_src_len = np.max([len(sentence) for sentence in src_sentences])

            src_batch = [[self.word_to_idx[self.pad_token] if i < (max_src_len - len(sentence)) else sentence[i - (max_src_len - len(sentence))] for i in range(max_src_len)] for sentence in src_sentences]

            to_call(np.array(src_batch))

        return



    def get_sentences(self, batch):
        """
        Recompute the natural language sentences from a batch of numbers

        Args:
            batch: A batch of numbers

        Returns:
            A list of list of words of same dimensions than the input batch
        """
        #to_remove = []
        to_remove = [self.word_to_idx[self.go_token], self.word_to_idx[self.end_token], self.word_to_idx[self.pad_token]]
        return [[str(self.idx_to_word[idx]) for idx in sentence if idx not in to_remove] for sentence in batch]
        
        
