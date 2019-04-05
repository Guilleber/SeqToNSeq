import numpy as np
import math
from modules.attention import Attention
from modules.variational import Variational
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    """
    A standart LSTM encoder with eventually a variational layer on the last hidden state.
    """
    def __init__(self, params, embedding_full):
        super(Encoder, self).__init__()
        self.params = params
        self.embedding_full = embedding_full

        self.encoder = nn.LSTM(input_size=self.params["embedding_size"], hidden_size=self.params["encoder_hidden_size"], num_layers=self.params["num_encoder_layers"], batch_first=True, bidirectional=True, dropout=self.params["dropout"])
        self.hidden_bridge = nn.Linear(self.params["encoder_hidden_size"]*2, self.params["decoder_hidden_size"])
        self.cell_bridge = nn.Linear(self.params["encoder_hidden_size"]*2, self.params["decoder_hidden_size"])
        if self.params["VAE"]:
            self.hidden_var = Variational(self.params["decoder_hidden_size"])
            self.cell_var = Variational(self.params["decoder_hidden_size"])
            #self.linear_h1 = nn.Linear(1000, self.params["decoder_hidden_size"])
            #self.linear_c1 = nn.Linear(1000, self.params["decoder_hidden_size"])
            #self.linear_h2 = nn.Linear(self.params["decoder_hidden_size"], 1000)
            #self.linear_c2 = nn.Linear(self.params["decoder_hidden_size"], 1000)


    def forward(self, encoder_input, kld_coef=1.0, var=True):
        use_cuda = encoder_input.is_cuda
        is_training = self.training
        [batch_size, _] = encoder_input.size()
        memory_bank, encoder_final_state = self.encoder(self.embedding_full(encoder_input))

        encoder_final_hidden = t.cat([encoder_final_state[0][0:encoder_final_state[0].size(0):2].contiguous(), encoder_final_state[0][1:encoder_final_state[0].size(0):2].contiguous()], 2)
        encoder_final_cell = t.cat([encoder_final_state[1][0:encoder_final_state[1].size(0):2].contiguous(), encoder_final_state[1][1:encoder_final_state[1].size(0):2].contiguous()], 2)

        [layers, _, _] = encoder_final_cell.size()
        decoder_initial_hidden = self.hidden_bridge(encoder_final_hidden.view(layers*batch_size, self.params["encoder_hidden_size"]*2))

        if self.params["VAE"]:
            decoder_initial_hidden, kld_hidden = self.hidden_var(decoder_initial_hidden, var=var)
        decoder_initial_hidden = decoder_initial_hidden.view(layers, batch_size, self.params["decoder_hidden_size"])
        
        decoder_initial_cell = self.cell_bridge(encoder_final_cell.view(layers*batch_size, self.params["encoder_hidden_size"]*2))

        if self.params["VAE"]:
            decoder_initial_cell, kld_cell = self.cell_var(decoder_initial_cell, var=var)

        decoder_initial_cell = decoder_initial_cell.view(layers, batch_size, self.params["decoder_hidden_size"])

        if is_training and self.params["VAE"]:
            kld = kld_coef*(kld_hidden + kld_cell)
            kld.backward(retain_graph=True)

        decoder_initial_state = (decoder_initial_hidden, decoder_initial_cell)

        return memory_bank, decoder_initial_state


class RNNAutoEncoder(nn.Module):
    """
    A simple LSTM encoder-decoder which can be parameterized to include variational layers and GAN normalization.
    """
    def __init__(self, params):
        super(RNNAutoEncoder, self).__init__()
        
        self.params = params
         
        self.embedding = nn.Embedding(self.params["vocab_size"], self.params["embedding_size"])
        nn.init.kaiming_uniform(self.embedding.weight, mode='fan_in')
        self.dropout_emb = nn.Dropout(0.3)
        self.embedding_full = nn.Sequential(self.embedding, self.dropout_emb)

        self.encoder = Encoder(self.params, self.embedding_full)

        for name, p in self.encoder.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
            else:
                p.data.uniform_(-0.1, 0.1)
            if "input_to_logvar.bias" in name:
                p.data.uniform_(-10.0, -5.0)

        self.dropout_decoder = nn.Dropout(self.params["dropout"])
		
        self.decoders = nn.ModuleList()
        for i in range(self.params["nb_decoders"]):
            decoder = nn.LSTM(input_size=self.params["embedding_size"], hidden_size=self.params["decoder_hidden_size"], num_layers=self.params["num_decoder_layers"], batch_first=True, bidirectional=False, dropout=self.params["dropout"])
            for p in decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)
                else:
                    p.data.uniform_(-0.1, 0.1)
            self.decoders.append(decoder)

        self.attention = Attention(self.params["decoder_hidden_size"])
        self.normalizer = nn.Linear(self.params["decoder_hidden_size"], self.params["vocab_size"])
        self.normalizer.weight.data.uniform_(-0.1, 0.1)
        self.train()
        self.step = 0
        self.to_track = -1
        self.offset = 0


    def forward(self, encoder_input, decoder_input, go_idx=0, end_idx=0, max_decoding_loop=None, kld_coef=1.0, var=True):        
        use_cuda = encoder_input.is_cuda
        is_training = self.training
        [batch_size, _] = encoder_input.size()

        memory_bank, decoder_initial_state = self.encoder(encoder_input, kld_coef=kld_coef, var=var)
        
        all_dec_out = []
        all_dec_align = []
        for decoder in self.decoders:
            if is_training:
                [_, seq_len] = decoder_input.size()
                decoder_output, _ = decoder(self.embedding_full(decoder_input), decoder_initial_state)
                output = decoder_output.contiguous()
                output, align_vectors = self.attention(output, memory_bank)

                output = output.view(batch_size*seq_len, self.params["decoder_hidden_size"])
                output = self.normalizer(self.dropout_decoder(output))
                output = output.view(batch_size, seq_len, self.params["vocab_size"])
            else:
                previous_word = Variable(t.from_numpy(np.array([[go_idx] for _ in range(batch_size)]))).long()
                if use_cuda:
                    previous_word = previous_word.cuda()

                decoder_state = (decoder_initial_state[0], decoder_initial_state[1])
                outputs = []
                num_decoding_loop = 0
                align_vectors = []
                while(num_decoding_loop < max_decoding_loop):
                    num_decoding_loop += 1
                    decoder_step, decoder_state = decoder(self.embedding_full(previous_word), decoder_state)
                    decoder_step = decoder_step.contiguous()
                    decoder_step, align_vector = self.attention(decoder_step, memory_bank)
                    align_vectors.append(align_vector)
                    decoder_step = decoder_step.view(batch_size, self.params["decoder_hidden_size"])
                    decoder_step = self.normalizer(self.dropout_decoder(decoder_step))
                    decoder_step = decoder_step.view(batch_size, 1, self.params["vocab_size"])

                    outputs.append(decoder_step)
                    previous_word_np = np.array([[np.argmax(sentence[0])] for sentence in outputs[-1].data.cpu().numpy()])
                    previous_word = Variable(t.from_numpy(previous_word_np)).long()
                    if use_cuda:
                        previous_word = previous_word.cuda()

                output = t.cat(outputs, 1)
                align_vectors = t.cat(align_vectors, 1)
            all_dec_out.append(output)
            all_dec_align.append(align_vectors)

        return all_dec_out, all_dec_align, decoder_initial_state


    def trainer(self, optimizer, batcher):
        """
        Create the train operator for this network
        """
        #if batcher.epoch == 1:
            #self.f = open("./results/curr.track", 'w')

        def train_step(batch_size, use_cuda=True, full_metrics=False, kld_coef=1.0):
            self.step += 1
            batch = batcher.next_batch(batch_size, 'train', src_seq_len=60, trg_seq_len=20, dropout=0.0)
            epoch = batcher.epoch
            if batcher.offset_new_epoch != -1:
                self.offset = batcher.offset_new_epoch
                self.to_track = -1

            if use_cuda:
                batch = [Variable(t.from_numpy(var)).long().cuda() for var in batch]
            else:
                batch = [Variable(t.from_numpy(var)).long() for var in batch]

            [encoder_input, target, decoder_input] = batch
            optimizer.zero_grad()
            #x = t.index_select(target, 1, Variable(t.LongTensor([10])).cuda())
            #x = t.t(x)
            #x = x.data.cpu().numpy()
            #x = [[1 if k==batcher.word_to_idx['PAD'] else 0 for k in x[0]]]
            #x = Variable(t.LongTensor(x)).cuda()
			
            all_dec_out, all_dec_align, decoder_initial_state = self(encoder_input, decoder_input, kld_coef=kld_coef, var=True)
            [_, seq_len, _] = all_dec_out[0].size()
            all_dec_losses = []
            for i in range(self.params["nb_decoders"]):
                loss = F.cross_entropy(all_dec_out[i].view(batch_size*seq_len, self.params["vocab_size"]), target.view(batch_size*seq_len), reduce=False).view(batch_size, seq_len)
                all_dec_losses.append(t.unsqueeze(t.mean(loss, 1), 0))

            min_losses = t.min(t.cat(all_dec_losses, 0), 0)
            loss = t.mean(min_losses[0])

            if self.params["pen"]:
                dloss1 = F.cross_entropy(all_dec_out[0].view(batch_size*seq_len, self.params["vocab_size"]), t.max(all_dec_out[1].view(batch_size*seq_len, self.params["vocab_size"]).detach(), 1)[1], ignore_index=batcher.word_to_idx['PAD'], reduce=False).view(batch_size, seq_len)
                dloss1 = t.mean(dloss1, 1)
                pen1 = t.max(self.params["pen"]-dloss1, Variable(t.zeros(batch_size).float()).cuda())

                dloss2 = F.cross_entropy(all_dec_out[1].view(batch_size*seq_len, self.params["vocab_size"]), t.max(all_dec_out[0].view(batch_size*seq_len, self.params["vocab_size"]).detach(), 1)[1], ignore_index=batcher.word_to_idx['PAD'], reduce=False).view(batch_size, seq_len)
                dloss2 = t.mean(dloss2, 1)
                pen2 = t.max(self.params["pen"]-dloss2, Variable(t.zeros(batch_size).float()).cuda())
                #print(dloss1.data.cpu().numpy()) 
                #print(dloss2.data.cpu().numpy())

                loss = loss + t.mean(pen1) + t.mean(pen2)
            #else:
                #loss = t.mean(t.gather(t.cat(all_dec_losses, 0), 0, 
            output = all_dec_out[0]
            align_vectors = all_dec_align[0]

            if self.to_track >= 0:
                for k in min_losses[1].data.cpu().numpy()[self.offset:min(batch_size, self.offset+self.to_track+1)]:
                    self.f.write(str(k))
                    self.f.write(" ")
                self.to_track -= batch_size-self.offset
                self.offset = 0
                if self.to_track < 0:
                    self.f.write("\n")
   
            if self.step%50 == 0:
                print(t.mean(min_losses[1].float()).data.cpu().numpy())
                for i in range(self.params["nb_decoders"]):
                    print(batcher.get_sentences([[np.argmax(word) for word in all_dec_out[i][0].data.cpu().numpy()]]))

            loss.backward()
            t.nn.utils.clip_grad_norm(self.parameters(), 0.5)
            optimizer.step()

            metrics = {}

            if full_metrics:
                output_sentences = [[np.argmax(word) for word in sentence] for sentence in output.data.cpu().numpy()]
                target_sentences = target.data.cpu().numpy()
                input_sentences = encoder_input.data.cpu().numpy()
                true_words = 0
                total_words = 0
                special_idxs = [batcher.word_to_idx[batcher.pad_token]]
                for s in range(len(target_sentences)):
                    for w in range(len(target_sentences[s])):
                        if not(target_sentences[s][w] in special_idxs):
                            total_words += 1
                        if target_sentences[s][w] == output_sentences[s][w] and target_sentences[s][w] not in special_idxs:
                            true_words += 1

                target_sentences = batcher.get_sentences(target_sentences)
                output_sentences = batcher.get_sentences(output_sentences)
                input_sentences = batcher.get_sentences(input_sentences)
                metrics = compute_metrics(target_sentences, output_sentences, input_sentences, align_vectors.data.cpu().numpy(), decoder_initial_state[1].data.cpu().numpy())
                metrics["acc"] = (true_words/total_words)*100

            metrics["epoch"] = batcher.epoch
            return loss, metrics
        return train_step


    def tester(self, batcher):
        """
        Create the test operator for this network
        """

        def test(batch_size, use_cuda=True, max_decoding_loop=20, full_metrics=False, full_batch=False, test_set='test'):
            self.eval()
            batch = batcher.full_batch(test_set) if full_batch else batcher.next_batch(batch_size, test_set, trg_seq_len=max_decoding_loop, src_seq_len=60)

            if use_cuda:
                batch = [Variable(t.from_numpy(var)).long().cuda() for var in batch]
            else:
                batch = [Variable(t.from_numpy(var)).long() for var in batch]

            encoder_input, target, decoder_input = batch
            
            all_dec_out, all_dec_align, decoder_initial_state = self(encoder_input, None, go_idx=batcher.word_to_idx[batcher.go_token], end_idx=batcher.word_to_idx[batcher.end_token], max_decoding_loop=max_decoding_loop)
            output = all_dec_out[0]
            align_vectors = all_dec_align[0]
            loss = F.cross_entropy(output.view(-1, self.params["vocab_size"]), target.view(-1))

            metrics = {}

            if full_metrics:
                target_sentences = batcher.get_sentences(target.data.cpu().numpy())
                input_sentences = batcher.get_sentences(encoder_input.data.cpu().numpy())
                output_sentences = batcher.get_sentences([[np.argmax(word) for word in sentence] for sentence in output.data.cpu().numpy()])
                metrics = compute_metrics(target_sentences, output_sentences, input_sentences, align_vectors.data.cpu().numpy(), decoder_initial_state[1].data.cpu().numpy())
            metrics["epoch"] = batcher.epoch

            self.train()
            return loss, metrics
        return test


    def translator(self, batcher):
        """
        Create the translation operator for this network
        """

        def translate(input_f, output_f, use_cuda=True, max_decoding_loop=20, dec=0):
            self.eval()
            def to_call(input_batch):
                if use_cuda:
                    input_batch = Variable(t.from_numpy(input_batch)).long().cuda()
                else:
                    input_batch = Variable(t.from_numpy(input_batch)).long()

                all_dec_out, _, _ = self(input_batch, None, go_idx=batcher.word_to_idx[batcher.go_token], end_idx=batcher.word_to_idx[batcher.end_token], max_decoding_loop=max_decoding_loop)
                output = all_dec_out[dec]

                output_sentences = [' '.join(sentence) for sentence in batcher.get_sentences([[np.argmax(word) for word in sentence] for sentence in output.data.cpu().numpy()])]
                output_f.write('\n'.join(output_sentences) + '\n')

                return

            batch = batcher.full_file(64, input_f, to_call, max_src_len=60)

            self.train()
            return
        return translate


def compute_metrics(target_sentences, output_sentences, input_sentences, align_vectors, decoder_initial_state):
    """
    Compute a lot of metrics on the current iteration
    """
    metrics = {}
    bleu = [sentence_bleu([target_sentences[i]], output_sentences[i]) for i in range(len(target_sentences))]
    target_bigrams = [[(sentence[i], sentence[i+1]) for i in range(len(sentence)-1)] for sentence in target_sentences]
    output_bigrams = [[(sentence[i], sentence[i+1]) for i in range(len(sentence)-1)] for sentence in output_sentences]
    rouge1precision = [precision(set(target_sentences[i]), set(output_sentences[i])) for i in range(len(target_sentences))]
    rouge1recall = [recall(set(target_sentences[i]), set(output_sentences[i])) for i in range(len(target_sentences))]
    rouge2precision = [precision(set(target_bigrams[i]), set(output_bigrams[i])) for i in range(len(target_sentences))]
    rouge2recall = [recall(set(target_bigrams[i]), set(output_bigrams[i])) for i in range(len(target_sentences))]
    metrics["bleu_sample"] = bleu[0]
    metrics["rouge1precision_sample"] = rouge1precision[0]
    metrics["rouge1recall_sample"] = rouge1recall[0]
    metrics["rouge2precision_sample"] = rouge2precision[0]
    metrics["rouge2recall_sample"] = rouge2recall[0]
    metrics["bleu_batch"] = np.mean([b for b in bleu if b!=None])
    metrics["rouge1precision_batch"] = np.mean([r for r in rouge1precision if r!=None])
    metrics["rouge1recall_batch"] = np.mean([r for r in rouge1recall if r!=None])
    metrics["rouge2precision_batch"] = np.mean([r for r in rouge2precision if r!=None])
    metrics["rouge2recall_batch"] = np.mean([r for r in rouge2recall if r!=None])
    metrics["target_sample"] = target_sentences[0]
    print(target_sentences[0])
    #if len(target_sentences) > 1:
        #metrics["target_sample_2"] = target_sentences[1]
    metrics["output_sample"] = output_sentences[0]
    print(output_sentences[0])
    #if len(output_sentences) > 1:
        #metrics["output_sample_2"] = output_sentences[1]
    metrics["input_sample"] = input_sentences[0]
    #metrics["attention"] = align_vectors[0]
    #metrics["middle_vector"] = decoder_initial_state[0][0]
    #metrics["mu"] = np.mean(decoder_initial_state[0])
    #metrics["std"] = np.std(decoder_initial_state[0])
    #metrics["r1"] = 2*metrics["rouge1precision_batch"]*metrics["rouge1recall_batch"]/(metrics["rouge1precision_batch"]+metrics["rouge1recall_batch"])
    return metrics
