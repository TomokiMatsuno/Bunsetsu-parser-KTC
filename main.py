import numpy as np
import dynet as dy
import glob

from collections import defaultdict
from read_file import *
from paths import *
from file_reader import DataFrameKtc
from file_reader import DataFrameUD


files = glob.glob(path2KTC + 'syn/*ED.*')
#files = [path2KTC + 'syn/9501ED.KNP']

print(path2UD)

print(files)

df = DataFrameKtc

train_sents = []
for file in files[0:-1]:
    print('[train] reading this file: ', file)
    lines = df.file2lines(df, file, ' ', 'euc-jp')
    train_sents.extend(df.lines2sents(df, lines))
wd, cd, bpd = df.sents2dicts(df, train_sents)

wd.freeze()
cd.freeze()
bpd.freeze()

dev_sents = []
for file in files[-1:]:
    print('[dev] reading this file: ', file)
    lines = df.file2lines(df, file, ' ', 'euc-jp')
    dev_sents.extend(df.lines2sents(df, lines))

for sent in dev_sents:
    for w in sent.word_forms:
        wd.add_entry(w)
    for c in sent.char_forms:
        cd.add_entry(c)
    for bp in sent.word_biposes:
        bpd.add_entry(bp)



train_word_seqs, train_char_seqs, train_word_bipos_seqs, train_chunk_bi_seqs = df.sents2ids([wd, cd, bpd], train_sents)

TRAIN = True
epoc = 100
train_iter = 1
batch_size = 10
show_loss_every = 100
show_acc_every = 100

LAYERS = 1
HIDDEN_DIM = 200
INPUT_DIM = HIDDEN_DIM * 2

dev_word_seqs, dev_char_seqs, dev_word_bipos_seqs, dev_chunk_bi_seqs = df.sents2ids([wd, cd, bpd], dev_sents)

###Neural Network

VOCAB_SIZE = len(cd.i2x) + 1
BIPOS_SIZE = len(bpd.i2x) + 1

pc = dy.ParameterCollection()

l2rlstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
r2llstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)

params = {}
params["lp_c"] = pc.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
params["lp_bp"] = pc.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))

params["R"] = pc.add_parameters((BIPOS_SIZE, HIDDEN_DIM * 2))
params["bias"] = pc.add_parameters((BIPOS_SIZE))

params["R_bi_b"] = pc.add_parameters((2, INPUT_DIM))
# params["R_bi_b"] = pc.add_parameters((2, INPUT_DIM * 2))
params["bias_bi_b"] = pc.add_parameters((2))


def linear_interpolation(bias, R, inputs):
    ret = bias
    for i in range(len(inputs)):
        ret += R * inputs[i]
    return ret



def do_one_sentence(l2rlstm, r2llstm, char_seq, bipos_seq):
    num_cor = 0
    num_cor_bi = 0
    num_cor_pos = 0

    dy.renew_cg()
    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    R = dy.parameter(params["R"])
    bias = dy.parameter(params["bias"])
    lp_c = params["lp_c"]

    s_l2r = s_l2r_0
    s_r2l = s_r2l_0

    loss = []

    cembs = [lp_c[c] for c in char_seq]

    l2r_outs = s_l2r.add_inputs(cembs)
    r2l_outs = s_r2l.add_inputs(reversed(cembs))
    lstm_outs = [dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()]) for i in range(len(l2r_outs))]

    for i in range(len(char_seq)):
        probs = dy.softmax(R*lstm_outs[i] + bias)
        loss.append(-dy.log(dy.pick(probs, bipos_seq[i])))

        if(not TRAIN):
            chosen = np.argmax(probs.npvalue())
            #print(bpd.i2x[chosen], " ", bpd.i2x[bipos_seq[i]])
            if(chosen == bipos_seq[i]):
                num_cor += 1
            if((bpd.i2x[chosen])[0] == (bpd.i2x[bipos_seq[i]])[0]):
                num_cor_bi += 1
            if ((bpd.i2x[chosen])[1:-1] == (bpd.i2x[bipos_seq[i]])[1:-1]):
                num_cor_pos += 1



    loss = dy.esum(loss)
    return loss, num_cor, num_cor_bi, num_cor_pos, lstm_outs


def get_wh_seq(char_seq, bipos_seq, bi_b_seq):
    ret = []
    w_1st_chars = []

    tmp = []
    lp_c = params["lp_c"]
    lp_bp = params["lp_bp"]

    i = 0
    tmp.append(dy.concatenate([lp_c[char_seq[i]], lp_bp[bipos_seq[i]]]))
    w_1st_chars.append(i)
    i += 1
    while i < len(char_seq):
        if bpd.i2x[bipos_seq[i]][0] != "B":
            tmp.append(dy.concatenate([lp_c[char_seq[i]], lp_bp[bipos_seq[i]]]))
            i += 1
        else:
            ret.append(tmp)
            tmp.clear()
            tmp.append(dy.concatenate([lp_c[char_seq[i]], lp_bp[bipos_seq[i]]]))
            w_1st_chars.append(i)
            i += 1
    return ret, w_1st_chars


def bi_b(wh_seq, w_1st_chars, bi_b_seq):
    num_cor = 0

    R_bi_b = dy.parameter(params["R_bi_b"])
    bias_bi_b = dy.parameter(params["bias_bi_b"])
    loss = []
    for i in range(len(wh_seq)):
        probs = dy.softmax(linear_interpolation(bias_bi_b, R_bi_b, wh_seq[i]))
        loss.append(-dy.log(dy.pick(probs, bi_b_seq[w_1st_chars[i]])))

        if not TRAIN:
            chosen = np.asscalar(np.argmax(probs.npvalue()))
            if(chosen == bi_b_seq[w_1st_chars[i]]):
                # print(chosen, " ", bi_b_seq[w_1st_chars[i]])
                num_cor += 1
    loss = dy.esum(loss)
    return loss, num_cor


def bi_bunsetsu(lstmout, bi_b_seq):
    num_cor = 0

    R_bi_b = dy.parameter(params["R_bi_b"])
    bias_bi_b = dy.parameter(params["bias_bi_b"])
    loss = []
    for i in range(len(bi_b_seq)):
        probs = dy.softmax(R_bi_b * lstmout[i] + bias_bi_b)
        loss.append(-dy.log(dy.pick(probs, bi_b_seq[i])))

        if not TRAIN:
            chosen = np.asscalar(np.argmax(probs.npvalue()))
            if(chosen == bi_b_seq[i]):
                # print(chosen, " ", bi_b_seq[w_1st_chars[i]])
                num_cor += 1
    loss = dy.esum(loss)
    return loss, num_cor



def train(l2rlstm, r2llstm, char_seqs, bipos_seqs, bi_b_seqs):
    trainer = dy.SimpleSGDTrainer(pc)
    losses = []
    for it in range(train_iter):
        for i in (range(len(char_seqs))):
            idx = i if not TRAIN else np.random.randint(len(char_seqs))
            if len(char_seqs[idx]) == 0 or len(bi_b_seqs[idx]) == 0:
                continue
            loss, _, _, _, lstmout = do_one_sentence(l2rlstm, r2llstm, char_seqs[idx], bipos_seqs[idx])
            losses.append(loss)
            if(i % batch_size):
                loss_value = loss.value()
                loss.backward()
                trainer.update()
            if i % show_loss_every == 0 and i != 0:
                print(i, " bipos loss")
                print(loss_value)

            seq_wh, w_1st_chars = get_wh_seq(char_seqs[idx], bipos_seqs[idx], bi_b_seqs[idx])
            if(len(seq_wh) == 0):
                continue
            # loss_bi_b, _ = bi_b(seq_wh, w_1st_chars, bi_b_seqs[i])
            loss_bi_bunsetsu, _ = bi_bunsetsu(lstmout, bi_b_seqs[idx])

            if i % batch_size:
                # loss_bi_b_value = loss_bi_b.value()
                # loss_bi_b.backward()
                loss_bi_bunsetsu_value = loss_bi_bunsetsu.value()
                loss_bi_bunsetsu.backward()
                trainer.update()
            if i % show_loss_every == 0 and i != 0:
                # print(i, " bi_b loss")
                # print(loss_bi_b_value)
                print(i, " bi_bunsetsu loss")
                print(loss_bi_bunsetsu_value)





def dev(l2rlstm, r2llstm, char_seqs, bipos_seqs, bi_b_seqs):
    num_tot = 0
    num_tot_cor = 0

    num_tot_cor_bi = 0
    num_tot_cor_pos = 0

    num_tot_bi_b = 0
    num_tot_cor_bi_b = 0
    for i in range(len(char_seqs)):
        if(len(char_seqs[i]) == 0):
            continue
        # loss, num_cor, num_cor_bi, num_cor_pos = do_one_sentence(l2rlstm, r2llstm, char_seqs[i], bipos_seqs[i])
        loss, num_cor, num_cor_bi, num_cor_pos, lstmouts = do_one_sentence(l2rlstm, r2llstm, char_seqs[i], bipos_seqs[i])
        num_tot += len(char_seqs[i])
        num_tot_cor += num_cor
        num_tot_cor_bi += num_cor_bi
        num_tot_cor_pos += num_cor_pos

        if i % show_acc_every == 0 and i != 0:
            print("accuracy bipos: ", num_tot_cor / num_tot)
            print("accuracy bi: ", num_tot_cor_bi / num_tot)
            print("accuracy pos: ", num_tot_cor_pos / num_tot)

            print("loss bipos: ", loss.value())

        seq_wh, w_1st_chars = get_wh_seq(char_seqs[i], bipos_seqs[i], bi_b_seqs[i])
        if (len(seq_wh) == 0):
            continue
        # loss_bi_b, num_cor_bi_b = bi_b(seq_wh, w_1st_chars, bi_b_seqs[i])
        loss_bi_b, num_cor_bi_b = bi_bunsetsu(lstmouts[i], bi_b_seqs[i])
        # num_tot_bi_b += len(seq_wh)
        num_tot_bi_b += len(char_seqs[i])
        num_tot_cor_bi_b += num_cor_bi_b
        if i % show_acc_every == 0 and i != 0:
            print("accuracy chunking: ", num_tot_cor_bi_b / num_tot_bi_b)
            print("loss chuncking: ", loss_bi_b.value())

    return


for e in range(epoc):
    TRAIN = True
    train(l2rlstm, r2llstm, train_char_seqs, train_word_bipos_seqs, train_chunk_bi_seqs)
    TRAIN = False
    dev(l2rlstm, r2llstm, dev_char_seqs, dev_word_bipos_seqs, dev_chunk_bi_seqs)
