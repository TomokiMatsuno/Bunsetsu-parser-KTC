import numpy as np
import dynet as dy
import glob

from config import *
from paths import *
from file_reader import DataFrameKtc


files = glob.glob(path2KTC + 'syn/*.*')
# files = [path2KTC + 'syn/9501ED.KNP']
# files = glob.glob(path2KTC + 'just-one-sentence.txt')

print(path2UD)

print(files)

df = DataFrameKtc

train_sents = []
for file in files[0:-1]:
    print('[train] reading this file: ', file)
    lines = df.file2lines(df, file, ' ', 'euc-jp')
    # lines = df.file2lines(df, file, ' ', 'utf-8')
    train_sents.extend(df.lines2sents(df, lines))
wd, cd, bpd = df.sents2dicts(df, train_sents)

wd.freeze()
cd.freeze()
bpd.freeze()

dev_sents = []
for file in files[-1:]:
    print('[dev] reading this file: ', file)
    lines = df.file2lines(df, file, ' ', 'euc-jp')
    # lines = df.file2lines(df, file, ' ', 'utf-8')
    dev_sents.extend(df.lines2sents(df, lines))

for sent in dev_sents:
    for w in sent.word_forms:
        wd.add_entry(w)
    for c in sent.char_forms:
        cd.add_entry(c)
    for bp in sent.word_biposes:
        bpd.add_entry(bp)



train_word_seqs, train_char_seqs, train_word_bipos_seqs, train_chunk_bi_seqs, train_chunk_deps = df.sents2ids([wd, cd, bpd], train_sents)


dev_word_seqs, dev_char_seqs, dev_word_bipos_seqs, dev_chunk_bi_seqs, dev_chunk_deps = df.sents2ids([wd, cd, bpd], dev_sents)

###Neural Network

VOCAB_SIZE = len(cd.i2x) + 1
BIPOS_SIZE = len(bpd.i2x) + 1

pc = dy.ParameterCollection()

l2rlstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
r2llstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)

params = {}
params["lp_c"] = pc.add_lookup_parameters((VOCAB_SIZE + 1, INPUT_DIM))
params["lp_bp"] = pc.add_lookup_parameters((VOCAB_SIZE + 1, INPUT_DIM))

params["R"] = pc.add_parameters((BIPOS_SIZE, HIDDEN_DIM * 2))
params["bias"] = pc.add_parameters((BIPOS_SIZE))

params["R_bemb"] = pc.add_parameters((HIDDEN_DIM * 2, HIDDEN_DIM * 2 + INPUT_DIM))
params["R_bemb_bias"] = pc.add_parameters((HIDDEN_DIM * 2))


params["R_bi_b"] = pc.add_parameters((2, HIDDEN_DIM * 2))
# params["R_bi_b"] = pc.add_parameters((2, INPUT_DIM * 2))
params["bias_bi_b"] = pc.add_parameters((2))

# params["R_bunsetsu_biaffine"] = pc.add_parameters((HIDDEN_DIM * 2, HIDDEN_DIM * 2 + 1))
params["R_bunsetsu_biaffine"] = pc.add_parameters((HIDDEN_DIM * 2, HIDDEN_DIM * 2))


def linear_interpolation(bias, R, inputs):
    ret = bias
    for i in range(len(inputs)):
        ret += R * inputs[i]
    return ret



def do_one_sentence(l2rlstm, r2llstm, char_seq, bipos_seq):
    num_cor = 0
    num_cor_bi = 0
    num_cor_pos = 0

    # dy.renew_cg()
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
        loss.append(dy.pickneglogsoftmax(probs, bipos_seq[i]))


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


def dep_bunsetsu(bembs):
    # lp_c = params["lp_c"]
    R_bunsetsu_biaffine = dy.parameter(params["R_bunsetsu_biaffine"])
    slen = len(bembs)
    # bembs = [lp_c[cd.x2i["ROOT"]]] + bembs
    bembs = dy.concatenate(bembs, 1)
    input_size = HIDDEN_DIM * 2

    blin = bilinear(bembs, R_bunsetsu_biaffine, bembs, input_size, slen, 1, 1, False, False)
    arc_loss = dy.reshape(blin, (slen,), slen)

    arc_preds = blin.npvalue().argmax(0)

    return arc_loss, arc_preds


def bunsetsu_range(bi_bunsetsu_seq):
    ret = []
    start = 0

    for i in range(1, len(bi_bunsetsu_seq)):
        if bi_bunsetsu_seq[i] == 0:
            end = i - 1
            ret.append((start, end))
            start = i

    return ret


def bunsetsu_embds(lstmout, bipos_seq, bunsetsu_ranges):
    ret = []
    # lp_c = dy.parameter(params["lp_c"])
    R_bemb = dy.parameter(params["R_bemb"])
    R_bemb_bias = dy.parameter(params["R_bemb_bias"])

    # ret.append(lp_c[wd.x2i["ROOT"]])

    for br in bunsetsu_ranges:
        # ret.append(linear_interpolation(R_bemb_bias, R_bemb, lstmout[br[0]: br[1]] + bipos_seq[br[0]: br[1]]))
        ret.append(linear_interpolation(R_bemb_bias, R_bemb, [dy.concatenate([l, b]) for l, b in zip(lstmout[br[0]: br[1]], bipos_seq[br[0]: br[1]])]))

    return ret



def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    # x,y: (input_size x seq_len) x batch_size
    if bias_x:
        x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
    if bias_y:
        y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])

    nx, ny = input_size + bias_x, input_size + bias_y
    # W: (num_outputs x ny) x nx
    lin = W * x
    if num_outputs > 1:
        lin = dy.reshape(lin, (ny, num_outputs * seq_len), batch_size=batch_size)
    blin = dy.transpose(y) * lin
    if num_outputs > 1:
        blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size=batch_size)
    # seq_len_y x seq_len_x if output_size == 1
    # seq_len_y x num_outputs x seq_len_x else
    return blin


def train(l2rlstm, r2llstm, char_seqs, bipos_seqs, bi_b_seqs):
    trainer = dy.SimpleSGDTrainer(pc)
    losses = []
    losses_bunsetsu = []
    losses_arcs = []

    lp_bp = params["lp_bp"]

    for it in range(train_iter):
        # dy.renew_cg()
        for i in (range(len(char_seqs))):
            if i % batch_size == 0:
                losses = []
                losses_bunsetsu = []
                losses_arcs = []

                dy.renew_cg()

            # idx = i if not TRAIN else np.random.randint(len(char_seqs))
            idx = i
            if len(char_seqs[idx]) == 0 or len(bi_b_seqs[idx]) == 0:
                continue
            loss, _, _, _, lstmout = do_one_sentence(l2rlstm, r2llstm, char_seqs[idx], bipos_seqs[idx])
            losses.append(loss)
            # dy.esum(losses)
            if(i % batch_size == 0) and i != 0:
                loss_value = loss.value()
                # dy.esum(losses)
                # loss.backward()
                # trainer.update()
            if i % show_loss_every == 0 and i != 0:
                print(i, " bipos loss")
                print(loss_value)

            # seq_wh, w_1st_chars = get_wh_seq(char_seqs[idx], bipos_seqs[idx], bi_b_seqs[idx])
            # if(len(seq_wh) == 0):
            #     continue
            # loss_bi_b, _ = bi_b(seq_wh, w_1st_chars, bi_b_seqs[i])
            loss_bi_bunsetsu, _ = bi_bunsetsu(lstmout, bi_b_seqs[idx])
            losses_bunsetsu.append(loss_bi_bunsetsu)
            # dy.esum(losses_bunsetsu)

            if i % batch_size == 0 and i != 0:
                # loss_bi_b_value = loss_bi_b.value()
                # loss_bi_b.backward()
                loss_bi_bunsetsu_value = loss_bi_bunsetsu.value()

                # loss_bi_bunsetsu.backward()
                # trainer.update()
            if i % show_loss_every == 0 and i != 0:
                # print(i, " bi_b loss")
                # print(loss_bi_b_value)
                print(i, " bi_bunsetsu loss")
                print(loss_bi_bunsetsu_value)

            bunsetsu_ranges = bunsetsu_range(bi_b_seqs[idx])
            bembs = bunsetsu_embds(lstmout, [lp_bp[bp] for bp in bipos_seqs[idx]], bunsetsu_ranges)
            arc_loss, arc_preds = dep_bunsetsu(bembs)

            losses_arcs.append(dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, train_chunk_deps[idx])))


            if i % batch_size == 0 and i != 0:
                # loss_bi_b_value = loss_bi_b.value()
                # loss_bi_b.backward()
                print(arc_preds)
                print(train_chunk_deps[idx])
                losses_arcs.extend(losses)
                losses_arcs.extend(losses_bunsetsu)

                sum_losses_arcs = dy.esum(losses_arcs)
                sum_losses_arcs_value = sum_losses_arcs.value()
                sum_losses_arcs.backward()
                trainer.update()

                # dy.renew_cg()
            if i % show_loss_every == 0 and i != 0:
                # print(i, " bi_b loss")
                # print(loss_bi_b_value)
                print(i, " arcs loss")
                print(sum_losses_arcs_value)



def dev(l2rlstm, r2llstm, char_seqs, bipos_seqs, bi_b_seqs):
    num_tot = 0
    num_tot_cor = 0

    num_tot_cor_bi = 0
    num_tot_cor_pos = 0

    num_tot_bi_b = 0
    num_tot_cor_bi_b = 0

    num_tot_bunsetsu_dep = 0
    num_tot_cor_bunsetsu_dep = 0

    lp_bp = params["lp_bp"]


    for i in range(len(char_seqs)):
        if(len(char_seqs[i]) == 0):
            continue
        # loss, num_cor, num_cor_bi, num_cor_pos = do_one_sentence(l2rlstm, r2llstm, char_seqs[i], bipos_seqs[i])
        loss, num_cor, num_cor_bi, num_cor_pos, lstmout = do_one_sentence(l2rlstm, r2llstm, char_seqs[i], bipos_seqs[i])
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
        loss_bi_b, num_cor_bi_b = bi_bunsetsu(lstmout, bi_b_seqs[i])
        # num_tot_bi_b += len(seq_wh)
        num_tot_bi_b += len(char_seqs[i])
        num_tot_cor_bi_b += num_cor_bi_b
        if i % show_acc_every == 0 and i != 0:
            print("accuracy chunking: ", num_tot_cor_bi_b / num_tot_bi_b)
            print("loss chuncking: ", loss_bi_b.value())

        bunsetsu_ranges = bunsetsu_range(bi_b_seqs[i])
        bembs = bunsetsu_embds(lstmout, [lp_bp[bp] for bp in bipos_seqs[i]], bunsetsu_ranges)
        arc_loss, arc_preds = dep_bunsetsu(bembs)


        num_tot_bunsetsu_dep += len(bembs)
        num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds, dev_chunk_deps[i]))

            # dy.renew_cg()
        if i % show_acc_every == 0 and i != 0:
            # print(i, " bi_b loss")
            # print(loss_bi_b_value)
            print(i, " accuracy dep ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)
        dy.renew_cg()

    return


for e in range(epoc):
    TRAIN = True
    train(l2rlstm, r2llstm, train_char_seqs, train_word_bipos_seqs, train_chunk_bi_seqs)
    TRAIN = False
    dev(l2rlstm, r2llstm, dev_char_seqs, dev_word_bipos_seqs, dev_chunk_bi_seqs)
