import numpy as np
import dynet as dy
from utils import *
import glob
import time

from config import *
from paths import *
from file_reader import DataFrameKtc


files = glob.glob(path2KTC + 'syn/*.*')
#files = [path2KTC + 'syn/9501ED.KNP', path2KTC + 'syn/9501ED.KNP']

if STANDARD_SPLIT:
    files = glob.glob(path2KTC + 'syn/95010[1-9].*')
    
if JOS:
    files = glob.glob(path2KTC + 'just-one-sentence.txt')
    files = [path2KTC + 'just-one-sentence.txt', path2KTC + 'just-one-sentence.txt']


save_file = 'Bunsetsu-parser-KTC' + \
            '_LAYERS-character' + str(LAYERS_character) + \
            '_LAYERS-word' + str(LAYERS_word) + \
            '_LAYERS-bunsetsu' + str(LAYERS_bunsetsu) + \
            '_HIDDEN-DIM' + str(HIDDEN_DIM) + \
            '_INPUT-DIM' + str(INPUT_DIM) + \
            '_batch-size' + str(batch_size) + \
            '_learning-rate' + str(learning_rate) + \
            '_pdrop' + str(pdrop)

load_file = save_file


print(files)

df = DataFrameKtc

train_sents = []
for file in files[0:-1]:
    print('[train] reading this file: ', file)
    lines = df.file2lines(df, file, ' ', encoding)
    train_sents.extend(df.lines2sents(df, lines))
wd, cd, bpd, td = df.sents2dicts(df, train_sents)

wd.freeze()
cd.freeze()
bpd.freeze()
td.freeze()

dev_sents = []
for file in files[-1:]:
    print('[dev] reading this file: ', file)
    lines = df.file2lines(df, file, ' ', encoding)
    dev_sents.extend(df.lines2sents(df, lines))

for sent in dev_sents:
    for w in sent.word_forms:
        wd.add_entry(w)
    for c in sent.char_forms:
        cd.add_entry(c)
    for bp in sent.word_biposes:
        bpd.add_entry(bp)
        td.add_entry(bp[2:])



train_word_seqs, train_char_seqs, train_word_bipos_seqs, train_chunk_bi_seqs, train_chunk_deps, train_pos_seqs, train_word_bi_seqs = df.sents2ids([wd, cd, bpd, td], train_sents)


dev_word_seqs, dev_char_seqs, dev_word_bipos_seqs, dev_chunk_bi_seqs, dev_chunk_deps, dev_pos_seqs, dev_word_bi_seqs = df.sents2ids([wd, cd, bpd, td], dev_sents)

###Neural Network
WORDS_SIZE = len(wd.i2x) + 1
CHARS_SIZE = len(cd.i2x) + 1
BIPOS_SIZE = len(bpd.i2x) + 1
POS_SIZE = len(td.i2x) + 1

pc = dy.ParameterCollection()

l2rlstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 2, HIDDEN_DIM, pc)
r2llstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 2, HIDDEN_DIM, pc)

l2rlstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM + HIDDEN_DIM * 2, HIDDEN_DIM, pc)
r2llstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM + HIDDEN_DIM * 2, HIDDEN_DIM, pc)

l2rlstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
r2llstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)


params = {}
params["lp_w"] = pc.add_lookup_parameters((WORDS_SIZE + 1, INPUT_DIM))
params["lp_c"] = pc.add_lookup_parameters((CHARS_SIZE + 1, INPUT_DIM))
params["lp_p"] = pc.add_lookup_parameters((POS_SIZE + 1, INPUT_DIM))
params["lp_bp"] = pc.add_lookup_parameters((BIPOS_SIZE + 1, INPUT_DIM))

params["R_bi_b"] = pc.add_parameters((2, HIDDEN_DIM * 2))
params["bias_bi_b"] = pc.add_parameters((2))

params["head_MLP"] = pc.add_parameters((HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM * 2))
params["head_MLP_bias"] = pc.add_parameters((HIDDEN_DIM * 2))

params["dep_MLP"] = pc.add_parameters((HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM * 2))
params["dep_MLP_bias"] = pc.add_parameters((HIDDEN_DIM * 2))

params["R_bunsetsu_biaffine"] = pc.add_parameters((HIDDEN_DIM * 2 + biaffine_bias_y, HIDDEN_DIM * 2 + biaffine_bias_x))


def linear_interpolation(bias, R, inputs):
    ret = bias
    for i in range(len(inputs)):
        ret += R * inputs[i]
    return ret


def inputs2lstmouts(l2rlstm, r2llstm, inputs):

    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    l2rlstm.set_dropouts(pdrop, pdrop)
    r2llstm.set_dropouts(pdrop, pdrop)

    s_l2r = s_l2r_0
    s_r2l = s_r2l_0

    l2r_outs = s_l2r.add_inputs(inputs)
    r2l_outs = s_r2l.add_inputs(reversed(inputs))
    lstm_outs = [dy.cube(dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()])) for i in range(len(l2r_outs))]

    return lstm_outs


def char_embds(l2rlstm, r2llstm, char_seq, bipos_seq):
    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    lp_c = params["lp_c"]
    lp_bp = params["lp_bp"]

    l2rlstm.set_dropouts(pdrop, pdrop)
    r2llstm.set_dropouts(pdrop, pdrop)

    s_l2r = s_l2r_0
    s_r2l = s_r2l_0

    cembs = [dy.concatenate([lp_c[char_seq[i]], lp_bp[bipos_seq[i]]]) for i in range(len(char_seq))]

    l2r_outs = s_l2r.add_inputs(cembs)
    r2l_outs = s_r2l.add_inputs(reversed(cembs))
    lstm_outs = [dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()]) for i in range(len(l2r_outs))]

    return lstm_outs


def bi_bunsetsu(lstmout, bi_b_seq):
    num_cor = 0

    R_bi_b = dy.parameter(params["R_bi_b"])
    bias_bi_b = dy.parameter(params["bias_bi_b"])
    loss = []
    preds = []

    for i in range(len(bi_b_seq)):
        probs = dy.softmax(R_bi_b * lstmout[i] + bias_bi_b)
        loss.append(-dy.log(dy.pick(probs, bi_b_seq[i])))

        if not TRAIN:
            chosen = np.asscalar(np.argmax(probs.npvalue()))
            preds.append(chosen)
            if(chosen == bi_b_seq[i]):
                # print(chosen, " ", bi_b_seq[w_1st_chars[i]])
                num_cor += 1
    loss = dy.esum(loss)
    return loss, preds, num_cor


def bi_bunsetsu_wembs(wembs, bi_w_seq):
    num_cor = 0

    R_bi_b = dy.parameter(params["R_bi_b"])
    bias_bi_b = dy.parameter(params["bias_bi_b"])
    loss = []
    preds = []

    for i in range(len(bi_w_seq)):
        probs = dy.softmax(R_bi_b * wembs[i] + bias_bi_b)
        loss.append(-dy.log(dy.pick(probs, bi_w_seq[i])))

        if not TRAIN:
            chosen = np.asscalar(np.argmax(probs.npvalue()))
            preds.append(chosen)
            if(chosen == bi_w_seq[i]):
                # print(chosen, " ", bi_b_seq[w_1st_chars[i]])
                num_cor += 1
    loss = dy.esum(loss)
    return loss, preds, num_cor


def dep_bunsetsu(bembs):
    dep_MLP = dy.parameter(params["dep_MLP"])
    dep_MLP_bias = dy.parameter(params["dep_MLP_bias"])
    head_MLP = dy.parameter(params["head_MLP"])
    head_MLP_bias = dy.parameter(params["head_MLP_bias"])


    R_bunsetsu_biaffine = dy.parameter(params["R_bunsetsu_biaffine"])
    slen_x = len(bembs) - 1
    slen_y = slen_x + 1
    bembs_dep = dy.dropout(dy.concatenate(bembs[1:], 1), pdrop)
    bembs_head = dy.dropout(dy.concatenate(bembs, 1), pdrop)
    input_size = HIDDEN_DIM * 2

    bembs_dep = leaky_relu(dep_MLP * bembs_dep + dep_MLP_bias)
    bembs_head = leaky_relu(head_MLP * bembs_head + head_MLP_bias)

    blin = bilinear(bembs_dep, R_bunsetsu_biaffine, bembs_head, input_size, slen_x, slen_y, 1, 1, biaffine_bias_x, biaffine_bias_y)
    arc_loss = dy.reshape(blin, (slen_y,), slen_x)

    arc_preds = blin.npvalue().argmax(0)

    return arc_loss, arc_preds


def bunsetsu_range(bi_bunsetsu_seq):
    ret = [(0, 1)]
    start = 1

    for i in range(2, len(bi_bunsetsu_seq)):
        if bi_bunsetsu_seq[i] == 0:
            end = i
            ret.append((start, end))
            start = i

    ret.append((start, len(bi_bunsetsu_seq)))

    return ret


def word_range(bipos_seq):
    ret = [(0, 1)]
    start = 1

    for i in range(2, len(bipos_seq)):
        if (bpd.i2x)[bipos_seq[i]][0] == 'B':
            end = i
            ret.append((start, end))
            start = i

    ret.append((start, len(bipos_seq)))

    return ret


# def word_embds(lstmout, char_seq, bipos_seq, word_ranges):
def word_embds(cembs, char_seq, word_ranges):
    ret = []

    lp_c = params["lp_c"]
    lp_w = params["lp_w"]

    for wr in word_ranges:
        str = ""
        tmp_char = []

        for c in char_seq[wr[0]: wr[1]]:
            str += cd.i2x[c]
            tmp_char.append(lp_c[c])

        tmp_char = dy.esum(tmp_char)

        if str in wd.x2i:
            tmp_word = lp_w[wd.x2i[str]]
        else:
            tmp_word = tmp_char

        if wr[1] < len(cembs):
            ret.append((dy.concatenate([cembs[wr[1]] - cembs[wr[0]], tmp_word])))
        elif wr[0] == len(cembs) - 1:
            ret.append((dy.concatenate([cembs[wr[0]], tmp_word])))
        else:
            ret.append((dy.concatenate([cembs[-1] - cembs[wr[0]], tmp_word])))

    return ret


def bunsetsu_embds(wembs, bunsetsu_ranges):
    ret = []

    for br in bunsetsu_ranges:
        if br[1] < len(wembs):
            ret.append(wembs[br[1]] - wembs[br[0]])
        elif br[0] == len(wembs) - 1:
            ret.append(wembs[br[0]])
        else:
            ret.append(wembs[-1] - wembs[br[0]])

    return ret


def bilinear(x, W, y, input_size, seq_len_x, seq_len_y, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    # x,y: (input_size x seq_len) x batch_size
    if bias_x:
        x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len_x), dtype=np.float32))])
    if bias_y:
        y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len_y), dtype=np.float32))])

    nx, ny = input_size + bias_x, input_size + bias_y
    # W: (num_outputs x ny) x nx
    lin = W * x
    if num_outputs > 1:
        lin = dy.reshape(lin, (ny, num_outputs * seq_len_y), batch_size=batch_size)
    blin = dy.transpose(y) * lin
    if num_outputs > 1:
        blin = dy.reshape(blin, (seq_len_y, num_outputs, seq_len_x), batch_size=batch_size)
    # seq_len_y x seq_len_x if output_size == 1
    # seq_len_y x num_outputs x seq_len_x else
    return blin


def train(char_seqs, bipos_seqs, bi_b_seqs):
    trainer = dy.AdagradTrainer(pc, learning_rate)
    losses_bunsetsu = []
    losses_arcs = []

    tot_loss_in_iter = 0

    print(pdrop)

    for it in range(train_iter):
        print("total loss in previous iteration: ", tot_loss_in_iter)
        tot_loss_in_iter = 0
        print("iteration: ", it)
        num_tot_bunsetsu_dep = 0
        num_tot_cor_bunsetsu_dep = 0

        for i in (range(len(char_seqs))):
            if i % batch_size == 0:
                losses_bunsetsu = []
                losses_arcs = []

                dy.renew_cg()

            if random_pickup:  idx = i if not TRAIN else np.random.randint(len(char_seqs))
            else: idx = i

            if len(char_seqs[idx]) == 0 or len(bi_b_seqs[idx]) == 0:
                continue

            cembs = char_embds(l2rlstm_char, r2llstm_char, char_seqs[idx], bipos_seqs[idx])

            bi_w_seq = [0 if (bpd.i2x[b])[0] == 'B' else 1 for b in bipos_seqs[idx]]
            bi_w_seq = word_bi(bi_w_seq, bi_b_seqs[idx])

            word_ranges = word_range(bipos_seqs[idx])
            wembs = word_embds(cembs, char_seqs[idx], word_ranges)
            wembs = inputs2lstmouts(l2rlstm_word, r2llstm_word, wembs)

            loss_bi_bunsetsu, _, _ = bi_bunsetsu_wembs(wembs, bi_w_seq)
            losses_bunsetsu.append(loss_bi_bunsetsu)

            if i % batch_size == 0 and i != 0:
                loss_bi_bunsetsu_value = loss_bi_bunsetsu.value()

            if i % show_loss_every == 0 and i != 0:
                print(i, " bi_bunsetsu loss")
                print(loss_bi_bunsetsu_value)

            bunsetsu_ranges = bunsetsu_range(bi_w_seq)

            bembs = bunsetsu_embds(wembs, bunsetsu_ranges)

            bembs = inputs2lstmouts(l2rlstm_bunsetsu, r2llstm_bunsetsu, bembs)
            arc_loss, arc_preds = dep_bunsetsu(bembs)

            num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds, train_chunk_deps[idx]))

            num_tot_bunsetsu_dep += len(bembs) - 1

            losses_arcs.append(dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, train_chunk_deps[idx])))

            if i % batch_size == 0 and i != 0:
                losses_arcs.extend(losses_bunsetsu)

                sum_losses_arcs = dy.esum(losses_arcs)
                sum_losses_arcs_value = sum_losses_arcs.value()
                sum_losses_arcs.backward()
                trainer.update()
                tot_loss_in_iter += sum_losses_arcs_value

            if i % show_loss_every == 0 and i != 0:
                print(i, " arcs loss")
                print(sum_losses_arcs_value)
                print(train_sents[idx].word_forms)

                print(arc_preds)
                print(train_chunk_deps[idx])

                print("dep accuracy: ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)


def word_bi(bi_w_seq, bi_b_seq):
    ret = []

    for i in range(len(bi_w_seq)):
        if bi_w_seq[i] == 0:
            if bi_b_seq[i] == 0:
                ret.append(0)
            else:
                ret.append(1)

    return ret


def dev(char_seqs, bipos_seqs, bi_b_seqs):
    num_tot = 0

    num_tot_bi_b = 0
    num_tot_cor_bi_b = 0

    num_tot_bunsetsu_dep = 0
    num_tot_cor_bunsetsu_dep = 0

    complete_chunking = 0
    failed_chunking = 0

    print(pdrop)

    for i in range(len(char_seqs)):
        dy.renew_cg()
        if len(char_seqs[i]) == 0:
            continue

        bi_w_seq = [0 if (bpd.i2x[b])[0] == 'B' else 1 for b in bipos_seqs[i]]
        bi_w_seq = word_bi(bi_w_seq, bi_b_seqs[i])

        cembs = char_embds(l2rlstm_char, r2llstm_char, char_seqs[i], bipos_seqs[i])

        num_tot += len(char_seqs[i])

        word_ranges = word_range(bipos_seqs[i])
        wembs = word_embds(cembs, char_seqs[i], word_ranges)

        wembs = inputs2lstmouts(l2rlstm_word, r2llstm_word, wembs)

        loss_bi_b, preds_bi_b, num_cor_bi_b = bi_bunsetsu_wembs(wembs, bi_w_seq)
        num_tot_bi_b += len(wembs)
        num_tot_cor_bi_b += num_cor_bi_b
        if i % show_acc_every == 0 and i != 0:
            print("accuracy chunking: ", num_tot_cor_bi_b / num_tot_bi_b)
            print("loss chuncking: ", loss_bi_b.value())

        bunsetsu_ranges = bunsetsu_range(preds_bi_b)

        bembs = bunsetsu_embds(wembs, bunsetsu_ranges)

        bembs = inputs2lstmouts(l2rlstm_bunsetsu, r2llstm_bunsetsu, bembs)

        if i % show_acc_every == 0 and i != 0:
            loss_bi_b_value = loss_bi_b.value()
            print(i, " bi_b loss")
            print(loss_bi_b_value)
            if num_tot_bi_b > 0:
                print(i, " accuracy chunking ", num_tot_cor_bi_b / num_tot_bi_b)
            if num_tot_bunsetsu_dep > 0:
                print(i, " accuracy dep ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)

        if len(wembs) == num_cor_bi_b:
            complete_chunking += 1

        if len(dev_chunk_deps[i]) != len(bembs) - 1:
            failed_chunking += 1
            continue
        arc_loss, arc_preds = dep_bunsetsu(bembs)

        num_tot_bunsetsu_dep += len(bembs) - 1

        num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds, dev_chunk_deps[i]))
    with open("result_accuracy.txt", mode = 'a', encoding = 'utf-8') as f:
        f.write(str(i) + " accuracy chunking " + str(num_tot_cor_bi_b / num_tot_bi_b) + '\n')
        f.write(str(i) + " accuracy dep " + str(num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)+ '\n')
        f.write("complete chunking rate: " + str(complete_chunking / len(char_seqs))+ '\n')
        f.write("failed_chunking rate: " + str(failed_chunking / len(char_seqs))+ '\n')
        f.write("complete chunking: " + str(complete_chunking)+ '\n')
        f.write("failed_chunking: " + str(failed_chunking)+ '\n')
    print("complete_chunking rate: " + str(complete_chunking / len(char_seqs)))
    print("failed_chunking rate: " + str(failed_chunking / len(char_seqs)))
    print("complete chunking: " + str(complete_chunking))
    print("failed_chunking: " + str(failed_chunking))
    return


prev = time.time()

if LOAD:
    pc.populate(load_file)
    print("loaded from: ", load_file)

for e in range(epoc):
    prev = time.time() - prev
    print("time: ", prev)
    print("epoc: ", e)
    with open("result_accuracy.txt", mode = 'a', encoding = 'utf-8') as f:
        f.write("time: " + str(prev)+ '\n')
        f.write("epoc: " + str(e)+ '\n')

    TRAIN = True
    global pdrop
    pdrop = 0.33
    train(train_char_seqs, train_word_bipos_seqs, train_chunk_bi_seqs)
    if SAVE:
        pc.save(save_file)
        print("saved into: ", save_file)
    pdrop = 0.0
    TRAIN = False
    dev(dev_char_seqs, dev_word_bipos_seqs, dev_chunk_bi_seqs)
