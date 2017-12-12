from collections import defaultdict

import numpy as np
import dynet as dy

from config import *
from utils import *

from tarjan import *
import glob
import time

from paths import *
from file_reader import DataFrameKtc
global circle_count, root_0, root_more_than_1
circle_count = 0
root_0 = 0
root_more_than_1 = 0


train_dev_boundary = -1
files = glob.glob(path2KTC + 'syn/*.*')

if CABOCHA_SPLIT:
    files = glob.glob(path2KTC + 'syn/95010[1-9].*')
    train_dev_boundary = -1
best_acc = 0.0
update = False
early_stop_count = 0


#files = [path2KTC + 'syn/9501ED.KNP', path2KTC + 'syn/9501ED.KNP']



if STANDARD_SPLIT:
    files = glob.glob(path2KTC + 'syn/95010[1-9].*')
    files.extend(glob.glob(path2KTC + 'syn/95011[0-1].*'))
    files.extend(glob.glob(path2KTC + 'syn/950[1-8]ED.*'))
    files.extend(glob.glob(path2KTC + 'syn/95011[4-7].*'))
    files.extend(glob.glob(path2KTC + 'syn/951[0-2]ED.*'))
    train_dev_boundary = -7

if JOS:
    files = glob.glob(path2KTC + 'just-one-sentence.txt')
    files = [path2KTC + 'just-one-sentence.txt', path2KTC + 'just-one-sentence.txt']

if MINI_SET:
    files = [path2KTC + 'miniKTC_train.txt', path2KTC + 'miniKTC_dev.txt' ]

save_file = 'Bunsetsu-parser-KTC' + \
            '_LAYERS-character' + str(LAYERS_character) + \
            '_LAYERS-word' + str(LAYERS_word) + \
            '_LAYERS-bunsetsu' + str(LAYERS_bunsetsu) + \
            '_HIDDEN-DIM' + str(HIDDEN_DIM) + \
            '_INPUT-DIM' + str(INPUT_DIM) + \
            '_batch-size' + str(batch_size) + \
            '_learning-rate' + str(learning_rate) + \
            '_pdrop' + str(pdrop) + \
            '_pdrop_bunsetsu' + str(pdrop_bunsetsu) + \
            '_orthogonal'

if bemb_attention:
    save_file = save_file + '_bembattn' + '-dropout' + str(pdrop * 2)

if bemb_lstm:
    save_file = save_file + '_bemblstm'

load_file = save_file

result_file = save_file + "_result_accuracy.txt"


print(files)

df = DataFrameKtc

train_sents = []
for file in files[0:train_dev_boundary]:
    print('[train] reading this file: ', file)
    lines = df.file2lines(df, file, ' ', encoding)
    train_sents.extend(df.lines2sents(df, lines))
wd, cd, bpd, td, tsd, wifd, witd = df.sents2dicts(df, train_sents)

wd.freeze()
cd.freeze()
bpd.freeze()
td.freeze()
tsd.freeze()
wifd.freeze()
witd.freeze()

dev_sents = []
for file in files[train_dev_boundary:]:
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
    for ps in sent.pos_sub:
        tsd.add_entry(ps)
    for wif in sent.word_inflection_forms:
        wifd.add_entry(wif)
    for wit in sent.word_inflection_types:
        witd.add_entry(wit)



train_word_seqs, train_char_seqs, train_word_bipos_seqs, \
train_chunk_bi_seqs, train_chunk_deps, train_pos_seqs, train_word_bi_seqs, \
train_pos_sub_seqs, train_wif_seqs, train_wit_seqs \
    = df.sents2ids([wd, cd, bpd, td, tsd, wifd, witd], train_sents)


dev_word_seqs, dev_char_seqs, dev_word_bipos_seqs, \
dev_chunk_bi_seqs, dev_chunk_deps, dev_pos_seqs, dev_word_bi_seqs, \
dev_pos_sub_seqs, dev_wif_seqs, dev_wit_seqs \
    = df.sents2ids([wd, cd, bpd, td, tsd, wifd, witd], dev_sents)

total_chunks = 0
for dcd in dev_chunk_deps:
    total_chunks += len(dcd)

print(total_chunks)

###Neural Network
WORDS_SIZE = len(wd.i2x) + 1
CHARS_SIZE = len(cd.i2x) + 1
BIPOS_SIZE = len(bpd.i2x) + 1
POS_SIZE = len(td.i2x) + 1
POSSUB_SIZE = len(tsd.i2x) + 1
WIF_SIZE = len(wifd.i2x) + 1
WIT_SIZE = len(witd.i2x) + 1

pc = dy.ParameterCollection()

trainer = dy.AdadeltaTrainer(pc)

global_step = 0

def update_parameters():
    if use_annealing:
        trainer.learning_rate = learning_rate * decay ** (global_step / decay_steps)
    trainer.update()


# trainer = dy.AdagradTrainer(pc, learning_rate)

if orthonormal:
    # l2rlstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 2, HIDDEN_DIM, pc)
    # r2llstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 2, HIDDEN_DIM, pc)
    l2rlstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, INPUT_DIM, HIDDEN_DIM, pc)
    r2llstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, INPUT_DIM, HIDDEN_DIM, pc)
    if use_wif_wit:
        l2rlstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 3 + HIDDEN_DIM * 2, word_HIDDEN_DIM, pc)
        r2llstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 3 + HIDDEN_DIM * 2, word_HIDDEN_DIM, pc)
    else:
        l2rlstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 2 + HIDDEN_DIM * 2, word_HIDDEN_DIM, pc)
        r2llstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 2 + HIDDEN_DIM * 2, word_HIDDEN_DIM, pc)

    l2rlstm_bemb = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
    r2llstm_bemb = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)

    if bemb_lstm:
        l2rlstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
        r2llstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
    # l2rlstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
    # r2llstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
else:
    # l2rlstm_char = dy.VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 2, HIDDEN_DIM, pc)
    # r2llstm_char = dy.VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 2, HIDDEN_DIM, pc)
    l2rlstm_char = dy.VanillaLSTMBuilder(LAYERS_character, INPUT_DIM, HIDDEN_DIM, pc)
    r2llstm_char = dy.VanillaLSTMBuilder(LAYERS_character, INPUT_DIM, HIDDEN_DIM, pc)

    # if use_wif_wit:
    l2rlstm_word = dy.VanillaLSTMBuilder(LAYERS_word, HIDDEN_DIM + HIDDEN_DIM * 2, word_HIDDEN_DIM, pc)
    r2llstm_word = dy.VanillaLSTMBuilder(LAYERS_word, HIDDEN_DIM + HIDDEN_DIM * 2, word_HIDDEN_DIM, pc)

    l2rlstm_word = dy.VanillaLSTMBuilder(LAYERS_word, word_HIDDEN_DIM, word_HIDDEN_DIM, pc)
    r2llstm_word = dy.VanillaLSTMBuilder(LAYERS_word, word_HIDDEN_DIM, word_HIDDEN_DIM, pc)
    # else:
    #     l2rlstm_word = dy.VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 2 + HIDDEN_DIM * 2, word_HIDDEN_DIM, pc)
    #     r2llstm_word = dy.VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 2 + HIDDEN_DIM * 2, word_HIDDEN_DIM, pc)

    l2rlstm_bemb = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM, bunsetsu_HIDDEN_DIM, pc)
    r2llstm_bemb = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM, bunsetsu_HIDDEN_DIM, pc)

    if bemb_lstm:
        if use_wembs or next_bemb:
            l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
            r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
        else:
            l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
            r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)

    # l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)
    # r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM, pc)


params = {}
params["lp_w"] = pc.add_lookup_parameters((WORDS_SIZE + 1, INPUT_DIM))
params["lp_c"] = pc.add_lookup_parameters((CHARS_SIZE + 1, INPUT_DIM))
params["lp_p"] = pc.add_lookup_parameters((POS_SIZE + 1, INPUT_DIM))
params["lp_ps"] = pc.add_lookup_parameters((POSSUB_SIZE + 1, INPUT_DIM))

if use_wif_wit:
    params["lp_wif"] = pc.add_lookup_parameters((WIF_SIZE + 1, INPUT_DIM))
    params["lp_wit"] = pc.add_lookup_parameters((WIT_SIZE + 1, INPUT_DIM))



# params["lp_bp"] = pc.add_lookup_parameters((BIPOS_SIZE + 1, INPUT_DIM))
if use_wembs:
    params["R_bi_b"] = pc.add_parameters((2, word_HIDDEN_DIM * 2))
else:
    params["R_bi_b"] = pc.add_parameters((2, word_HIDDEN_DIM))
params["bias_bi_b"] = pc.add_parameters((2))

# params["R_w"] = pc.add_parameters((HIDDEN_DIM, INPUT_DIM * 2))
# params["bias_w"] = pc.add_parameters((HIDDEN_DIM))
#
# params["R_p"] = pc.add_parameters((HIDDEN_DIM, INPUT_DIM * 2))
# params["bias_p"] = pc.add_parameters((HIDDEN_DIM))
#
# params["R_wi"] = pc.add_parameters((HIDDEN_DIM, INPUT_DIM * 2))
# params["bias_wi"] = pc.add_parameters((HIDDEN_DIM))

# params["R_w"] = pc.add_parameters((word_HIDDEN_DIM, INPUT_DIM * 6))
# params["bias_w"] = pc.add_parameters((word_HIDDEN_DIM))
#
# params["R_p"] = pc.add_parameters((word_HIDDEN_DIM, INPUT_DIM * 6))
#
# params["R_wi"] = pc.add_parameters((word_HIDDEN_DIM, INPUT_DIM * 6))

params["R_w"] = pc.add_parameters((word_HIDDEN_DIM , INPUT_DIM * 2))
params["bias_w"] = pc.add_parameters((word_HIDDEN_DIM))

params["R_p"] = pc.add_parameters((word_HIDDEN_DIM, INPUT_DIM * 2))
params["bias_p"] = pc.add_parameters((word_HIDDEN_DIM))
params["R_wi"] = pc.add_parameters((word_HIDDEN_DIM, INPUT_DIM * 2))
params["bias_wi"] = pc.add_parameters((word_HIDDEN_DIM))

params["R_b1"] = pc.add_parameters((bunsetsu_HIDDEN_DIM, word_HIDDEN_DIM))
params["R_b2"] = pc.add_parameters((bunsetsu_HIDDEN_DIM, word_HIDDEN_DIM * 2))
params["R_b3"] = pc.add_parameters((bunsetsu_HIDDEN_DIM, word_HIDDEN_DIM * 3))
params["b1_bias"] = pc.add_parameters((bunsetsu_HIDDEN_DIM))
params["b2_bias"] = pc.add_parameters((bunsetsu_HIDDEN_DIM))
params["b3_bias"] = pc.add_parameters((bunsetsu_HIDDEN_DIM))

params["filter_2d"] = pc.add_parameters((2, 2, 1, 2))

# params["head_MLP"] = pc.add_parameters((HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM * 2))
# params["head_MLP_bias"] = pc.add_parameters((HIDDEN_DIM * 2))
#
# params["dep_MLP"] = pc.add_parameters((HIDDEN_DIM * 2, bunsetsu_HIDDEN_DIM * 2))
# params["dep_MLP_bias"] = pc.add_parameters((HIDDEN_DIM * 2))

params["head_MLP"] = pc.add_parameters((MLP_HIDDEN_DIM, bunsetsu_HIDDEN_DIM * 2))
params["head_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM))

params["dep_MLP"] = pc.add_parameters((MLP_HIDDEN_DIM, bunsetsu_HIDDEN_DIM * 2))
params["dep_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM))

# params["R_bunsetsu_biaffine"] = pc.add_parameters((HIDDEN_DIM * 2 + biaffine_bias_y, HIDDEN_DIM * 2 + biaffine_bias_x))
# params["R_word_biaffine"] = pc.add_parameters((HIDDEN_DIM * 2, HIDDEN_DIM * 2))

params["R_bunsetsu_biaffine"] = pc.add_parameters((MLP_HIDDEN_DIM + biaffine_bias_y, MLP_HIDDEN_DIM + biaffine_bias_x))

if bemb_attention:
    params["R_bemb_biaffine"] = pc.add_parameters((word_HIDDEN_DIM, word_HIDDEN_DIM))


def linear_interpolation(bias, R, inputs):
    ret = bias
    for i in range(len(inputs)):
        ret += R * inputs[i]
    return ret


def inputs2lstmouts(l2rlstm, r2llstm, inputs, pdrop):

    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    l2rlstm.set_dropouts(pdrop, pdrop)
    r2llstm.set_dropouts(pdrop, pdrop)

    s_l2r = s_l2r_0
    s_r2l = s_r2l_0

    l2r_outs = s_l2r.add_inputs(inputs)
    r2l_outs = s_r2l.add_inputs(reversed(inputs))
    lstm_outs = [dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()]) for i in range(len(l2r_outs))]

    return lstm_outs

def inputs2lstmouts_wembs(l2rlstm, r2llstm, inputs, pdrop):

    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    l2rlstm.set_dropouts(pdrop, pdrop)
    r2llstm.set_dropouts(pdrop, pdrop)

    s_l2r = s_l2r_0
    s_r2l = s_r2l_0

    l2r_outs = s_l2r.add_inputs(inputs)
    r2l_outs = s_r2l.add_inputs(reversed(inputs))
    lstm_outs = [dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()]) for i in range(len(l2r_outs))]
    l2r_outs = [l2r_outs[i].output() for i in range(len(l2r_outs))]
    r2l_outs = [r2l_outs[i].output() for i in range(len(r2l_outs))]

    return lstm_outs, l2r_outs, r2l_outs



def char_embds(char_seq):

    lp_c = params["lp_c"]

    cembs = [lp_c[char_seq[i]] for i in range(len(char_seq))]

    return cembs

#
# def char_embds(l2rlstm, r2llstm, char_seq, bipos_seq):
#     s_l2r_0 = l2rlstm.initial_state()
#     s_r2l_0 = r2llstm.initial_state()
#
#     lp_c = params["lp_c"]
#     lp_bp = params["lp_bp"]
#
#     l2rlstm.set_dropouts(pdrop, pdrop)
#     r2llstm.set_dropouts(pdrop, pdrop)
#
#     s_l2r = s_l2r_0
#     s_r2l = s_r2l_0
#
#     cembs = [dy.concatenate([lp_c[char_seq[i]], lp_bp[bipos_seq[i]]]) for i in range(len(char_seq))]
#
#     l2r_outs = s_l2r.add_inputs(cembs)
#     r2l_outs = s_r2l.add_inputs(reversed(cembs))
#     lstm_outs = [dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()]) for i in range(len(l2r_outs))]
#
#     return lstm_outs


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
    wembs = wembs[1: -1]


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
    # slen_x = len(bembs) - 1
    slen_x = len(bembs)
    # slen_y = slen_x + 1
    slen_y = slen_x
    # bembs_dep = dy.dropout(dy.concatenate(bembs[1:], 1), pdrop)
    bembs_dep = dy.dropout(dy.concatenate(bembs, 1), pdrop_bunsetsu)
    bembs_head = dy.dropout(dy.concatenate(bembs, 1), pdrop_bunsetsu)
    input_size = HIDDEN_DIM * 2

    bembs_dep = leaky_relu(dep_MLP * bembs_dep + dep_MLP_bias)
    bembs_head = leaky_relu(head_MLP * bembs_head + head_MLP_bias)

    blin = bilinear(bembs_dep, R_bunsetsu_biaffine, bembs_head, input_size, slen_x, slen_y, 1, 1, biaffine_bias_x, biaffine_bias_y)

    arc_preds_not_argmax = blin.npvalue().argmax(0)
    msk = [1] * slen_x
    # msk[0] = 0
    arc_probs = dy.softmax(blin).npvalue()
    # arc_probs = arc_probs[1:]
    arc_probs = np.transpose(arc_probs)
    arc_preds = arc_argmax(arc_probs, slen_x, msk)
    arc_preds = arc_preds[1:]
    arc_loss = dy.reshape(blin, (slen_y,), slen_x)
    arc_loss = dy.pick_batch_elems(arc_loss, [i for i in range(1, slen_x)])


    return arc_loss, arc_preds, arc_preds_not_argmax


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


# def word_embds(char_seq, pos_seq, word_ranges):
#     ret = []
#
#     lp_c = params["lp_c"]
#     lp_w = params["lp_w"]
#     lp_p = params["lp_p"]
#
#     for wr in word_ranges:
#         str = ""
#         tmp_char = []
#         tmp_pos = []
#
#         for c in char_seq[wr[0]: wr[1]]:
#             str += cd.i2x[c]
#             tmp_char.append(lp_c[c])
#
#         for p in pos_seq[wr[0]: wr[1]]:
#             tmp_pos.append(lp_p[p])
#
#         tmp_char_len = len(tmp_char)
#         tmp_char = dy.esum(tmp_char) / tmp_char_len
#         # tmp_pos = dy.esum(tmp_pos)
#         tmp_pos = tmp_pos[0]
#
#         rnd_int = np.random.randint(0, 2)
#
#         if TRAIN:
#             if rnd_int == 0 and str in wd.x2i:
#                 tmp_word = dy.concatenate([lp_w[wd.x2i[str]], tmp_pos])
#             else:
#                 tmp_word = dy.concatenate([tmp_char, tmp_pos])
#         else:
#             if str in wd.x2i:
#                 tmp_word = dy.concatenate([lp_w[wd.x2i[str]], tmp_pos])
#             else:
#                 tmp_word = dy.concatenate([tmp_char, tmp_pos])
#
#         ret.append(tmp_word)
#
#     return ret


# def word_embds(cembs, char_seq, pos_seq, pos_sub_seq, wif_seq, wit_seq, word_ranges):
#     ret = []
#
#     lp_c = params["lp_c"]
#     lp_w = params["lp_w"]
#     lp_p = params["lp_p"]
#     lp_ps = params["lp_ps"]
#
#     R_w = dy.parameter(params["R_w"])
#     bias_w = dy.parameter(params["bias_w"])
#
#     R_p = dy.parameter(params["R_p"])
#     bias_p = dy.parameter(params["bias_p"])
#
#     if use_wif_wit:
#         lp_wif = params["lp_wif"]
#         lp_wit = params["lp_wit"]
#
#         R_wi = dy.parameter(params["R_wi"])
#         bias_wi = dy.parameter(params["bias_wi"])
#
#     for idx, wr in enumerate(word_ranges):
#         str = ""
#         tmp_char = []
#         tmp_pos = []
#
#         for c in char_seq[wr[0]: wr[1]]:
#             str += cd.i2x[c]
#             tmp_char.append(dy.dropout(lp_c[c], pdrop))
#
#         for p in pos_seq[wr[0]: wr[1]]:
#             tmp_pos.append(lp_p[p])
#
#         tmp_char_len = len(tmp_char)
#
#         tmp_char = dy.esum(tmp_char) / tmp_char_len
#
#         # pos_sub_pos = bias_p + R_p * dy.concatenate([lp_p[pos_seq[wr[0]]], lp_ps[pos_sub_seq[idx]]])
#         pos_sub_pos = dy.concatenate([lp_p[pos_seq[wr[0]]], lp_ps[pos_sub_seq[idx]]])
#         if use_wif_wit:
#             wif_wit = bias_wi + R_wi * dy.concatenate([lp_wif[wif_seq[idx]], lp_wit[wit_seq[idx]]])
#             # wif_wit = dy.concatenate([lp_wif[wif_seq[idx]], lp_wit[wit_seq[idx]]])
#             tmp_word = pos_sub_pos + wif_wit
#             # tmp_word = dy.concatenate([pos_sub_pos, wif_wit])
#         else:
#             tmp_word = linear_interpolation(bias_p, R_p, dy.concatenate([lp_p[pos_seq[wr[0]]], lp_ps[pos_sub_seq[idx]]]))
#             # tmp_word = pos_sub_pos
#         if TRAIN:
#             rnd_int = np.random.randint(0, 2)
#             # rnd_pos = np.random.randint(0, 6)
#
#             # if rnd_pos == 0:
#             #     tmp_pos = tmp_char
#             # elif rnd_pos == 1:
#             #     tmp_pos = dy.zeros((INPUT_DIM))
#
#             if str in wd.x2i and rnd_int == 0:
#                 # tmp_word = dy.concatenate([dy.dropout(lp_w[wd.x2i[str]], pdrop), tmp_word])
#                 tmp_word = bias_w + R_w * dy.concatenate(
#                     [dy.dropout(lp_w[wd.x2i[str]], pdrop), tmp_char]) + tmp_word
#             else:
#                 # tmp_word = dy.concatenate([tmp_char, tmp_word])
#                 tmp_word = bias_w + R_w * dy.concatenate(
#                     [tmp_char, tmp_char]) + tmp_word
#         else:
#             if str in wd.x2i:
#                 # tmp_word = dy.concatenate([lp_w[wd.x2i[str]], tmp_word])
#                 tmp_word = bias_w + R_w * dy.concatenate(
#                     [lp_w[wd.x2i[str]], tmp_char]) + tmp_word
#             else:
#                 # tmp_word = dy.concatenate([tmp_char, tmp_word])
#                 tmp_word = bias_w + R_w * dy.concatenate(
#                     [tmp_char, tmp_char]) + tmp_word
#         tmp_word = dy.cube(tmp_word)
#
#         if wr[1] < len(cembs):
#             ret.append((dy.concatenate([cembs[wr[1]] - cembs[wr[0]], tmp_word])))
#         elif wr[0] == len(cembs) - 1:
#             ret.append((dy.concatenate([cembs[wr[0]], tmp_word])))
#         else:
#             ret.append((dy.concatenate([cembs[-1] - cembs[wr[0]], tmp_word])))
#
#     return ret


def word_embds(char_seq, pos_seq, pos_sub_seq, wif_seq, wit_seq, word_ranges):
    ret = []

    lp_c = params["lp_c"]
    lp_w = params["lp_w"]
    lp_p = params["lp_p"]
    lp_ps = params["lp_ps"]

    if use_wif_wit:
        lp_wif = params["lp_wif"]
        lp_wit = params["lp_wit"]

    wembs = []


    psp = (dy.concatenate([lp_p[td.x2i["SOS"]], lp_ps[tsd.x2i["SOS"]]]))
    wft = (dy.concatenate([lp_wif[wifd.x2i["SOS"]], lp_wit[witd.x2i["SOS"]]]))
    wch = (dy.concatenate([lp_w[wd.x2i["SOS"]], lp_c[cd.x2i["SOS"]]]))

    R_w = dy.parameter(params["R_w"])
    bias_w = dy.parameter(params["bias_w"])

    R_p = dy.parameter(params["R_p"])
    bias_p = dy.parameter(params["bias_p"])

    R_wi = dy.parameter(params["R_wi"])
    bias_wi = dy.parameter(params["bias_wi"])

    wembs.append((R_p * psp + R_wi * wft + R_w * wch + bias_w + bias_p + bias_wi))

    for idx, wr in enumerate(word_ranges):
        str = ""
        tmp_char = []
        tmp_pos = []

        for c in char_seq[wr[0]: wr[1]]:
            str += cd.i2x[c]
            tmp_char.append(dy.dropout(lp_c[c], pdrop))

        for p in pos_seq[wr[0]: wr[1]]:
            tmp_pos.append(lp_p[p])

        tmp_char_len = len(tmp_char)

        tmp_char = dy.esum(tmp_char) / tmp_char_len

        pos_sub_pos = bias_p + R_p * dy.concatenate([lp_p[pos_seq[wr[0]]], lp_ps[pos_sub_seq[idx]]])
        wif_wit = bias_wi + R_wi * dy.concatenate([lp_wif[wif_seq[idx]], lp_wit[wit_seq[idx]]])
        tmp_word = pos_sub_pos + wif_wit
        if TRAIN:
            rnd_int = np.random.randint(0, 2)

            if str in wd.x2i and rnd_int == 0:
                tmp_word = bias_w + R_w * dy.concatenate(
                    [dy.dropout(lp_w[wd.x2i[str]], pdrop), tmp_char]) + tmp_word
            else:
                tmp_word = bias_w + R_w * dy.concatenate(
                    [tmp_char, tmp_char]) + tmp_word
        else:
            if str in wd.x2i:
                tmp_word = bias_w + R_w * dy.concatenate(
                    [lp_w[wd.x2i[str]], tmp_char]) + tmp_word
            else:
                tmp_word = bias_w + R_w * dy.concatenate(
                    [tmp_char, tmp_char]) + tmp_word

        wembs.append((tmp_word))

    psp = (dy.concatenate([lp_p[td.x2i["EOS"]], lp_ps[tsd.x2i["EOS"]]]))
    wft = (dy.concatenate([lp_wif[wifd.x2i["EOS"]], lp_wit[witd.x2i["EOS"]]]))
    wch = (dy.concatenate([lp_w[wd.x2i["EOS"]], lp_c[cd.x2i["EOS"]]]))

    wembs.append((R_p * psp + R_wi * wft + R_w * wch + bias_w + bias_wi + bias_p))

    for widx in range(len(wembs)):
       ret.append(dy.cube(wembs[widx]))

    return ret

# def word_embds(char_seq, pos_seq, pos_sub_seq, wif_seq, wit_seq, word_ranges):
#     ret = []
#
#     lp_c = params["lp_c"]
#     lp_w = params["lp_w"]
#     lp_p = params["lp_p"]
#     lp_ps = params["lp_ps"]
#
#     if use_wif_wit:
#         lp_wif = params["lp_wif"]
#         lp_wit = params["lp_wit"]
#
#     psp_seq = []
#     wft_seq = []
#     wch_seq = []
#     wembs = []
#
#     psp_seq.append(dy.concatenate([lp_p[td.x2i["SOS"]], lp_ps[tsd.x2i["SOS"]]]))
#     wft_seq.append(dy.concatenate([lp_wif[wifd.x2i["SOS"]], lp_wit[witd.x2i["SOS"]]]))
#     wch_seq.append(dy.concatenate([lp_w[wd.x2i["SOS"]], lp_c[cd.x2i["SOS"]]]))
#
#     psp = (dy.concatenate([lp_p[td.x2i["SOS"]], lp_ps[tsd.x2i["SOS"]]]))
#     wft = (dy.concatenate([lp_wif[wifd.x2i["SOS"]], lp_wit[witd.x2i["SOS"]]]))
#     wch = (dy.concatenate([lp_w[wd.x2i["SOS"]], lp_c[cd.x2i["SOS"]]]))
#
#     R_w = dy.parameter(params["R_w"])
#     bias_w = dy.parameter(params["bias_w"])
#
#     R_p = dy.parameter(params["R_p"])
#     bias_p = dy.parameter(params["bias_p"])
#
#     R_wi = dy.parameter(params["R_wi"])
#     bias_wi = dy.parameter(params["bias_wi"])
#
#     R_b3 = dy.parameter(params["R_b3"])
#     b3_bias = dy.parameter(params["b3_bias"])
#
#     # filter_2d = dy.parameter(params["filter_2d"])
#
#     # wembs.append(dy.cube(R_p * psp + R_wi * wft + R_w * wch + bias_w + bias_p + bias_wi))
#     wembs.append((R_p * psp + R_wi * wft + R_w * wch + bias_w + bias_p + bias_wi))
#
#     for idx, wr in enumerate(word_ranges):
#         str = ""
#         tmp_char = []
#         tmp_pos = []
#
#         for c in char_seq[wr[0]: wr[1]]:
#             str += cd.i2x[c]
#             tmp_char.append(dy.dropout(lp_c[c], pdrop))
#
#         for p in pos_seq[wr[0]: wr[1]]:
#             tmp_pos.append(lp_p[p])
#
#         tmp_char_len = len(tmp_char)
#
#         tmp_char = dy.esum(tmp_char) / tmp_char_len
#
#         pos_sub_pos = bias_p + R_p * dy.concatenate([lp_p[pos_seq[wr[0]]], lp_ps[pos_sub_seq[idx]]])
#         # pos_sub_pos = dy.concatenate([lp_p[pos_seq[wr[0]]], lp_ps[pos_sub_seq[idx]]])
#         wif_wit = bias_wi + R_wi * dy.concatenate([lp_wif[wif_seq[idx]], lp_wit[wit_seq[idx]]])
#         # wif_wit = dy.concatenate([lp_wif[wif_seq[idx]], lp_wit[wit_seq[idx]]])
#         tmp_word = pos_sub_pos + wif_wit
#         # tmp_word = dy.concatenate([pos_sub_pos, wif_wit])
#         if TRAIN:
#             rnd_int = np.random.randint(0, 2)
#
#             if str in wd.x2i and rnd_int == 0:
#                 # tmp_word = dy.concatenate([dy.dropout(lp_w[wd.x2i[str]], pdrop), tmp_char])
#                 # tmp_word = dy.concatenate([dy.dropout(lp_w[wd.x2i[str]], pdrop), tmp_word])
#                 tmp_word = bias_w + R_w * dy.concatenate(
#                     [dy.dropout(lp_w[wd.x2i[str]], pdrop), tmp_char]) + tmp_word
#             else:
#                 # tmp_word = dy.concatenate([tmp_char, tmp_char])
#                 tmp_word = bias_w + R_w * dy.concatenate(
#                     [tmp_char, tmp_char]) + tmp_word
#         else:
#             if str in wd.x2i:
#                 # tmp_word = dy.concatenate([lp_w[wd.x2i[str]], tmp_char])
#                 # tmp_word = dy.concatenate([lp_w[wd.x2i[str]], tmp_word])
#                 tmp_word = bias_w + R_w * dy.concatenate(
#                     [lp_w[wd.x2i[str]], tmp_char]) + tmp_word
#             else:
#                 # tmp_word = dy.concatenate([tmp_char, tmp_char])
#                 # tmp_word = dy.concatenate([tmp_char, tmp_word])
#                 tmp_word = bias_w + R_w * dy.concatenate(
#                     [tmp_char, tmp_char]) + tmp_word
#
#         psp_seq.append(pos_sub_pos)
#         wft_seq.append(wif_wit)
#         wch_seq.append(tmp_word)
#
#         # wembs.append(dy.cube(tmp_word))
#         wembs.append((tmp_word))
#
#         # if wr[1] < len(cembs):
#         #     ret.append((dy.concatenate([cembs[wr[1]] - cembs[wr[0]], tmp_word])))
#         # elif wr[0] == len(cembs) - 1:
#         #     ret.append((dy.concatenate([cembs[wr[0]], tmp_word])))
#         # else:
#         #     ret.append((dy.concatenate([cembs[-1] - cembs[wr[0]], tmp_word])))
#     psp_seq.append(dy.concatenate([lp_p[td.x2i["EOS"]], lp_ps[tsd.x2i["EOS"]]]))
#     wft_seq.append(dy.concatenate([lp_wif[wifd.x2i["EOS"]], lp_wit[witd.x2i["EOS"]]]))
#     wch_seq.append(dy.concatenate([lp_w[wd.x2i["EOS"]], lp_c[cd.x2i["EOS"]]]))
#
#     psp = (dy.concatenate([lp_p[td.x2i["EOS"]], lp_ps[tsd.x2i["EOS"]]]))
#     wft = (dy.concatenate([lp_wif[wifd.x2i["EOS"]], lp_wit[witd.x2i["EOS"]]]))
#     wch = (dy.concatenate([lp_w[wd.x2i["EOS"]], lp_c[cd.x2i["EOS"]]]))
#
#     # wembs.append(dy.cube(R_p * psp + R_wi * wft + R_w * wch + bias_w))
#     wembs.append((R_p * psp + R_wi * wft + R_w * wch + bias_w + bias_wi + bias_p))
#
#     for widx in range(1, len(wembs) - 1):
#         # psp = dy.concatenate([psp_seq[widx - 1], psp_seq[widx], psp_seq[widx + 1]])
#         # wft = dy.concatenate([wft_seq[widx - 1], wft_seq[widx], wft_seq[widx + 1]])
#         # wch = dy.concatenate([wch_seq[widx - 1], wch_seq[widx], wch_seq[widx + 1]])
#         # wemb = dy.concatenate([psp, wft, wch], 1)
#         # conv1 = dy.conv2d(wemb, filter_2d, [2, 2])
#         # maxpool1 = dy.maxpooling2d(conv1, [2, 2], [1, 1])
#
#         ret.append(dy.cube(R_p * psp_seq[widx] + R_wi * wft_seq[widx] + R_w * wch_seq[widx] + bias_w))
#         # ret.append(b3_bias + R_b3 * dy.concatenate([wembs[widx - 1], wembs[widx], wembs[widx + 1]]))
#
#     return ret


# def bunsetsu_embds(wembs, bunsetsu_ranges):
#     ret3 = []
#
#     if bemb_with_lstm:
#         for br in bunsetsu_ranges:
#             tmp = []
#             if br[1] < len(wembs):
#                 tmp.extend(inputs2lstmouts(l2rlstm_bemb, r2llstm_bemb, wembs[br[0]: br[1]], pdrop))
#             elif br[0] == len(wembs) - 1:
#                 tmp.extend(inputs2lstmouts(l2rlstm_bemb, r2llstm_bemb, [wembs[br[0]]], pdrop))
#             else:
#                 tmp.extend(inputs2lstmouts(l2rlstm_bemb, r2llstm_bemb, wembs[br[0]: br[-1]], pdrop))
#             tmp_len = len(tmp)
#             ret3.append(dy.esum(tmp) / tmp_len)
#         return ret3
#
#
#     ret = []
#     ret2 = []
#
#     if bemb_attention:
#         R_word_biaffine = dy.parameter(params["R_bemb_biaffine"])
#
#     if few_words_bemb:
#         R_b1 = dy.parameter(params["R_b1"])
#         R_b2 = dy.parameter(params["R_b2"])
#         R_b3 = dy.parameter(params["R_b3"])
#         b1_bias = dy.parameter(params["b1_bias"])
#         b2_bias = dy.parameter(params["b2_bias"])
#         b3_bias = dy.parameter(params["b3_bias"])
#
#
#     for br in bunsetsu_ranges:
#         if br[0] == 0:
#             ret.append(dy.concatenate([wembs[1] - wembs[0], wembs[0] - wembs[1]]))
#         elif br[1] < len(wembs):
#             ret.append(dy.concatenate([wembs[br[1]] - wembs[br[0]], wembs[br[0]] - wembs[br[1]]]))
#             welms = wembs[br[0]: br[1]]
#         # elif br[0] == len(wembs) - 1:
#         #     ret.append(wembs[br[0]])
#         #     welms = [wembs[br[0]]]
#         else:
#             ret.append(dy.concatenate([wembs[-1] - wembs[br[0]], wembs[br[0]] - wembs[-1]]))
#             # ret.append(wembs[-1] - wembs[br[0]])
#             welms = wembs[br[0]: br[-1]]
#         if bemb_attention:
#             y = ret[-1]
#             x = dy.concatenate(welms, 1)
#             attention = dy.softsign(dy.transpose(bilinear(x, R_word_biaffine, y, word_HIDDEN_DIM, len(welms), 1, 1)))
#             bemb = dy.cmult(dy.transpose(attention), dy.dropout(x, pdrop))
#             bemb = dy.sum_dim(bemb, [1])
#
#             ret2.append(bemb)
#         if few_words_bemb:
#             if len(welms) == 1:
#                 bemb = R_b1 * dy.concatenate(welms) + b1_bias
#             elif len(welms) == 2:
#                 bemb = R_b2 * dy.concatenate(welms) + b2_bias
#             elif len(welms) == 3:
#                 bemb = R_b3 * dy.concatenate(welms[-3:]) + b3_bias
#             else:
#                 bemb = R_b3 * dy.concatenate(welms[-3:]) + dy.esum(welms[:-3]) + b3_bias
#             ret2.append(dy.dropout(bemb, pdrop))
#     if next_bemb:
#         bembs = ret
#         ret = []
#
#         for bidx in range(len(bembs)):
#             # ret.append(dy.concatenate([bembs[bidx - 1], bembs[bidx]]))
#             ret.append(dy.concatenate([bembs[bidx], bembs[(bidx + 1) % len(bembs)]]))
#
#     if few_words_bemb:
#         return ret2
#
#     if bemb_attention:
#         return ret2
#     else:
#         return ret



def bunsetsu_embds(l2r_outs, r2l_outs, bunsetsu_ranges):

    ret = []

    for br in bunsetsu_ranges:
        start = br[0] + 1
        end = br[1] + 1

        ret.append(dy.concatenate([l2r_outs[end] - l2r_outs[start], r2l_outs[start] - r2l_outs[end]]))

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
    losses_bunsetsu = []
    losses_arcs = []
    prev = time.time()

    tot_loss_in_iter = 0
    #trainer = dy.AdagradTrainer(pc, learning_rate)

    print(pdrop)
    print(pdrop_bunsetsu)

    for it in range(train_iter):
        print("total loss in previous iteration: ", tot_loss_in_iter)
        tot_loss_in_iter = 0
        print("iteration: ", it)
        num_tot_bunsetsu_dep = 0
        num_tot_cor_bunsetsu_dep = 0
        num_tot_cor_bunsetsu_dep_not_argmax = 0

        for i in (range(len(char_seqs))):
            if i % batch_size == 0:
                losses_bunsetsu = []
                losses_arcs = []

                dy.renew_cg()

            if random_pickup:  idx = i if not TRAIN else np.random.randint(len(char_seqs))
            else: idx = i

            if len(char_seqs[idx]) == 0 or len(bi_b_seqs[idx]) == 0:
                continue

            # cembs = char_embds(char_seqs[idx])
            # cembs = inputs2lstmouts(l2rlstm_char, r2llstm_char, cembs, pdrop)

            bi_w_seq = [0 if (bpd.i2x[b])[0] == 'B' else 1 for b in bipos_seqs[idx]]
            bi_w_seq = word_bi(bi_w_seq, bi_b_seqs[idx])

            word_ranges = word_range(bipos_seqs[idx])
            # wembs = word_embds(cembs, char_seqs[idx], train_pos_seqs[idx],
            #                    train_pos_sub_seqs[idx], train_wif_seqs[idx], train_wit_seqs[idx], word_ranges)

            wembs = word_embds(char_seqs[idx], train_pos_seqs[idx],
                               train_pos_sub_seqs[idx], train_wif_seqs[idx], train_wit_seqs[idx], word_ranges)
            if use_wembs:
                wembs, l2r_outs, r2l_outs = inputs2lstmouts_wembs(l2rlstm_word, r2llstm_word, wembs, pdrop)

            loss_bi_bunsetsu, _, _ = bi_bunsetsu_wembs(wembs, bi_w_seq)
            losses_bunsetsu.append(loss_bi_bunsetsu)

            if i % batch_size == 0 and i != 0:
                loss_bi_bunsetsu_value = loss_bi_bunsetsu.value()

            if i % show_loss_every == 0 and i != 0:
                print(i, " bi_bunsetsu loss")
                print(loss_bi_bunsetsu_value)

            bunsetsu_ranges = bunsetsu_range(bi_w_seq)

            bembs = bunsetsu_embds(l2r_outs, r2l_outs, bunsetsu_ranges)
            if bemb_lstm:
                bembs = inputs2lstmouts(l2rlstm_bunsetsu, r2llstm_bunsetsu, bembs, pdrop_bunsetsu)
            arc_loss, arc_preds, arc_preds_not_argmax = dep_bunsetsu(bembs)

            num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds, train_chunk_deps[idx]))
            num_tot_cor_bunsetsu_dep_not_argmax += np.sum(np.equal(arc_preds_not_argmax[1:], train_chunk_deps[idx]))

            num_tot_bunsetsu_dep += len(bembs) - 1

            losses_arcs.append(dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, train_chunk_deps[idx])))
            global global_step
            if i % batch_size == 0 and i != 0:
                losses_arcs.extend(losses_bunsetsu)

                sum_losses_arcs = dy.esum(losses_arcs)
                sum_losses_arcs_value = sum_losses_arcs.value()
                sum_losses_arcs.backward()
                update_parameters()
                global_step += 1
                #trainer.update()
                tot_loss_in_iter += sum_losses_arcs_value

            if i % show_loss_every == 0 and i != 0:
                print("time: ", time.time() - prev)
                prev = time.time()
                print(i, " arcs loss")
                print(sum_losses_arcs_value)
                print(train_sents[idx].word_forms)

                print(arc_preds)
                print(train_chunk_deps[idx])

                print("dep accuracy: ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)
                print("dep accuracy not argmax: ", num_tot_cor_bunsetsu_dep_not_argmax / num_tot_bunsetsu_dep)
    print("total loss in this epoch: ", tot_loss_in_iter)


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
    num_tot_cor_bunsetsu_dep_not_argmax = 0

    complete_chunking = 0
    failed_chunking = 0
    chunks_excluded = 0

    prev = time.time()

    print(pdrop)
    print(pdrop_bunsetsu)
    total_loss = 0

    for i in range(len(char_seqs)):
        dy.renew_cg()
        if len(char_seqs[i]) == 0:
            continue

        bi_w_seq = [0 if (bpd.i2x[b])[0] == 'B' else 1 for b in bipos_seqs[i]]
        bi_w_seq = word_bi(bi_w_seq, bi_b_seqs[i])

        # cembs = char_embds(char_seqs[i])
        # cembs = inputs2lstmouts(l2rlstm_char, r2llstm_char, cembs, pdrop)

        num_tot += len(char_seqs[i])

        word_ranges = word_range(bipos_seqs[i])
        # wembs = word_embds(cembs, char_seqs[i], dev_pos_seqs[i],
        #                    dev_pos_sub_seqs[i], dev_wif_seqs[i], dev_wit_seqs[i], word_ranges)
        wembs = word_embds(char_seqs[i], dev_pos_seqs[i],
                           dev_pos_sub_seqs[i], dev_wif_seqs[i], dev_wit_seqs[i], word_ranges)
        if use_wembs:
            wembs, l2r_outs, r2l_outs = inputs2lstmouts_wembs(l2rlstm_word, r2llstm_word, wembs, pdrop)

        loss_bi_b, preds_bi_b, num_cor_bi_b = bi_bunsetsu_wembs(wembs, bi_w_seq)
        num_tot_bi_b += len(wembs)
        num_tot_cor_bi_b += num_cor_bi_b
        if i % show_acc_every == 0 and i != 0:
            print("accuracy chunking: ", num_tot_cor_bi_b / num_tot_bi_b)
            print("loss chuncking: ", loss_bi_b.value())
        gold_bunsetsu_ranges = bunsetsu_range(bi_w_seq)

        failed_chunk = []
        for bidx, br in enumerate(gold_bunsetsu_ranges[1:]):
            start = br[0]
            end = br[1]
            if end == len(gold_bunsetsu_ranges):
                end = - 1
            if np.sum(np.equal(bi_w_seq[start: end], preds_bi_b[start: end])) != len(bi_w_seq[start: end]):
                failed_chunk.append(bidx)


        remains = [True] * len(gold_bunsetsu_ranges)
        for fc in failed_chunk:
            remains[fc]
            dev_chunk_deps[i]
            chunks_excluded += np.sum(np.equal(dev_chunk_deps[i], fc)) + remains[fc]
            remains = [r * (1 - d) for r, d in zip(remains, np.equal(dev_chunk_deps[i], fc))]
            remains[fc] = False


        # pred_bunsetsu_ranges = bunsetsu_range(preds_bi_b)

        bembs = bunsetsu_embds(l2r_outs, r2l_outs, gold_bunsetsu_ranges)
        if bemb_lstm:
            bembs = inputs2lstmouts(l2rlstm_bunsetsu, r2llstm_bunsetsu, bembs, pdrop_bunsetsu)

        if i % show_acc_every == 0 and i != 0:
            loss_bi_b_value = loss_bi_b.value()
            print(i, " bi_b loss")
            print(loss_bi_b_value)
            if num_tot_bi_b > 0:
                print(i, " accuracy chunking ", num_tot_cor_bi_b / num_tot_bi_b)
            if num_tot_bunsetsu_dep > 0:
                print(i, " accuracy dep ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)
                print(i, " accuracy dep ", num_tot_cor_bunsetsu_dep_not_argmax / num_tot_bunsetsu_dep)
            print("time: ", time.time() - prev)
            prev = time.time()
        if len(wembs) == num_cor_bi_b:
            complete_chunking += 1

        if len(dev_chunk_deps[i]) != len(bembs) - 1:
            failed_chunking += 1
            # continue
        arc_loss, arc_preds, arc_preds_not_argmax = dep_bunsetsu(bembs)

        num_tot_bunsetsu_dep += len(bembs) - 1

        # num_tot_cor_bunsetsu_dep += np.sum(np.equal(np.equal(arc_preds, dev_chunk_deps[i]), remains))
        num_tot_cor_bunsetsu_dep += np.sum([r * d for r, d in zip(remains, np.equal(arc_preds, dev_chunk_deps[i]))])
        #num_tot_cor_bunsetsu_dep_not_argmax += np.sum([r * d for r, d in zip(remains, np.equal(arc_preds_not_argmax[1:], dev_chunk_deps[i]))])
        num_tot_cor_bunsetsu_dep_not_argmax += np.sum(
            np.equal(arc_preds_not_argmax[1:], dev_chunk_deps[i]))

    global best_acc
    global update
    global early_stop_count
    if best_acc + 0.001 < num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep:
        best_acc = num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep
        update = True
        early_stop_count = 0

    with open(result_file, mode='a', encoding='utf-8') as f:
        f.write(str(i) + " accuracy chunking " + str(num_tot_cor_bi_b / num_tot_bi_b) + '\n')
        f.write(str(i) + " accuracy dep " + str(num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)+ '\n')
        f.write("complete chunking rate: " + str(complete_chunking / len(char_seqs))+ '\n')
        f.write("failed_chunking rate: " + str(failed_chunking / len(char_seqs))+ '\n')
        f.write("complete chunking: " + str(complete_chunking)+ '\n')
        f.write("failed_chunking: " + str(failed_chunking)+ '\n')
        #f.write("total arc loss: " + str(total_loss) + '\n')
    print("complete_chunking rate: " + str(complete_chunking / len(char_seqs)))
    print("failed_chunking rate: " + str(failed_chunking / len(char_seqs)))
    print("complete chunking: " + str(complete_chunking))
    print("failed_chunking: " + str(failed_chunking))
    print("chunks_excluded: ", chunks_excluded)
    #print("total arc loss: " + str(total_loss))
    return


prev = time.time()

if LOAD:
    pc.populate(load_file)
    print("loaded from: ", load_file)

for e in range(epoc):
    print("time: ", time.time() - prev)
    prev = time.time()
    print("epoc: ", e)
    with open(result_file, mode='a', encoding='utf-8') as f:
        f.write("time: " + str(prev) + '\n')
        f.write("epoc: " + str(e) + '\n')

    TRAIN = True
    global pdrop
    global pdrop_bunsetsu
    pdrop = pdrop_stash
    pdrop_bunsetsu = pdrop_bunsetsu_stash

    train(train_char_seqs, train_word_bipos_seqs, train_chunk_bi_seqs)

    pdrop = 0.0
    pdrop_bunsetsu = 0.0
    TRAIN = False
    update = False

    dev(dev_char_seqs, dev_word_bipos_seqs, dev_chunk_bi_seqs)

    global early_stop_count
    if not update:
        early_stop_count += 1

    if SAVE and update:
        pc.save(save_file)
        print("saved into: ", save_file)

    if early_stop_count > early_stop:
        print("best_acc: ", best_acc)
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write("best_acc: " + str(best_acc))

        break
