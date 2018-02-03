from collections import defaultdict

import numpy as np
import dynet as dy
import matplotlib.pyplot as plt
import os
import shutil

from config import *
import config
from utils import *
import getfiles
import get_pret_embs

from tarjan import *
import glob
import time

from paths import *
from file_reader import DataFrameKtc

train_loss = []
dev_loss = []
acc = []

def plot_loss(plt, loss, num_epoc, subplot_idx, xlim, ylim, ylim_lower):
    x = np.arange(0, num_epoc)
    y = np.array(loss)
    plt.subplot(2, 2, subplot_idx)
    plt.plot(x, y)
    plt.xlim(0, xlim)
    plt.ylim(ylim_lower, ylim)

    return

train_dev_boundary = -1
files = glob.glob(path2KTC + 'syn/*.*')

if CABOCHA_SPLIT:
    files = glob.glob(path2KTC + 'syn/95010[1-9].*')
    train_dev_boundary = -1
best_acc = 0.0
least_loss = 1000.0
update = False
global early_stop_count
early_stop_count = 0


# if STANDARD_SPLIT:
#     files = glob.glob(path2KTC + 'syn/95010[1-9].*')
#     files.extend(glob.glob(path2KTC + 'syn/95011[0-1].*'))
#     files.extend(glob.glob(path2KTC + 'syn/950[1-8]ED.*'))
#     if TEST:
#         files.extend(glob.glob(path2KTC + 'syn/95011[4-7].*'))
#         files.extend(glob.glob(path2KTC + 'syn/951[0-2]ED.*'))
#         train_dev_boundary = -7
#     else:
#         files.extend(glob.glob(path2KTC + 'syn/95011[2-3].*'))
#         files.extend(glob.glob(path2KTC + 'syn/9509ED.*'))
#         train_dev_boundary = -3
#
# if JOS:
#     files = glob.glob(path2KTC + 'just-one-sentence.txt')
#     files = [path2KTC + 'just-one-sentence.txt', path2KTC + 'just-one-sentence.txt']
#
# if MINI_SET:
#     files = [path2KTC + 'miniKTC_train.txt', path2KTC + 'miniKTC_dev.txt']
#
# save_file = 'KTC'
#
# split_name = ""
#
# if CABOCHA_SPLIT:
#     split_name = "_CABOCHA"
# elif STANDARD_SPLIT:
#     split_name = "_STANDARD"
# elif MINI_SET:
#     split_name = "_MINISET"
#
# save_file = save_file_directory + save_file + split_name
#
# print(files)

save_file = 'KTC'

split_name = ""

if config.CABOCHA_SPLIT:
    split_name = "_CABOCHA"
elif config.STANDARD_SPLIT:
    split_name = "_STANDARD"
elif config.MINI_SET:
    split_name = "_MINISET"

save_file = config.save_file_directory + save_file + split_name

files, train_dev_boundary = getfiles.getfiles()

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

train_vocab = set()
for sent in train_sents:
    for w in sent.word_forms:
        train_vocab.add(w)

dev_vocab = set()
for sent in dev_sents:
    for w in sent.word_forms:
        dev_vocab.add(w)

if config.add_pret_embs:
    pret_embs = get_pret_embs.get_pret_embs()
    pret_vocab = set(pret_embs.index2word)

    td_is = dev_vocab.intersection(train_vocab)
    tp_uni = train_vocab.union(pret_vocab)
    print(len(dev_vocab.difference(train_vocab)))
    print(len(dev_vocab.difference(tp_uni)))
    print(len(pret_vocab.intersection(train_vocab)))

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

###Neural Network
# WORDS_SIZE = len(wd.appeared_i2x) + 1
# CHARS_SIZE = len(cd.appeared_i2x) + 1
# BIPOS_SIZE = len(bpd.appeared_i2x) + 1
# POS_SIZE = len(td.appeared_i2x) + 1
# POSSUB_SIZE = len(tsd.appeared_i2x) + 1
# WIF_SIZE = len(wifd.appeared_i2x) + 1
# WIT_SIZE = len(witd.appeared_i2x) + 1

WORDS_SIZE = len(wd.i2x) + 1
CHARS_SIZE = len(cd.i2x) + 1
BIPOS_SIZE = len(bpd.i2x) + 1
POS_SIZE = len(td.i2x) + 1
POSSUB_SIZE = len(tsd.i2x) + 1
WIF_SIZE = len(wifd.i2x) + 1
WIT_SIZE = len(witd.i2x) + 1


pc = dy.ParameterCollection()

if not use_annealing:
    trainer = dy.AdadeltaTrainer(pc)
else:
    # trainer = dy.AdamTrainer(pc, config.learning_rate , config.beta_1, config.beta_2, config.epsilon)
    trainer = dy.SimpleSGDTrainer(pc, config.learning_rate)
    trainer.set_clip_threshold(3.)

global_step = 0

# def update_parameters():
#     if use_annealing:
#         trainer.learning_rate = config.learning_rate * decay ** (global_step / config.decay_steps)
#     trainer.update()
def update_parameters():
    if use_annealing:
        trainer.learning_rate = config.learning_rate / (1 + decay * (global_step - 1))
    trainer.update()


if not orthonormal:
    l2rlstm_char = dy.VanillaLSTMBuilder(LAYERS_character, config.char_INPUT_DIM * 1, config.char_HIDDEN_DIM, pc, layer_norm)
    r2llstm_char = dy.VanillaLSTMBuilder(LAYERS_character, config.char_INPUT_DIM * 1, config.char_HIDDEN_DIM, pc, layer_norm)

    l2rlstm_word = dy.VanillaLSTMBuilder(LAYERS_word, config.char_HIDDEN_DIM + INPUT_DIM * 4 + config.char_INPUT_DIM, word_HIDDEN_DIM, pc, layer_norm)
    r2llstm_word = dy.VanillaLSTMBuilder(LAYERS_word, config.char_HIDDEN_DIM + INPUT_DIM * 4 + config.char_INPUT_DIM, word_HIDDEN_DIM, pc, layer_norm)

    if bembs_average_flag:
        l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, INPUT_DIM * ((2 * (use_cembs) + 2) + use_wif_wit * 2), bunsetsu_HIDDEN_DIM, pc, layer_norm)
        r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, INPUT_DIM * ((2 * (use_cembs) + 2) + use_wif_wit * 2), bunsetsu_HIDDEN_DIM, pc, layer_norm)
    else:
        l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM + INPUT_DIM * 5, bunsetsu_HIDDEN_DIM, pc, layer_norm)
        r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM + INPUT_DIM * 5, bunsetsu_HIDDEN_DIM, pc, layer_norm)

else:
    l2rlstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, config.char_INPUT_DIM * 1, config.char_HIDDEN_DIM, pc)
    r2llstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, config.char_INPUT_DIM * 1, config.char_HIDDEN_DIM, pc)

    l2rlstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, config.char_HIDDEN_DIM + INPUT_DIM * 4 + config.char_INPUT_DIM, word_HIDDEN_DIM, pc)
    r2llstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, config.char_HIDDEN_DIM + INPUT_DIM * 4 + config.char_INPUT_DIM, word_HIDDEN_DIM, pc)

    if bembs_average_flag:
        l2rlstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, INPUT_DIM * ((2 * (use_cembs) + 2) + use_wif_wit * 2), bunsetsu_HIDDEN_DIM, pc)
        r2llstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, INPUT_DIM * ((2 * (use_cembs) + 2) + use_wif_wit * 2), bunsetsu_HIDDEN_DIM, pc)
    else:
        l2rlstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM + INPUT_DIM * 4 + config.char_INPUT_DIM, bunsetsu_HIDDEN_DIM, pc)
        r2llstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM + INPUT_DIM * 4 + config.char_INPUT_DIM, bunsetsu_HIDDEN_DIM, pc)


params = {}
# params["lp_w"] = pc.add_lookup_parameters((WORDS_SIZE + 1, INPUT_DIM), init=dy.ConstInitializer(0.))
params["lp_c"] = pc.add_lookup_parameters((CHARS_SIZE + 1, char_INPUT_DIM), init=dy.ConstInitializer(0.))
# params["lp_bp"] = pc.add_lookup_parameters((BIPOS_SIZE + 1, INPUT_DIM), init=dy.ConstInitializer(0.))
# params["lp_p"] = pc.add_lookup_parameters((POS_SIZE + 1, (INPUT_DIM // 10) * 5 * (1 + use_wif_wit)), init=dy.ConstInitializer(0.))
# params["lp_ps"] = pc.add_lookup_parameters((POSSUB_SIZE + 1, (INPUT_DIM // 10) * 5 * (1 + use_wif_wit)), init=dy.ConstInitializer(0.))
# params["lp_wif"] = pc.add_lookup_parameters((WIF_SIZE + 1, (INPUT_DIM // 10) * 5), init=dy.ConstInitializer(0.))
# params["lp_wit"] = pc.add_lookup_parameters((WIT_SIZE + 1, (INPUT_DIM // 10) * 5), init=dy.ConstInitializer(0.))

params["root_emb"] = pc.add_lookup_parameters((1, bunsetsu_HIDDEN_DIM * 2))

# params["word_lin"] = pc.add_parameters((INPUT_DIM, INPUT_DIM * 2))
# params["word_lin_bias"] = pc.add_parameters((INPUT_DIM))

params["R_bi_b"] = pc.add_parameters((2, word_HIDDEN_DIM * 2))
params["bias_bi_b"] = pc.add_parameters((2), init=dy.ConstInitializer(0.))

# params["cont_MLP"] = pc.add_parameters((word_HIDDEN_DIM, word_HIDDEN_DIM // (2 - wemb_lstm) * 2))
# params["cont_MLP_bias"] = pc.add_parameters((word_HIDDEN_DIM), init=dy.ConstInitializer(0.))
#
# params["func_MLP"] = pc.add_parameters((word_HIDDEN_DIM, word_HIDDEN_DIM // (2 - wemb_lstm) * 2))
# params["func_MLP_bias"] = pc.add_parameters((word_HIDDEN_DIM), init=dy.ConstInitializer(0.))

if not TEST or orthonormal:
    W = orthonormal_initializer(MLP_HIDDEN_DIM, 2 * bunsetsu_HIDDEN_DIM)
    params["head_MLP"] = pc.parameters_from_numpy(W)
    params["head_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

    params["dep_MLP"] = pc.parameters_from_numpy(W)
    params["dep_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

else:
    params["head_MLP"] = pc.add_parameters((MLP_HIDDEN_DIM, bunsetsu_HIDDEN_DIM * 2))
    params["head_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

    params["dep_MLP"] = pc.add_parameters((MLP_HIDDEN_DIM, bunsetsu_HIDDEN_DIM * 2))
    params["dep_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

params["R_bunsetsu_biaffine"] = pc.add_parameters((MLP_HIDDEN_DIM + biaffine_bias_y, MLP_HIDDEN_DIM + biaffine_bias_x), init=dy.ConstInitializer(0.))

def inputs2lstmouts(l2rlstm, r2llstm, inputs, pdrop):

    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    l2rlstm.set_dropouts(pdrop, .0)
    r2llstm.set_dropouts(pdrop, .0)

    s_l2r = s_l2r_0
    s_r2l = s_r2l_0

    l2r_outs = s_l2r.add_inputs(inputs)
    r2l_outs = s_r2l.add_inputs(reversed(inputs))

    lstm_outs = [dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()]) for i in range(len(l2r_outs))]
    l2r_outs = [l2r_outs[i].output() for i in range(len(l2r_outs))]
    r2l_outs = [r2l_outs[i].output() for i in range(len(r2l_outs))]

    return lstm_outs, l2r_outs, r2l_outs

def inputs2singlelstmouts(lstm, inputs, pdrop):

    s_0 = lstm.initial_state()

    lstm.set_dropouts(pdrop, .0)

    s = s_0

    outs = s.add_inputs(inputs)

    lstm_outs = [outs[i].output() for i in range(len(outs))]

    return lstm_outs



def bi_bunsetsu(embs_token, chunk_bi):
    num_cor = 0

    R_bi_b = dy.parameter(params["R_bi_b"])
    bias_bi_b = dy.parameter(params["bias_bi_b"])
    loss = []
    preds = []

    for i in range(len(chunk_bi)):
        probs = dy.softmax(R_bi_b * embs_token[i] + bias_bi_b)
        loss.append(-dy.log(dy.pick(probs, chunk_bi[i])))

        if show_acc or scheduled_learning or not TRAIN:
            chosen = np.asscalar(np.argmax(probs.npvalue()))
            preds.append(chosen)
            if(chosen == chunk_bi[i]):
                # print(chosen, " ", bi_b_seq[w_1st_chars[i]])
                num_cor += 1
    loss = dy.esum(loss)
    return loss, preds, num_cor


def dep_bunsetsu(bembs, pdrop):
    root_emb = params["root_emb"]

    dep_MLP = dy.parameter(params["dep_MLP"])
    dep_MLP_bias = dy.parameter(params["dep_MLP_bias"])
    head_MLP = dy.parameter(params["head_MLP"])
    head_MLP_bias = dy.parameter(params["head_MLP_bias"])

    R_bunsetsu_biaffine = dy.parameter(params["R_bunsetsu_biaffine"])

    input_size = bembs[0].dim()[0][0]

    bembs = [root_emb[0]] + bembs

    slen_x = slen_y = len(bembs)

    bembs_dep = bembs_head = bembs

    bembs_dep = dy.dropout(
        dy.concatenate(bembs_dep, 1), pdrop)
    bembs_head = dy.dropout(
        dy.concatenate(bembs_head, 1), pdrop)

    bembs_dep = dy.dropout(leaky_relu(
        dy.affine_transform([dep_MLP_bias, dep_MLP, bembs_dep])), pdrop)
    bembs_head = dy.dropout(leaky_relu(
        dy.affine_transform([head_MLP_bias, head_MLP, bembs_head])), pdrop)

    blin = bilinear(bembs_dep, R_bunsetsu_biaffine, bembs_head, input_size, slen_x, slen_y, 1, 1, biaffine_bias_x, biaffine_bias_y)

    arc_preds = []
    arc_preds_not_argmax = []

    if show_acc or scheduled_learning or not TRAIN:
        arc_preds_not_argmax = blin.npvalue().argmax(0)
        msk = [1] * slen_x
        arc_probs = dy.softmax(blin).npvalue()
        arc_probs = np.transpose(arc_probs)
        arc_preds = arc_argmax(arc_probs, slen_x, msk)
        arc_preds = arc_preds[1:]
    arc_loss = dy.reshape(blin, (slen_y,), slen_x)
    arc_loss = dy.pick_batch_elems(arc_loss, [i for i in range(1, slen_x)])

    return arc_loss, arc_preds, arc_preds_not_argmax

def ranges(bi_seq):
    ret = []
    start = 0

    for i in range(1, len(bi_seq)):
        if bi_seq[i] == 0:
            end = i
            ret.append((start, end))
            start = i

    ret.append((start, len(bi_seq)))

    return ret

def segment_embds(l2r_outs, r2l_outs, ranges, offset=0, segment_concat=False):
    ret = []
    l2rs = []
    r2ls = []
    if offset == 0:
        st = 0
        en = len(ranges)
    elif offset == -1:
        st = 0
        en = len(ranges)
        offset = 0
    else:
        st = offset
        en = -offset

    for r in ranges[st:en]:
        start = r[0] - offset
        end = r[1] - offset

        if segment_concat:
            l2r = dy.concatenate([l2r_outs[end - 1], l2r_outs[start]])
            r2l = dy.concatenate([r2l_outs[start], r2l_outs[end - 1]])
        else:
            l2r = l2r_outs[end] - l2r_outs[start]
            r2l = r2l_outs[start + 1] - r2l_outs[end + 1]

        ret.append(dy.concatenate([l2r, r2l]))
        l2rs.append(l2r)
        r2ls.append(r2l)

    return ret, l2rs, r2ls

def embd_mask_generator(pdrop, slen):
    if not TRAIN:
        pdrop = 0.0

    masks_w = np.random.binomial(1, 1 - pdrop, slen)
    masks_t = np.random.binomial(1, 1 - pdrop, slen)
    scales = [3. / (2. * mask_w + mask_t + 1e-12) for mask_w, mask_t in zip(masks_w, masks_t)]
    masks_w = [mask_w * scale for mask_w, scale in zip(masks_w, scales)]
    masks_t = [mask_t * scale for mask_t, scale in zip(masks_t, scales)]
    return masks_w, masks_t

def char_embds(char_seq, pdrop):
    lp_c = params["lp_c"]
    pret_chars = []
    if not TRAIN:
        pdrop = 0.0

    if config.add_pret_embs:
        for i in range(len(char_seq)):
            c = cd.i2x[char_seq[i]]

            if c in pret_embs:
                pret_chars.append(dy.inputTensor(pret_embs[c]))
            else:
                pret_chars.append(dy.inputTensor(np.zeros(char_INPUT_DIM, dtype=np.float32)))
        cembs = [dy.dropout(lp_c[char_seq[i]] + pret_chars[i], pdrop)for i in range(len(char_seq))]
    else:
        cembs = [dy.dropout(lp_c[char_seq[i]], pdrop) for i in range(len(char_seq))]

    return cembs


params["pred_pos"] = pc.add_parameters((POS_SIZE, char_HIDDEN_DIM*2), init=dy.ConstInitializer(0.))
params["pred_pos_bias"] = pc.add_parameters(POS_SIZE, init=dy.ConstInitializer(0.))
params["pred_possub"] = pc.add_parameters((POSSUB_SIZE, char_HIDDEN_DIM*2), init=dy.ConstInitializer(0.))
params["pred_possub_bias"] = pc.add_parameters(POSSUB_SIZE, init=dy.ConstInitializer(0.))
params["pred_wif"] = pc.add_parameters((WIF_SIZE, char_HIDDEN_DIM*2), init=dy.ConstInitializer(0.))
params["pred_wif_bias"] = pc.add_parameters(WIF_SIZE, init=dy.ConstInitializer(0.))
params["pred_wit"] = pc.add_parameters((WIT_SIZE, char_HIDDEN_DIM*2), init=dy.ConstInitializer(0.))
params["pred_wit_bias"] = pc.add_parameters(WIT_SIZE, init=dy.ConstInitializer(0.))

params["param_pos"] = pc.add_parameters((INPUT_DIM, POS_SIZE))
params["param_possub"] = pc.add_parameters((INPUT_DIM, POSSUB_SIZE))
params["param_wif"] = pc.add_parameters((INPUT_DIM, WIF_SIZE))
params["param_wit"] = pc.add_parameters((INPUT_DIM, WIT_SIZE))

# probs = dy.softmax(R_bi_b * wembs[i] + bias_bi_b)
# loss.append(-dy.log(dy.pick(probs, chunk_bi[i])))


def params_regularization(param_pos_prev, param_possub_prev, param_wif_prev, param_wit_prev, lp_c_prev):
    param_pos = dy.parameter(params["param_pos"])
    param_possub = dy.parameter(params["param_possub"])
    param_wif = dy.parameter(params["param_wif"])
    param_wit = dy.parameter(params["param_wit"])

    # pred_pos = dy.parameter(params["pred_pos"])
    # pred_pos_bias = dy.parameter(params["pred_pos_bias"])
    # pred_possub = dy.parameter(params["pred_possub"])
    # pred_possub_bias = dy.parameter(params["pred_possub_bias"])
    # pred_wif = dy.parameter(params["pred_wif"])
    # pred_wif_bias = dy.parameter(params["pred_wif_bias"])
    # pred_wit = dy.parameter(params["pred_wit"])
    # pred_wit_bias = dy.parameter(params["pred_wit_bias"])

    lp_c = dy.parameter(params["lp_c"])


    # loss = dy.sum_elems(param_pos - dy.inputTensor(param_pos_prev)) + \
    #        dy.sum_elems(param_possub - dy.inputTensor(param_possub_prev)) + \
    #        dy.sum_elems(param_wif - dy.inputTensor(param_wif_prev)) + \
    #        dy.sum_elems(param_wit - dy.inputTensor(param_wit_prev))

    loss = dy.scalarInput(0)

    if config.reg_feats:
        loss += dy.squared_distance(param_pos, dy.inputTensor(param_pos_prev)) + \
               dy.squared_distance(param_possub, dy.inputTensor(param_possub_prev)) + \
               dy.squared_distance(param_wif, dy.inputTensor(param_wif_prev)) + \
               dy.squared_distance(param_wit, dy.inputTensor(param_wit_prev))
    if config.reg_lpc:
        loss += dy.squared_distance(lp_c, dy.inputTensor(lp_c_prev))

    return loss * config.params_delta




def pred_feats(lstmouts, feats=None):
    slen = len(lstmouts)

    pred_pos = dy.parameter(params["pred_pos"])
    pred_pos_bias = dy.parameter(params["pred_pos_bias"])
    pred_possub = dy.parameter(params["pred_possub"])
    pred_possub_bias = dy.parameter(params["pred_possub_bias"])
    pred_wif = dy.parameter(params["pred_wif"])
    pred_wif_bias = dy.parameter(params["pred_wif_bias"])
    pred_wit = dy.parameter(params["pred_wit"])
    pred_wit_bias = dy.parameter(params["pred_wit_bias"])

    param_pos = dy.parameter(params["param_pos"])
    param_possub = dy.parameter(params["param_possub"])
    param_wif = dy.parameter(params["param_wif"])
    param_wit = dy.parameter(params["param_wit"])

    smx_pos = []
    smx_possub = []
    smx_wif = []
    smx_wit = []

    emb_pos = []
    emb_possub = []
    emb_wif = []
    emb_wit = []

    loss = []

    for ci in range(len(lstmouts)):
        # smx_pos.append(dy.softmax(dy.affine_transform([pred_pos_bias, pred_pos, lstmouts[c]])))
        # smx_possub.append(dy.softmax(dy.affine_transform([pred_possub_bias, pred_possub, lstmouts[c]])))
        # smx_wif.append(dy.softmax(dy.affine_transform([pred_wif_bias, pred_wif, lstmouts[c]])))
        # smx_wit.append(dy.softmax(dy.affine_transform([pred_wit_bias, pred_wit, lstmouts[c]])))
        tmp_smx_pos = (dy.softmax(dy.affine_transform([pred_pos_bias, pred_pos, lstmouts[ci]])))
        tmp_smx_possub = (dy.softmax(dy.affine_transform([pred_possub_bias, pred_possub, lstmouts[ci]])))
        tmp_smx_wif = (dy.softmax(dy.affine_transform([pred_wif_bias, pred_wif, lstmouts[ci]])))
        tmp_smx_wit = (dy.softmax(dy.affine_transform([pred_wit_bias, pred_wit, lstmouts[ci]])))

        # tmp_ssn_pos = (dy.softsign(dy.affine_transform([pred_pos_bias, pred_pos, lstmouts[ci]])))
        # tmp_ssn_possub = (dy.softsign(dy.affine_transform([pred_possub_bias, pred_possub, lstmouts[ci]])))
        # tmp_ssn_wif = (dy.softsign(dy.affine_transform([pred_wif_bias, pred_wif, lstmouts[ci]])))
        # tmp_ssn_wit = (dy.softsign(dy.affine_transform([pred_wit_bias, pred_wit, lstmouts[ci]])))

        loss.append(-dy.log(dy.pick(tmp_smx_pos, feats[0][ci])))
        loss.append(-dy.log(dy.pick(tmp_smx_possub, feats[1][ci])))
        loss.append(-dy.log(dy.pick(tmp_smx_wif, feats[2][ci])))
        loss.append(-dy.log(dy.pick(tmp_smx_wit, feats[3][ci])))

        emb_pos.append(param_pos * tmp_smx_pos)
        emb_possub.append(param_possub * tmp_smx_possub)
        emb_wif.append(param_wif * tmp_smx_wif)
        emb_wit.append(param_wit * tmp_smx_wit)

        # emb_pos.append(param_pos * tmp_ssn_pos)
        # emb_possub.append(param_possub * tmp_ssn_possub)
        # emb_wif.append(param_wif * tmp_ssn_wif)
        # emb_wit.append(param_wit * tmp_ssn_wit)

    return dy.esum(loss), emb_pos, emb_possub, emb_wif, emb_wit

    # if feats is not None:
    #     loss = []
    #     lstmouts = dy.concatenate_cols(lstmouts)
    #
    #     expr_pos = dy.affine_transform([pred_pos_bias, pred_pos, lstmouts])
    #     expr_possub = dy.affine_transform([pred_possub_bias, pred_possub, lstmouts])
    #     expr_wif = dy.affine_transform([pred_wif_bias, pred_wif, lstmouts])
    #     expr_wit = dy.affine_transform([pred_wit_bias, pred_wit, lstmouts])
    #
    #     batched_pos = dy.reshape(expr_pos, (POS_SIZE, ), slen)
    #     batched_possub = dy.reshape(expr_possub, (POSSUB_SIZE, ), slen)
    #     batched_wif = dy.reshape(expr_wif, (WIF_SIZE, ), slen)
    #     batched_wit = dy.reshape(expr_wit, (WIT_SIZE, ), slen)
    #
    #     loss.append(dy.sum_batches(dy.pickneglogsoftmax_batch(batched_pos, feats[0])))
    #     loss.append(dy.sum_batches(dy.pickneglogsoftmax_batch(batched_possub, feats[1])))
    #     loss.append(dy.sum_batches(dy.pickneglogsoftmax_batch(batched_wif, feats[2])))
    #     loss.append(dy.sum_batches(dy.pickneglogsoftmax_batch(batched_wit, feats[3])))
    #
    #     return loss
    # else:
    #     dy.softmax(batched_pos)
    #     loss.append(dy.sum_batches(dy.pickneglogsoftmax_batch(batched_possub, feats[1])))
    #     loss.append(dy.sum_batches(dy.pickneglogsoftmax_batch(batched_wif, feats[2])))
    #     loss.append(dy.sum_batches(dy.pickneglogsoftmax_batch(batched_wit, feats[3])))






def word_embds(char_seq, pos_seq, pos_sub_seq, wif_seq, wit_seq, word_ranges):
    lp_w = params["lp_w"]
    lp_p = params["lp_p"]
    lp_ps = params["lp_ps"]
    lp_wif = params["lp_wif"]
    lp_wit = params["lp_wit"]

    word_lin = dy.parameter(params["word_lin"])
    word_lin_bias = dy.parameter(params["word_lin_bias"])

    wembs_w = []
    wembs_t = []
    masks_w, masks_t = embd_mask_generator(pdrop_embs, len(pos_seq))

    for idx, wr in enumerate(word_ranges):
        str = ""

        for c in char_seq[wr[0]: wr[1]]:
            str += cd.i2x[c]

        pos_lp = dy.concatenate([lp_p[pos_seq[idx]], lp_ps[pos_sub_seq[idx]]])

        if str in pret_embs and config.add_pret_embs:
            pret_emb = dy.inputTensor(pret_embs[str])
        else:
            pret_emb = dy.inputTensor(np.zeros(INPUT_DIM, dtype=np.float32))

        if config.w2v200:
            pret_emb = dy.affine_transform([word_lin_bias, word_lin, pret_emb])

        if str in wd.x2i:
            if config.use_wif_wit:
                word_form = dy.concatenate([lp_w[wd.x2i[str]] + pret_emb, lp_wif[wif_seq[idx]], lp_wit[wit_seq[idx]]])
            else:
                word_form = lp_w[wd.x2i[str]] + pret_emb
        else:
            if config.use_wif_wit:
                word_form = dy.concatenate([lp_w[wd.x2i["UNK"]] + pret_emb, lp_wif[wif_seq[idx]], lp_wit[wit_seq[idx]]])
            else:
                word_form = lp_w[wd.x2i["UNK"]] + pret_emb

        wembs_w.append(word_form)
        wembs_t.append(pos_lp)

    return [dy.concatenate([wemb_w * mask_w, wemb_t * mask_t])
            for wemb_w, wemb_t, mask_w, mask_t in zip(wembs_w, wembs_t, masks_w, masks_t)]


def bunsetsu_embds(l2r_outs, r2l_outs, bunsetsu_ranges, aux_position, pdrop):

    ret = []

    bembs_l2r = []
    bembs_r2l = []

    for br, aux_idx in zip(bunsetsu_ranges, aux_position):
        start = br[0]
        end = br[1]

        if not cont_aux_separated:
            ret.append(dy.concatenate([dy.dropout(leaky_relu(l2r_outs[end] - l2r_outs[start]), pdrop),
                                       dy.dropout(leaky_relu(r2l_outs[start + 1] - r2l_outs[end + 1]), pdrop),
                                       ]))
            bembs_l2r.append(l2r_outs[end] - l2r_outs[start])
            bembs_r2l.append(r2l_outs[start + 1] - r2l_outs[end + 1])
        else:

            bembs_l2r.append(l2r_outs[start + aux_idx] - l2r_outs[start])
            bembs_l2r.append(l2r_outs[end] -l2r_outs[start + aux_idx])

            bembs_r2l.append(r2l_outs[start + 1] - r2l_outs[start + aux_idx + 1])
            bembs_r2l.append(r2l_outs[start + aux_idx + 1] - r2l_outs[end + 1])

    return ret, bembs_l2r, bembs_r2l


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


def word_pos(pos_seq, word_ranges):
    ret = []

    for wr in word_ranges:
        ret.append(pos_seq[wr[0]])

    return ret


def word_bi(bi_w_seq, bi_b_seq):
    ret = []

    for i in range(len(bi_w_seq)):
        if bi_w_seq[i] == 0:
            if bi_b_seq[i] == 0:
                ret.append(0)
            else:
                ret.append(1)

    return ret


def aux_position(bunsetsu_ranges, pos_seq, pos_sub_seq):
    ret = []

    for br in bunsetsu_ranges:
        ret.append(-1)
        for widx in range(br[1] - br[0]):
            ch1 = (td.i2x[pos_seq[1:][br[0] + widx]])[0]
            ch2 = (tsd.i2x[pos_sub_seq[1:][br[0] + widx]])[-1]

            if ch1 == '助' or ch1 == '判' or ch1 == 'D' or ch2 == '点':
                ret[-1] = widx + 1
                break

    return ret


def bembs_average(wembs, ranges):
    ret = []
    for r in ranges:
        tmp = []
        for widx in range(r[0], r[1]):
            tmp.append(wembs[widx + 1])
        ret.append(dy.average(tmp))
    return ret

def bembs_MLPs(l2r_outs, r2l_outs):
    ret = []

    cont_MLP_bias = dy.parameter(params["cont_MLP_bias"])
    cont_MLP = dy.parameter(params["cont_MLP"])
    func_MLP_bias = dy.parameter(params["func_MLP_bias"])
    func_MLP = dy.parameter(params["func_MLP"])

    for idx in range(0, len(l2r_outs), 2):
        tmp = dy.concatenate([
            dy.affine_transform([cont_MLP_bias, cont_MLP, dy.concatenate([l2r_outs[idx], r2l_outs[idx]])]),
            dy.affine_transform([func_MLP_bias, func_MLP, dy.concatenate([l2r_outs[idx + 1], r2l_outs[idx + 1]])])
        ])
        ret.append(
            dy.cube(tmp)
        )

    return ret

cor_parsed_count = defaultdict(int)
done_sents = set()

def train(char_seqs,
          pos_seqs,
          pos_sub_seqs,
          wif_seqs,
          wit_seqs,
          word_bi_seqs,
          chunk_bi_seqs):
    losses_bunsetsu = []
    losses_arcs = []
    losses_chunk = []
    losses_feats = []
    prev = time.time()

    # param_pos_prev = dy.parameter(params["param_pos"])
    # param_possub_prev = dy.parameter(params["param_possub"])
    # param_wif_prev = dy.parameter(params["param_wif"])
    # param_wit_prev = dy.parameter(params["param_wit"])

    param_pos_prev = dy.parameter(params["param_pos"]).npvalue()
    param_possub_prev = dy.parameter(params["param_possub"]).npvalue()
    param_wif_prev = dy.parameter(params["param_wif"]).npvalue()
    param_wit_prev = dy.parameter(params["param_wit"]).npvalue()
    lp_c_prev = params["lp_c"].expr().npvalue()

    for it in range(train_iter):
        dy.renew_cg()

        num_tot_bunsetsu_dep = 0
        num_tot_cor_bunsetsu_dep = 0
        num_tot_cor_bunsetsu_dep_not_argmax = 0
        tot_loss_in_iter = 0
        tot_loss_feats = 0
        sent_ids = [sent_id for sent_id in range(len(char_seqs))]
        np.random.shuffle(sent_ids)

        for i in range(len(char_seqs) // divide_train):

            if random_pickup:
                idx = i if not TRAIN else sent_ids[i]
            else:
                idx = i

            if len(char_seqs[idx]) == 0 or len(chunk_bi_seqs[idx]) == 0:
                continue

            if char_base:
                bunsetsu_ranges = ranges([0] + train_sents[idx].chunk_bi_char + [0])
                chunk_bi_char = train_sents[idx].chunk_bi_char

                cembs = char_embds(char_seqs[idx], pdrop_cemb)

                cembs_lstmouts, l2rs_char, r2ls_char = inputs2lstmouts(l2rlstm_char, r2llstm_char, cembs, pdrop_lstm)
                loss_feats, embs_pos, embs_possub, embs_wif, embs_wit = \
                    pred_feats(cembs_lstmouts, [pos_seqs[idx], pos_sub_seqs[idx], wif_seqs[idx], wit_seqs[idx]])
                losses_feats.append(loss_feats)

                if not config.share_cembs:
                    cembs = cembs_lstmouts

                embs_feats = [dy.concatenate([cemb, pos, possub, wif, wit])
                             for cemb, pos, possub, wif, wit in zip(cembs, embs_pos, embs_possub, embs_wif, embs_wit)]

                cembs_l2r = [dy.concatenate([l2r_char, emb_feats])
                             for l2r_char, emb_feats in zip(l2rs_char, embs_feats)]
                cembs_r2l = [dy.concatenate([r2l_char, emb_feats])
                             for r2l_char, emb_feats in zip(r2ls_char, embs_feats)]

                l2rs_char, r2ls_char = inputs2singlelstmouts(l2rlstm_word, cembs_l2r, pdrop_lstm), \
                                       inputs2singlelstmouts(r2llstm_word, cembs_r2l, pdrop_lstm)

                cembs_bidir = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(l2rs_char, r2ls_char)]
                loss_chunk, preds_chunk, cor_chunk = bi_bunsetsu(cembs_bidir, chunk_bi_char)
                losses_chunk.append(loss_chunk)

                l2rs_char, r2ls_char = \
                    [dy.concatenate([l2r_char, emb_feats]) for l2r_char, emb_feats in zip(l2rs_char, embs_feats)], \
                    [dy.concatenate([r2l_char, emb_feats]) for r2l_char, emb_feats in zip(r2ls_char, embs_feats)]

                # cembs, l2rs_char, r2ls_char = inputs2lstmouts(l2rlstm_word, r2llstm_word, cembs, pdrop_lstm)
                bembs, l2rs_bunsetsu, r2ls_bunsetsu = segment_embds(l2rs_char, r2ls_char, bunsetsu_ranges, offset=1)
                bembs_l2r, bembs_r2l = inputs2singlelstmouts(l2rlstm_bunsetsu, l2rs_bunsetsu, pdrop_lstm), \
                                       inputs2singlelstmouts(r2llstm_bunsetsu, r2ls_bunsetsu, pdrop_lstm)

                bembs = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(bembs_l2r, bembs_r2l)]

                if i % batch_pos == 0 or i == len(char_seqs) - 1:
                # if i % batch_size == 0 or i == len(char_seqs) - 1:
                    loss_feats_batch = dy.esum(losses_feats) + \
                        params_regularization(
                            param_pos_prev,
                            param_possub_prev,
                            param_wif_prev,
                            param_wit_prev,
                            lp_c_prev
                        )
                    loss_feats_batch += dy.esum(losses_chunk)
                    losses_feats = []
                    losses_chunk = []
                    tot_loss_feats += loss_feats_batch.npvalue()
                    if i == len(char_seqs) - 1:
                        print("feats loss", tot_loss_feats)
                    loss_feats_batch.backward()
                    update_parameters()
                # print(trainer.get_clip_threshold())

            else:
                bi_w_seq = chunk_bi_seqs[idx]

                word_ranges = ranges(word_bi_seqs[idx])
                wembs = word_embds(char_seqs[idx], pos_seqs[idx],
                                   pos_sub_seqs[idx], wif_seqs[idx], wit_seqs[idx], word_ranges)

                if use_cembs:
                    cembs = char_embds(char_seqs[idx])
                    cembs, l2r_char, r2l_char = inputs2lstmouts(l2rlstm_char, r2llstm_char, cembs, pdrop_cemb)
                    cembs = segment_embds(l2r_char, r2l_char, word_ranges, offset=0, segment_concat=True)
                    wembs = [dy.concatenate([wemb, cemb]) for wemb, cemb in zip(wembs, cembs)]



                if wemb_lstm:
                    wembs, wembs_l2r, wembs_r2l = inputs2lstmouts(l2rlstm_word, r2llstm_word, wembs, pdrop_lstm)

                if relu_toprecur:
                    wembs = [leaky_relu(wemb) for wemb in wembs]

                if chunker:
                    loss_bi_bunsetsu, bi_bunsetsu_preds, _ = bi_bunsetsu(wembs, bi_w_seq)
                    losses_bunsetsu.append(loss_bi_bunsetsu)

                bunsetsu_ranges = ranges(bi_w_seq)
                aux_positions = aux_position(bunsetsu_ranges, pos_seqs[idx], pos_sub_seqs[idx])

                if wemb_lstm:
                    bembs, bembs_l2r, bembs_r2l = bunsetsu_embds(wembs_l2r, wembs_r2l, bunsetsu_ranges, aux_positions, pdrop_lstm)
                else:
                    if bembs_average_flag:
                        bembs = bembs_average(wembs, bunsetsu_ranges)
                    else:
                        bembs = bunsetsu_embds(wembs, wembs, bunsetsu_ranges, aux_positions, pdrop_lstm)

                if bemb_lstm:
                    if not bembs_average_flag:
                        bembs_l2r = inputs2singlelstmouts(l2rlstm_bunsetsu, bembs_l2r, pdrop_lstm)
                        bembs_r2l = inputs2singlelstmouts(r2llstm_bunsetsu, bembs_r2l, pdrop_lstm)
                        if cont_aux_separated:
                            bembs = bembs_MLPs(bembs_l2r, bembs_r2l)
                        else:
                            bembs = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(bembs_l2r, bembs_r2l)]
                    else:
                        bembs, _, _ = inputs2lstmouts(l2rlstm_bunsetsu, r2llstm_bunsetsu, bembs, pdrop_lstm)

                bembs = [dy.dropout(bemb, pdrop) for bemb in bembs]

            arc_loss, arc_preds, arc_preds_not_argmax = dep_bunsetsu(bembs, pdrop)

            if show_acc:
                num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds[:-1], train_chunk_deps[idx][:-1]))
                num_tot_cor_bunsetsu_dep_not_argmax += np.sum(np.equal(arc_preds_not_argmax[1:], train_chunk_deps[idx]))

            num_tot_bunsetsu_dep += len(bembs) - config.root - 1

            # loss_feats += params_regularization(param_pos_prev, param_possub_prev, param_wif_prev, param_wit_prev)

            losses_arcs.append(dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, train_chunk_deps[idx])))

            if (i % batch_dep == 0 or i == len(train_sents) - 1) and i != 0:
            # if (i % batch_size == 0 or i == len(train_sents) - 1) and i != 0:

                # losses_arcs.extend(losses_bunsetsu)

                if char_base:
                    sum_losses_arcs = dy.esum(losses_arcs) + loss_feats_batch
                else:
                    sum_losses_arcs = dy.esum(losses_arcs)
                sum_losses_arcs_value = sum_losses_arcs.value()

                sum_losses_arcs.backward()
                update_parameters()
                # global global_step
                # global_step += 1
                losses_arcs = []
                tot_loss_in_iter += sum_losses_arcs_value

            if i % batch_dep == 0 and i % batch_pos == 0 and i > 0:
            # if i % batch_size == 0:
                losses_chunk = []
                losses_arcs = []
                losses_feats = []

                dy.renew_cg()


            if i % show_loss_every == 0 and i != 0:
                if show_acc:
                    print("dep accuracy: ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)
                    print("dep accuracy not argmax: ", num_tot_cor_bunsetsu_dep_not_argmax / num_tot_bunsetsu_dep)
        if show_time:
            print("time in this iter: ", time.time() - prev)
        global global_step
        global_step += 1
        prev = time.time()
        print(it, "\t[train] average loss:\t", tot_loss_in_iter / len(train_sents))
        train_loss_in_iter = tot_loss_in_iter / len(train_sents)
        train_loss.extend([train_loss_in_iter])


def chunk_masks(pred_chunks, gold_chunks, gold_deps):
    ret = [1] * (len(gold_deps) + 1)
    matched_chunks = []

    idx = 0
    matched = True

    while idx < len(gold_chunks):
        if gold_chunks[idx] == 0 and idx > 0:
            if matched and gold_chunks[idx] == pred_chunks[idx]:
                matched_chunks.append(1)
            else:
                matched_chunks.append(0)
            matched = True
        if gold_chunks[idx] != pred_chunks[idx]:
            matched = False
        idx += 1

    for idx, mc in enumerate(matched_chunks):
        if mc == 0:
            ret[gold_deps[idx]] = 0

    ret = [r * mc for r, mc in zip(ret, matched_chunks)]

    return ret, np.sum(matched_chunks)


def dev(char_seqs,
          pos_seqs,
          pos_sub_seqs,
          wif_seqs,
          wit_seqs,
          word_bi_seqs,
          chunk_bi_seqs):
    tot_chunk = 0

    tot_cor_chunk = 0

    num_tot_bunsetsu_dep = 0
    num_tot_cor_bunsetsu_dep = 0
    num_tot_cor_bunsetsu_dep_not_argmax = 0
    num_tot_cor_bunsetsu_dep_chunk = 0
    num_tot_cor_bunsetsu_dep_not_argmax_chunk = 0

    complete_chunking = 0
    failed_chunking = 0
    chunks_excluded = 0

    prev = time.time()

    total_loss = 0

    for i in range(len(char_seqs) // divide_dev):
        dy.renew_cg()
        if len(char_seqs[i]) == 0:
            continue
        idx = i

        bunsetsu_ranges = ranges([0] + dev_sents[idx].chunk_bi_char + [0])

        chunk_bi_char = dev_sents[idx].chunk_bi_char

        cembs = char_embds(char_seqs[idx], pdrop_cemb)

        cembs_lstmouts, l2rs_char, r2ls_char = inputs2lstmouts(l2rlstm_char, r2llstm_char, cembs, pdrop_lstm)
        loss_feats, embs_pos, embs_possub, embs_wif, embs_wit = \
            pred_feats(cembs_lstmouts, [pos_seqs[idx], pos_sub_seqs[idx], wif_seqs[idx], wit_seqs[idx]])

        if not config.share_cembs:
            cembs = cembs_lstmouts

        embs_feats = [dy.concatenate([cemb, pos, possub, wif, wit])
                      for cemb, pos, possub, wif, wit in zip(cembs, embs_pos, embs_possub, embs_wif, embs_wit)]

        cembs_l2r = [dy.concatenate([l2r_char, emb_feats])
                     for l2r_char, emb_feats in zip(l2rs_char, embs_feats)]
        cembs_r2l = [dy.concatenate([r2l_char, emb_feats])
                     for r2l_char, emb_feats in zip(r2ls_char, embs_feats)]

        l2rs_char, r2ls_char = inputs2singlelstmouts(l2rlstm_word, cembs_l2r, pdrop_lstm), \
                               inputs2singlelstmouts(r2llstm_word, cembs_r2l, pdrop_lstm)

        cembs_bidir = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(l2rs_char, r2ls_char)]
        loss_chunk, preds_chunk, _ = bi_bunsetsu(cembs_bidir, chunk_bi_char)

        l2rs_char, r2ls_char = \
            [dy.concatenate([l2r_char, emb_feats]) for l2r_char, emb_feats in zip(l2rs_char, embs_feats)], \
            [dy.concatenate([r2l_char, emb_feats]) for r2l_char, emb_feats in zip(r2ls_char, embs_feats)]

        # cembs, l2rs_char, r2ls_char = inputs2lstmouts(l2rlstm_word, r2llstm_word, cembs, pdrop_lstm)
        bembs, l2rs_bunsetsu, r2ls_bunsetsu = segment_embds(l2rs_char, r2ls_char, bunsetsu_ranges, offset=1)
        bembs_l2r, bembs_r2l = inputs2singlelstmouts(l2rlstm_bunsetsu, l2rs_bunsetsu, pdrop_lstm), \
                               inputs2singlelstmouts(r2llstm_bunsetsu, r2ls_bunsetsu, pdrop_lstm)

        bembs = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(bembs_l2r, bembs_r2l)]
        tot_chunk += len(bembs)

        masks_chunk, cor_chunk = chunk_masks(preds_chunk, chunk_bi_char, dev_chunk_deps[i])
        tot_cor_chunk += cor_chunk

        arc_loss, arc_preds, arc_preds_not_argmax = dep_bunsetsu(bembs, pdrop)

        total_loss += dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, dev_chunk_deps[i])).value()

        num_tot_bunsetsu_dep += len(bembs) - config.root - 1

        if len(arc_preds) != len(dev_chunk_deps[i]):
            failed_chunking += 1
            continue

        num_tot_cor_bunsetsu_dep_chunk += np.sum([r * d for r, d in zip(masks_chunk, np.equal(arc_preds[:-1], dev_chunk_deps[i][:-1]))])
        num_tot_cor_bunsetsu_dep_not_argmax_chunk += np.sum([r * d for r, d in zip(masks_chunk, np.equal(arc_preds_not_argmax[1:-1], dev_chunk_deps[i][:-1]))])
        num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds[:-1], dev_chunk_deps[i][:-1]))
        num_tot_cor_bunsetsu_dep_not_argmax += np.sum(
            np.equal(arc_preds_not_argmax[1:], dev_chunk_deps[i]))

    global best_acc
    global update
    global early_stop_count
    global least_loss

    if best_acc < num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep:
        best_acc = num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep
        update = True
        early_stop_count = 0

    if least_loss > (total_loss / len(dev_sents)):
        least_loss = total_loss / len(dev_sents)


    print(i, " accuracy dep ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep, end='\t')
    print(i, " accuracy dep chunk", num_tot_cor_bunsetsu_dep_chunk / num_tot_bunsetsu_dep, end='\t')
    print(i, " accuracy chunk ", tot_cor_chunk / num_tot_bunsetsu_dep, end='\n')
    # acc.append(num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)
    acc.extend([num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep] * train_iter)
    if output_result:
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write(str(i) + " accuracy dep " + str(num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep) + '\n')
            f.write("total arc loss: " + str(total_loss / len(dev_sents)) + '\n')
            if chunker:
                f.write(str(i) + " accuracy chunking " + str(tot_cor_chunk / tot_chunk) + '\n')
                # f.write("complete chunking rate: " + str(complete_chunking / len(char_seqs))+ '\n')
                # f.write("failed_chunking rate: " + str(failed_chunking / len(char_seqs))+ '\n')
                # f.write("complete chunking: " + str(complete_chunking)+ '\n')
                # f.write("failed_chunking: " + str(failed_chunking)+ '\n')

    print("[dev]average loss: " + str(total_loss / len(dev_sents)))
    # dev_loss.append(total_loss / len(dev_sents))
    dev_loss.extend([total_loss / len(dev_sents)] * train_iter)
    if chunker:
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write(str(i) + " accuracy chunking " + str(tot_cor_chunk / tot_chunk) + '\n')    #     print("failed_chunking rate: " + str(failed_chunking / len(char_seqs)))
    #     print("complete chunking: " + str(complete_chunking))
    #     print("failed_chunking: " + str(failed_chunking))
    #     print("chunks_excluded: ", chunks_excluded)
    return


prev = time.time()

if LOAD and not TEST:
    pc.populate(load_file)
    print("loaded from: ", load_file)

prev_epoc = 0

detail_file = "detail.txt"
result_file = "result.txt"

for e in range(epoc):

    if LOAD:
        pc.populate(load_file + str(param_id))
        print("loaded from: ", load_file + str(param_id))
        param_id += 1

    if e * train_iter >= change_train_iter and train_iter != 1:
        train_iter = 1
        config.divide_train = 1

    print("epoc: ", prev_epoc)
    prev_epoc += train_iter

    TRAIN = True
    global pdrop
    global pdrop_lstm
    pdrop = pdrop_stash
    pdrop_lstm = pdrop_lstm_stash

    if not TEST:
        train(train_char_seqs, train_pos_seqs, train_pos_sub_seqs, train_wif_seqs, train_wit_seqs, train_word_bi_seqs, train_chunk_bi_seqs)
    # plot_loss(plt, train_loss, prev_epoc, "train_loss.png")
        if train_loss_ylim <= train_loss[-1]:
            train_loss_ylim = train_loss[-1] + 0.1

        plot_loss(plt, train_loss, prev_epoc, 1, train_loss_xlim, train_loss_ylim, 0)

    print("time: ", time.time() - prev)
    prev = time.time()

    pdrop = 0.0
    pdrop_lstm = 0.0
    TRAIN = False
    update = False

    if e == 0 and not TEST:
        fidx = 0
        save_file_valify = save_file

        while os.path.exists(save_file_valify):
            save_file_valify = save_file + str(fidx)
            fidx += 1

        save_file = save_file_valify

        directory = save_file_directory + save_file + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        detail_file = directory + "detail.txt"
        result_file = directory + "result.txt"
        save_file = directory + "parameter"

        python_codes = glob.glob('./*.py')
        for pycode in python_codes:
            shutil.copy2(pycode, directory)


    dev(dev_char_seqs, dev_pos_seqs, dev_pos_sub_seqs, dev_wif_seqs, dev_wit_seqs, dev_word_bi_seqs, dev_chunk_bi_seqs)

    if dev_loss_ylim <= dev_loss[-1]:
        dev_loss_ylim = dev_loss[-1] + 0.1
    if dev_loss_ylim_lower >= dev_loss[-1]:
        dev_loss_ylim_lower = dev_loss[-1] - 0.1
    if accuracy_ylim_lower >= acc[-1]:
        accuracy_ylim_lower = acc[-1] - 0.1


    plot_loss(plt, dev_loss, prev_epoc, 2, dev_loss_xlim, dev_loss_ylim, dev_loss_ylim_lower)
    plot_loss(plt, acc, prev_epoc, 3, accuracy_xlim, accuracy_ylim, accuracy_ylim_lower)


    if output_result:
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write("time: " + str(prev) + '\n')
            f.write("epoc: " + str(prev_epoc) + '\n')


    with open(detail_file, mode='w', encoding='utf-8') as f:
        f.write("train_loss" + '\t')
        for tl in train_loss:
            f.write(str(tl) + '\t')
        f.write("\n")

        f.write("dev_loss" + '\t')
        for dl in dev_loss:
            f.write(str(dl) + '\t')
        f.write("\n")

        f.write("accuracy" + '\t')
        for a in acc:
            f.write(str(a) + '\t')


    # plt.subplot(224)
    # plt.text(0, 0, str(best_acc))
    # plt.text(0, 1, str(least_loss))
    if not TEST:
        plt.savefig(directory + "image.png")

    global early_stop_count
    if not update:
        early_stop_count += train_iter

    print("time: ", time.time() - prev)
    prev = time.time()

    if SAVE and not TEST:
        pc.save(save_file + str(early_stop_count))
        print("saved into: ", save_file + str(early_stop_count))

    if early_stop_count > early_stop:
        print("best_acc: ", best_acc, end='\t')
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write("best_acc: " + str(best_acc))

        break
    else:
        print("best_acc: ", best_acc, end='\t')
        print("least_loss: ", least_loss, end='\t')
        print("lr: ", trainer.learning_rate, end='\t')
        print("early_stop_count: ", early_stop_count)
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write("best_acc: " + str(best_acc) + '\n')
            f.write("early_stop_count: " + str(early_stop_count) + '\n')



