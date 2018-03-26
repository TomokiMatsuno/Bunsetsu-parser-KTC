from collections import defaultdict

import numpy as np
import dynet as dy
import matplotlib.pyplot as plt
import os
import shutil
from get_pret_embs import get_pret_embs

from config import *
import config
from utils import *

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


#files = [path2KTC + 'syn/9501ED.KNP', path2KTC + 'syn/9501ED.KNP']



if STANDARD_SPLIT:
    files = glob.glob(path2KTC + 'syn/95010[1-9].*')
    files.extend(glob.glob(path2KTC + 'syn/95011[0-1].*'))
    files.extend(glob.glob(path2KTC + 'syn/950[1-8]ED.*'))
    if TEST:
        files.extend(glob.glob(path2KTC + 'syn/95011[4-7].*'))
        files.extend(glob.glob(path2KTC + 'syn/951[0-2]ED.*'))
        train_dev_boundary = -7
    else:
        files.extend(glob.glob(path2KTC + 'syn/95011[2-3].*'))
        files.extend(glob.glob(path2KTC + 'syn/9509ED.*'))
        train_dev_boundary = -3

if JOS:
    files = glob.glob(path2KTC + 'just-one-sentence.txt')
    files = [path2KTC + 'just-one-sentence.txt', path2KTC + 'just-one-sentence.txt']

if MINI_SET:
    files = [path2KTC + 'miniKTC_train.txt', path2KTC + 'miniKTC_dev.txt']

save_file = 'KTC'
split_name = ""

if CABOCHA_SPLIT:
    split_name = "_CABOCHA"
elif STANDARD_SPLIT:
    split_name = "_STANDARD"
elif MINI_SET:
    split_name = "_MINISET"

save_file = save_file_directory + save_file + split_name

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

if not use_annealing:
    trainer = dy.AdadeltaTrainer(pc)
else:
    if adam:
        trainer = dy.AdamTrainer(pc, config.learning_rate , config.beta_1, config.beta_2, config.epsilon)
    else:
        trainer = dy.SimpleSGDTrainer(pc, config.learning_rate)
        trainer.set_clip_threshold(1.)

trainer.set_sparse_updates(config.sparse_update)
global_step = 0

if adam:
    def update_parameters():
         if use_annealing:
             trainer.learning_rate = config.learning_rate * decay ** (global_step / config.decay_steps)
         trainer.update()
else:
    def update_parameters():
        if use_annealing:
            trainer.learning_rate = config.learning_rate / (1 + decay * (global_step - 1))
        trainer.update()


if not orthonormal:
    l2rlstm_char = dy.VanillaLSTMBuilder(LAYERS_character, char_INPUT_DIM * 1, INPUT_DIM // 2, pc, layer_norm)
    r2llstm_char = dy.VanillaLSTMBuilder(LAYERS_character, char_INPUT_DIM * 1, INPUT_DIM // 2, pc, layer_norm)

    l2rlstm_word = dy.VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 4, word_HIDDEN_DIM, pc, layer_norm)
    r2llstm_word = dy.VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 4, word_HIDDEN_DIM, pc, layer_norm)

    l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM, bunsetsu_HIDDEN_DIM, pc, layer_norm)
    r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM, bunsetsu_HIDDEN_DIM, pc, layer_norm)
else:
    l2rlstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, char_INPUT_DIM * 1, INPUT_DIM // 2, pc)
    r2llstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, char_INPUT_DIM * 1, INPUT_DIM // 2, pc)

    l2rlstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 4, word_HIDDEN_DIM, pc)
    r2llstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 4, word_HIDDEN_DIM, pc)

    # l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 4, bunsetsu_HIDDEN_DIM, pc)
    # r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 4, bunsetsu_HIDDEN_DIM, pc)

    if bembs_average_flag:
        l2rlstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, INPUT_DIM * ((2 * (use_cembs) + 2) + use_wif_wit * 2), bunsetsu_HIDDEN_DIM, pc)
        r2llstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, INPUT_DIM * ((2 * (use_cembs) + 2) + use_wif_wit * 2), bunsetsu_HIDDEN_DIM, pc)
    else:
        l2rlstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM, bunsetsu_HIDDEN_DIM, pc)
        r2llstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM, bunsetsu_HIDDEN_DIM, pc)

    if False and cont_aux_separated:
        l2rlstm_cont = orthonormal_VanillaLSTMBuilder(LAYERS_contfunc, word_HIDDEN_DIM * 2 // (2 - wemb_lstm), bunsetsu_HIDDEN_DIM, pc)
        r2llstm_cont = orthonormal_VanillaLSTMBuilder(LAYERS_contfunc, word_HIDDEN_DIM * 2 // (2 - wemb_lstm), bunsetsu_HIDDEN_DIM, pc)

        l2rlstm_func = orthonormal_VanillaLSTMBuilder(LAYERS_contfunc, word_HIDDEN_DIM * 2 // (2 - wemb_lstm), bunsetsu_HIDDEN_DIM, pc)
        r2llstm_func = orthonormal_VanillaLSTMBuilder(LAYERS_contfunc, word_HIDDEN_DIM * 2 // (2 - wemb_lstm), bunsetsu_HIDDEN_DIM, pc)


params = {}
params["lp_w"] = pc.add_lookup_parameters((WORDS_SIZE + 1, INPUT_DIM))
params["lp_c"] = pc.add_lookup_parameters((CHARS_SIZE + 1, char_INPUT_DIM))
params["lp_bp"] = pc.add_lookup_parameters((BIPOS_SIZE + 1, INPUT_DIM))
params["lp_p"] = pc.add_lookup_parameters((POS_SIZE + 1, (INPUT_DIM)))
params["lp_ps"] = pc.add_lookup_parameters((POSSUB_SIZE + 1, (INPUT_DIM)))
params["lp_wif"] = pc.add_lookup_parameters((WIF_SIZE + 1, (INPUT_DIM)))
params["lp_wit"] = pc.add_lookup_parameters((WIT_SIZE + 1, (INPUT_DIM)))
params["root_emb"] = pc.add_lookup_parameters((1, bunsetsu_HIDDEN_DIM * 2))

if orthonormal or not TEST:
    W = orthonormal_initializer(MLP_HIDDEN_DIM, 2 * bunsetsu_HIDDEN_DIM + config.word_order * word_order_DIM)
    params["head_MLP"] = pc.parameters_from_numpy(W)
    params["head_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

    params["dep_MLP"] = pc.parameters_from_numpy(W)
    params["dep_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

else:
    params["head_MLP"] = pc.add_parameters((MLP_HIDDEN_DIM, bunsetsu_HIDDEN_DIM * 2 + config.word_order * word_order_DIM))
    params["head_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

    params["dep_MLP"] = pc.add_parameters((MLP_HIDDEN_DIM, bunsetsu_HIDDEN_DIM * 2 + config.word_order * word_order_DIM))
    params["dep_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

# params["R_bunsetsu_biaffine"] = pc.add_parameters((MLP_HIDDEN_DIM + biaffine_bias_y, MLP_HIDDEN_DIM + biaffine_bias_x))
params["R_bunsetsu_biaffine"] = pc.add_parameters((MLP_HIDDEN_DIM))


def inputs2lstmouts(l2rlstm, r2llstm, inputs, pdrop):
    l2rlstm.set_dropouts(pdrop, pdrop * config.adam)
    r2llstm.set_dropouts(pdrop, pdrop * config.adam)

    s_l2r_0 = l2rlstm.initial_state()
    s_r2l_0 = r2llstm.initial_state()

    s_l2r = s_l2r_0
    s_r2l = s_r2l_0

    # l2r_outs = s_l2r.add_inputs(inputs)
    # r2l_outs = s_r2l.add_inputs(reversed(inputs))

    l2r_outs, r2l_outs = s_l2r.transduce(inputs), s_r2l.transduce(reversed(inputs))
    r2l_outs = [l for l in reversed(r2l_outs)]
    lstm_outs = [dy.concatenate([f, b]) for f, b in zip(l2r_outs, r2l_outs)]

    # lstm_outs = [dy.concatenate([l2r_outs[i].output(), r2l_outs[i].output()]) for i in range(len(l2r_outs))]
    # lstm_outs = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(l2r_outs, reversed(r2l_outs))]
    # l2r_outs = [l2r_outs[i].output() for i in range(len(l2r_outs))]
    # r2l_outs = [r2l_outs[i].output() for i in range(len(r2l_outs))]

    return lstm_outs, l2r_outs, r2l_outs


def inputs2singlelstmouts(lstm, inputs, pdrop, reverse=False):

    lstm.set_dropouts(pdrop, pdrop * config.adam)

    s_0 = lstm.initial_state()

    s = s_0

    if reverse:
        inputs = [i for i in reversed(inputs)]

    lstm_outs = s.transduce(inputs)

    if reverse:
        lstm_outs = [l for l in reversed(lstm_outs)]

    # lstm_outs = [outs[i].output() for i in range(len(outs))]

    return lstm_outs



def bi_bunsetsu(wembs, chunk_bi):
    num_cor = 0

    R_bi_b = dy.parameter(params["R_bi_b"])
    bias_bi_b = dy.parameter(params["bias_bi_b"])
    loss = []
    preds = []

    for i in range(len(chunk_bi) - config.BOS_EOS * 2):
        probs = dy.softmax(R_bi_b * wembs[i] + bias_bi_b)
        loss.append(-dy.log(dy.pick(probs, chunk_bi[i])))

        if show_acc or scheduled_learning or not TRAIN:
            chosen = np.asscalar(np.argmax(probs.npvalue()))
            preds.append(chosen)
            if(chosen == chunk_bi[i]):
                # print(chosen, " ", bi_b_seq[w_1st_chars[i]])
                num_cor += 1
    loss = dy.esum(loss)
    return loss, preds, num_cor


def dep_bunsetsu(bembs, pdrop, golds):
    if not config.root:
        root_emb = params["root_emb"]

    dep_MLP = dy.parameter(params["dep_MLP"])
    dep_MLP_bias = dy.parameter(params["dep_MLP_bias"])
    head_MLP = dy.parameter(params["head_MLP"])
    head_MLP_bias = dy.parameter(params["head_MLP_bias"])

    R_bunsetsu_biaffine = dy.parameter(params["R_bunsetsu_biaffine"])

    input_size = bembs[0].dim()[0][0]

    if not config.root:
        bembs = [root_emb[0]] + bembs

    slen_x = slen_y = len(bembs)

    bembs_dep = bembs
    bembs_head = bembs

    bembs_dep = dy.dropout(
        dy.concatenate(bembs_dep, 1), pdrop)
    bembs_head = dy.dropout(
        dy.concatenate(bembs_head, 1), pdrop)

    bembs_dep = dy.dropout(leaky_relu(
        dy.affine_transform([dep_MLP_bias, dep_MLP, bembs_dep])), pdrop)
    bembs_head = dy.dropout(leaky_relu(
        dy.affine_transform([head_MLP_bias, head_MLP, bembs_head])), pdrop)
    # bembs_dep = dy.dropout((
    #     dy.affine_transform([dep_MLP_bias, dep_MLP, bembs_dep])), pdrop)
    # bembs_head = dy.dropout((
    #     dy.affine_transform([head_MLP_bias, head_MLP, bembs_head])), pdrop)

    # blin = bilinear(bembs_dep, R_bunsetsu_biaffine, bembs_head, input_size, slen_x, slen_y, 1, 1, biaffine_bias_x, biaffine_bias_y)
    W = dy.concatenate_cols([R_bunsetsu_biaffine] * slen_x)
    blin = (dy.transpose(dy.cmult(bembs_dep, W)) * bembs_head)
    # blin += dist_matrix(slen_x)

    arc_preds = []
    arc_preds_not_argmax = []
    loss = dy.scalarInput(0)

    if not TRAIN:
    # arc_preds_not_argmax = blin.npvalue().argmax(0)
        msk = [1] * slen_x
        arc_probs = dy.softmax(blin).npvalue()
        arc_probs = np.transpose(arc_probs)
        arc_preds = arc_argmax(arc_probs, slen_x, msk)
        arc_preds = arc_preds[1:]

    arc_loss = dy.reshape(blin, (slen_y,), slen_x) + loss
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

    # for r in ranges[st:en]:
    for r in ranges[st:en]:
        start = r[0] - offset
        end = r[1] - offset

        if segment_concat:
            l2r = l2r_outs[end - 1] + l2r_outs[start]
            r2l = r2l_outs[start] + r2l_outs[end - 1]
            # l2r = l2r_outs[end - 1]
            # r2l = r2l_outs[start]
        else:
            l2r = l2r_outs[end] - l2r_outs[start]
            r2l = r2l_outs[start + 1] - r2l_outs[end + 1]

        ret.append(dy.concatenate([l2r, r2l]))

    return ret


def char_embds(char_seq):
    lp_c = params["lp_c"]
    chars = [cd.i2x[char_seq[i]] for i in range(len(char_seq))]
    pret_embs_chars = [dy.inputTensor(pret_embs[c]) if c in pret_embs else dy.inputTensor(np.zeros(char_INPUT_DIM, dtype=np.float32)) for c in chars]

    cembs = [dy.dropout(lp_c[char_seq[i]] + pret_embs_chars[i], pdrop_embs) for i in range(len(char_seq))]

    return cembs


def embd_mask_generator(pdrop, slen):
    if not TRAIN:
        pdrop = 0.0

    masks_w = np.random.binomial(1, 1 - pdrop, slen)
    masks_t = np.random.binomial(1, 1 - pdrop, slen)
    # scales = [3. / (2. * mask_w + mask_t + 1e-12) for mask_w, mask_t in zip(masks_w, masks_t)]
    scales = [2. / (mask_w + mask_t + 1e-12) for mask_w, mask_t in zip(masks_w, masks_t)]
    masks_w = [mask_w * scale for mask_w, scale in zip(masks_w, scales)]
    masks_t = [mask_t * scale for mask_t, scale in zip(masks_t, scales)]
    return masks_w, masks_t


if config.add_pret_embs:
    pret_embs = get_pret_embs()
    pret_vocab = set(pret_embs.index2word)


params["lin_lpw"] = pc.add_parameters((INPUT_DIM, config.char_INPUT_DIM))
params["lin_lpw_bias"] = pc.add_parameters((INPUT_DIM), init=dy.ConstInitializer(0.))

def word_embds(char_seq, pos_seq, pos_sub_seq, wif_seq, wit_seq, word_ranges, cembs):
    lp_w = params["lp_w"]
    lp_p = params["lp_p"]
    lp_ps = params["lp_ps"]
    lp_wif = params["lp_wif"]
    lp_wit = params["lp_wit"]

    word_lin = dy.parameter(params["lin_lpw"])
    word_lin_bias = dy.parameter(params["lin_lpw_bias"])

    wembs_w = []
    wembs_t = []
    masks_w, masks_t = embd_mask_generator(pdrop_embs, len(pos_seq))

    for idx, wr in enumerate(word_ranges):
        str = ""

        for c in char_seq[wr[0]: wr[1]]:
            str += cd.i2x[c]

        pos_lp = dy.concatenate([lp_p[pos_seq[idx]] + lp_ps[pos_sub_seq[idx]], lp_wif[wif_seq[idx]] + lp_wit[wit_seq[idx]]])

        if config.add_pret_embs and str in pret_embs:
            pret_emb = dy.inputTensor(pret_embs[str])
        else:
            pret_emb = dy.inputTensor(np.zeros(char_INPUT_DIM, dtype=np.float32))

        if config.w2v200:
            pret_emb = dy.affine_transform([word_lin_bias, word_lin, pret_emb])

        if str in wd.x2i:
            word_form = dy.concatenate([lp_w[wd.x2i[str]] + pret_emb, cembs[idx]])
        else:
            word_form = dy.concatenate([lp_w[wd.x2i["UNK"]] + pret_emb, cembs[idx]])

        # word_form = dy.cube(word_form)
        # pos_lp = dy.cube(pos_lp)

        wembs_w.append(word_form)
        wembs_t.append(pos_lp)

    return [dy.concatenate([wemb_w * mask_w, wemb_t * mask_t])
            for wemb_w, wemb_t, mask_w, mask_t in zip(wembs_w, wembs_t, masks_w, masks_t)]


params['lp_distance'] = pc.add_lookup_parameters((4, INPUT_DIM))
params['W_dist'] = pc.add_parameters((word_order_DIM, INPUT_DIM))
params['bias_dist'] = pc.add_parameters((word_order_DIM))

def dist_matrix(dim):
    lp_distance = params['lp_distance']

    bias_dist = dy.parameter(params['bias_dist'])
    W_dist = dy.parameter(params['W_dist'])
    W = []

    for di in range(dim):
        # W.append([])
        tmp = []
        for dj in range(dim):
            if di == dj:
                dist_type = 0
            elif abs(di - dj) == 1:
                dist_type = 1
            elif abs(di - dj) >= 6:
                dist_type = 3
            else:
                dist_type = 2
            dist = leaky_relu(dy.affine_transform([bias_dist, W_dist, lp_distance[dist_type]]))
            tmp.append(dy.sum_elems(dist))

        W.append(dy.concatenate(tmp))

    W = dy.concatenate_cols(W)
    # W = dy.transpose(W)
    # for di in range(dim):
    #     W.append([])
    #     for dj in range(dim):
    #         if di == dj:
    #             W[-1].append(0)
    #         elif abs(di - dj) == 1:
    #             W[-1].append(1)
    #         elif abs(di - dj) >= 6:
    #             W[-1].append(3)
    #         else:
    #             W[-1].append(2)
    # W = dy.lookup_batch(lp_distance, W)

    return W


def bunsetsu_embds(l2r_outs, r2l_outs, bunsetsu_ranges):
    ret = []

    bembs_l2r = []
    bembs_r2l = []

    for br in bunsetsu_ranges:
        start = br[0]
        end = br[1]

        ret.append(dy.concatenate([l2r_outs[end] - l2r_outs[start],
                                   r2l_outs[start + 1] - r2l_outs[end + 1],
                                   ]))
        bembs_l2r.append(l2r_outs[end] - l2r_outs[start])
        bembs_r2l.append(r2l_outs[start + 1] - r2l_outs[end + 1])

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
    for r in ranges[1:-1]:
        tmp = []
        for widx in range(r[0], r[1]):
            tmp.append(wembs[widx])
        ret.append(dy.average(tmp))
    return ret


cor_parsed_count = defaultdict(int)
done_sents = set()

def train(char_seqs,
          pos_seqs,
          pos_sub_seqs,
          wif_seqs,
          wit_seqs,
          word_bi_seqs,
          chunk_bi_seqs,
              ):
    losses_bunsetsu = []
    losses_arcs = []
    prev = time.time()

    for it in range(train_iter):
        # print("iteration: ", it)
        num_tot_bunsetsu_dep = 0
        tot_loss_in_iter = 0
        sent_ids = [sent_id for sent_id in range(len(char_seqs))]
        np.random.shuffle(sent_ids)

        # for i in (range((len(char_seqs) // batch_size) * batch_size)):
        for i in range(len(char_seqs) // divide_train):
            if i % batch_size == 0:
                losses_bunsetsu = []
                losses_arcs = []

                dy.renew_cg()

            if random_pickup:
                idx = i if not TRAIN else sent_ids[i]
            else:
                idx = i

            if len(char_seqs[idx]) == 0 or len(chunk_bi_seqs[idx]) == 0:
                continue

            bi_w_seq = chunk_bi_seqs[idx]
            word_ranges = ranges(word_bi_seqs[idx])

            cembs = char_embds(char_seqs[idx])
            cembs, l2r_char, r2l_char = inputs2lstmouts(l2rlstm_char, r2llstm_char, cembs, pdrop)
            cembs = segment_embds(l2r_char, r2l_char, word_ranges, offset=0, segment_concat=True)

            wembs = word_embds(char_seqs[idx], pos_seqs[idx],
                               pos_sub_seqs[idx], wif_seqs[idx], wit_seqs[idx], word_ranges, cembs)

            wembs, l2r_outs, r2l_outs = inputs2lstmouts(l2rlstm_word, r2llstm_word, wembs, pdrop)

            if chunker:
                loss_bi_bunsetsu, bi_bunsetsu_preds, _ = bi_bunsetsu(wembs, bi_w_seq)
                losses_bunsetsu.append(loss_bi_bunsetsu)

            bunsetsu_ranges = ranges(bi_w_seq)
            aux_positions = aux_position(bunsetsu_ranges, pos_seqs[idx], pos_sub_seqs[idx])


            bembs, bembs_l2r, bembs_r2l = bunsetsu_embds(l2r_outs, r2l_outs, bunsetsu_ranges, aux_positions, pdrop_lstm)

            bembs_l2r = inputs2singlelstmouts(l2rlstm_bunsetsu, bembs_l2r, pdrop)
            bembs_r2l = inputs2singlelstmouts(r2llstm_bunsetsu, bembs_r2l, pdrop, reverse=True)
            bembs = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(bembs_l2r, bembs_r2l)]

            arc_loss, arc_preds, arc_preds_not_argmax = dep_bunsetsu(bembs, pdrop_lstm, train_chunk_deps[idx])

            num_tot_bunsetsu_dep += len(bembs) - config.root - 1

            losses_arcs.append(dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, train_chunk_deps[idx])))

            global global_step
            if (i % batch_size == 0 or i == len(train_sents) - 1) and i != 0:

                losses_arcs.extend(losses_bunsetsu)

                sum_losses_arcs = dy.esum(losses_arcs)
                sum_losses_arcs_value = sum_losses_arcs.value()

                sum_losses_arcs.backward()
                update_parameters()

                tot_loss_in_iter += sum_losses_arcs_value

                if adam:
                    global_step += 1

        if show_time:
            print("time in this iter: ", time.time() - prev)
        prev = time.time()
        print(it, "\t[train] average loss:\t", tot_loss_in_iter / len(train_sents))
        train_loss_in_iter = tot_loss_in_iter / len(train_sents)
        train_loss.extend([train_loss_in_iter])
        if not adam:
            global_step += 1

def param_regularizer():
    # dep_MLP = params['dep_MLP']
    # head_MLP = params['head_MLP']
    # R_bunsetsu_biaffine = params['R_bunsetsu_biaffine']
    dep_MLP = dy.parameter(params['dep_MLP'])
    head_MLP = dy.parameter(params['head_MLP'])
    R_bunsetsu_biaffine = dy.parameter(params['R_bunsetsu_biaffine'])

    loss = dy.scalarInput(0)

    for expr in l2rlstm_char.get_parameter_expressions():
        for ex in expr:
            loss += dy.l2_norm(ex)
    for expr in r2llstm_char.get_parameter_expressions():
        for ex in expr:
            loss += dy.l2_norm(ex)
    for expr in l2rlstm_word.get_parameter_expressions():
        for ex in expr:
            loss += dy.l2_norm(ex)
    for expr in r2llstm_word.get_parameter_expressions():
        for ex in expr:
            loss += dy.l2_norm(ex)
    for expr in l2rlstm_bunsetsu.get_parameter_expressions():
        for ex in expr:
            loss += dy.l2_norm(ex)
    for expr in r2llstm_bunsetsu.get_parameter_expressions():
        for ex in expr:
            loss += dy.l2_norm(ex)

    loss *= config.L2_coef_lstm

    loss += config.L2_coef_classifier * (dy.l2_norm(dep_MLP) + dy.l2_norm(head_MLP) +
                                         dy.l2_norm(R_bunsetsu_biaffine))

    return loss





def train_dev(char_seqs,
          pos_seqs,
          pos_sub_seqs,
          wif_seqs,
          wit_seqs,
          word_bi_seqs,
          chunk_bi_seqs,
              chunk_deps):
    num_tot = 0

    num_tot_cor_bunsetsu_dep = 0


    prev = time.time()

    total_loss = 0
    num_tot_bunsetsu_dep = 0
    tot_loss_in_iter = 0
    sent_ids = [sent_id for sent_id in range(len(char_seqs))]
    np.random.shuffle(sent_ids)

    for i in range(len(char_seqs) // divide_dev):
        if i % batch_size == 0 or not TRAIN:
            losses_arcs = []

            dy.renew_cg()

        if random_pickup:
            idx = i if not TRAIN else sent_ids[i]
        else:
            idx = i

        if len(char_seqs[i]) == 0:
            continue

        bi_w_seq = chunk_bi_seqs[idx]
        num_tot += len(char_seqs[idx])

        word_ranges = ranges(word_bi_seqs[idx])

        cembs = char_embds(char_seqs[idx])
        cembs, l2r_char, r2l_char = inputs2lstmouts(l2rlstm_char, r2llstm_char, cembs, pdrop)
        cembs = segment_embds(l2r_char, r2l_char, word_ranges, offset=0, segment_concat=True)

        wembs = word_embds(char_seqs[idx], pos_seqs[idx],
                           pos_sub_seqs[idx], wif_seqs[idx], wit_seqs[idx], word_ranges, cembs)

        # wembs = [leaky_relu(wemb) for wemb in wembs]

        wembs, l2r_outs, r2l_outs = inputs2lstmouts(l2rlstm_word, r2llstm_word, wembs, pdrop)

        gold_bunsetsu_ranges = ranges(bi_w_seq)

        bembs, bembs_l2r, bembs_r2l = bunsetsu_embds(l2r_outs, r2l_outs, gold_bunsetsu_ranges)

        # bembs_l2r = [leaky_relu(bemb) for bemb in bembs_l2r]
        # bembs_r2l = [leaky_relu(bemb) for bemb in bembs_r2l]

        bembs_l2r = inputs2singlelstmouts(l2rlstm_bunsetsu, bembs_l2r, pdrop)
        bembs_r2l = inputs2singlelstmouts(r2llstm_bunsetsu, bembs_r2l, pdrop, reverse=True)
        bembs = [dy.concatenate([l2r, r2l]) for l2r, r2l in zip(bembs_l2r, bembs_r2l)]

        arc_loss, arc_preds, arc_preds_not_argmax = dep_bunsetsu(bembs, pdrop, chunk_deps[idx])
        arc_loss = dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, chunk_deps[idx]))
        losses_arcs.append(arc_loss)

        if not TRAIN:
            total_loss += arc_loss.value()
            num_tot_bunsetsu_dep += len(bembs) - config.root - 1
            num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds[:-1], chunk_deps[idx][:-1]))

        if TRAIN:
            global global_step
            if (i % batch_size == 0 or i == len(train_sents) - 1) and i != 0:
                sum_losses_arcs = dy.esum(losses_arcs)
                sum_losses_arcs += param_regularizer()
                sum_losses_arcs_value = sum_losses_arcs.value()

                sum_losses_arcs.backward()
                update_parameters()

                tot_loss_in_iter += sum_losses_arcs_value

                if adam:
                    global_step += 1

    if TRAIN:
        prev = time.time()
        print("[train] average loss:\t", tot_loss_in_iter / len(train_sents))
        train_loss_in_iter = tot_loss_in_iter / len(train_sents)
        train_loss.extend([train_loss_in_iter])
        if not adam:
            global_step += 1
    else:
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
        acc.extend([num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep] * train_iter)
        if output_result:
            with open(result_file, mode='a', encoding='utf-8') as f:
                f.write(str(i) + " accuracy dep " + str(num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep) + '\n')
                f.write("total arc loss: " + str(total_loss / len(char_seqs)) + '\n')

        print("[dev]average loss: " + str(total_loss / len(char_seqs)))
        dev_loss.extend([total_loss / len(char_seqs)] * train_iter)
    return


prev = time.time()


prev_epoc = 0

detail_file = "detail.txt"
result_file = "result.txt"
save_file = "parameter"

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
    pdrop_embs = pdrop_embs_stash

    if not TEST:
        for iter in range(train_iter):
            train_dev(train_char_seqs, train_pos_seqs, train_pos_sub_seqs, train_wif_seqs, train_wit_seqs,
                      train_word_bi_seqs, train_chunk_bi_seqs, train_chunk_deps)
            # if train_loss_ylim <= train_loss[-1]:
            #     train_loss_ylim = train_loss[-1] + 0.1
            #
            # plot_loss(plt, train_loss, prev_epoc, 1, train_loss_xlim, train_loss_ylim, 0)

    print("time: ", time.time() - prev)
    prev = time.time()

    pdrop = 0.0
    pdrop_lstm = 0.0
    pdrop_embs = 0.0
    TRAIN = False
    update = False

    train_dev(dev_char_seqs, dev_pos_seqs, dev_pos_sub_seqs, dev_wif_seqs, dev_wit_seqs,
              dev_word_bi_seqs, dev_chunk_bi_seqs, dev_chunk_deps)

    if not TEST:
        if dev_loss_ylim <= dev_loss[-1]:
            dev_loss_ylim = dev_loss[-1] + 0.1
        if dev_loss_ylim_lower >= dev_loss[-1]:
            dev_loss_ylim_lower = dev_loss[-1] - 0.1
        if accuracy_ylim_lower >= acc[-1]:
            accuracy_ylim_lower = acc[-1] - 0.1


        plot_loss(plt, dev_loss, prev_epoc, 2, dev_loss_xlim, dev_loss_ylim, dev_loss_ylim_lower)
        plot_loss(plt, acc, prev_epoc, 3, accuracy_xlim, accuracy_ylim, accuracy_ylim_lower)

    if e == 0 and not TEST:
        fidx = 0
        save_file_valify = save_file

        while os.path.exists(save_file_valify):
            save_file_valify = save_file + str(fidx)
            fidx += 1

        save_file = save_file_valify

        directory = save_file + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        detail_file = directory + "detail.txt"
        result_file = directory + "result.txt"
        save_file = directory + "parameter"

        python_codes = glob.glob('./*.py')
        for pycode in python_codes:
            shutil.copy2(pycode, directory)

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



