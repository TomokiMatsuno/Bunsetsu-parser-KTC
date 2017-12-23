from collections import defaultdict

import numpy as np
import dynet as dy

from config import *
import config
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
            '_word-HIDDEN-DIM' + str(word_HIDDEN_DIM) + \
            '_bunsetsu-HIDDEN-DIM' + str(bunsetsu_HIDDEN_DIM) + \
            '_MLP-HIDDEN-DIM' + str(MLP_HIDDEN_DIM) + \
            '_INPUT-DIM' + str(INPUT_DIM) + \
            '_batch-size' + str(batch_size) + \
            '_pdrop' + str(pdrop) + \
            '_pdrop_bunsetsu' + str(pdrop_bunsetsu)
            #'_learning-rate' + str(learning_rate) + \

split_name = ""

if CABOCHA_SPLIT:
    split_name = "_CABOCHA"
elif STANDARD_SPLIT:
    split_name = "_STANDARD"
elif MINI_SET:
    split_name = "_MINISET"

save_file = save_file + split_name

if scheduled_learning:
    save_file = save_file + "_scheduledLearning"

load_file = save_file

result_file = save_file + "_result.txt"


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


if not orthonormal:
    l2rlstm_char = dy.VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 1, INPUT_DIM, pc)
    r2llstm_char = dy.VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 1, INPUT_DIM, pc)

    l2rlstm_word = dy.VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 4, word_HIDDEN_DIM, pc)
    r2llstm_word = dy.VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 4, word_HIDDEN_DIM, pc)

    l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2 * (1 + cont_aux_separated), bunsetsu_HIDDEN_DIM, pc)
    r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * 2 * (1 + cont_aux_separated), bunsetsu_HIDDEN_DIM, pc)
else:
    l2rlstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 1, INPUT_DIM, pc)
    r2llstm_char = orthonormal_VanillaLSTMBuilder(LAYERS_character, INPUT_DIM * 1, INPUT_DIM, pc)

    l2rlstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 4, word_HIDDEN_DIM, pc)
    r2llstm_word = orthonormal_VanillaLSTMBuilder(LAYERS_word, INPUT_DIM * 4, word_HIDDEN_DIM, pc)

    l2rlstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * (1 + cont_aux_separated), bunsetsu_HIDDEN_DIM, pc)
    r2llstm_bunsetsu = orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, word_HIDDEN_DIM * (1 + cont_aux_separated), bunsetsu_HIDDEN_DIM, pc)


params = {}
params["lp_w"] = pc.add_lookup_parameters((WORDS_SIZE + 1, INPUT_DIM))
params["lp_c"] = pc.add_lookup_parameters((CHARS_SIZE + 1, INPUT_DIM))
params["lp_bp"] = pc.add_lookup_parameters((BIPOS_SIZE + 1, INPUT_DIM))
params["lp_p"] = pc.add_lookup_parameters((POS_SIZE + 1, INPUT_DIM))
params["lp_ps"] = pc.add_lookup_parameters((POSSUB_SIZE + 1, INPUT_DIM))
params["lp_wif"] = pc.add_lookup_parameters((WIF_SIZE + 1, INPUT_DIM))
params["lp_wit"] = pc.add_lookup_parameters((WIT_SIZE + 1, INPUT_DIM))

params["lp_func"] = pc.add_lookup_parameters((2, word_HIDDEN_DIM))

params["R_bi_b"] = pc.add_parameters((2, word_HIDDEN_DIM * 2))
params["bias_bi_b"] = pc.add_parameters((2))

params["head_MLP"] = pc.add_parameters((MLP_HIDDEN_DIM, bunsetsu_HIDDEN_DIM * 2))
params["head_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM))

params["dep_MLP"] = pc.add_parameters((MLP_HIDDEN_DIM, bunsetsu_HIDDEN_DIM * 2))
params["dep_MLP_bias"] = pc.add_parameters((MLP_HIDDEN_DIM))

params["R_bunsetsu_biaffine"] = pc.add_parameters((MLP_HIDDEN_DIM + biaffine_bias_y, MLP_HIDDEN_DIM + biaffine_bias_x))

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
    l2r_outs = [l2r_outs[i].output() for i in range(len(l2r_outs))]
    r2l_outs = [r2l_outs[i].output() for i in range(len(r2l_outs))]

    return lstm_outs, l2r_outs, r2l_outs


def bi_bunsetsu_wembs(wembs, bi_w_seq):
    num_cor = 0

    R_bi_b = dy.parameter(params["R_bi_b"])
    bias_bi_b = dy.parameter(params["bias_bi_b"])
    loss = []
    preds = []

    for i in range(len(bi_w_seq) - config.BOS_EOS * 2):
        probs = dy.softmax(R_bi_b * wembs[i] + bias_bi_b)
        loss.append(-dy.log(dy.pick(probs, bi_w_seq[i])))

        if show_acc or scheduled_learning or not TRAIN:
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
    slen_x = len(bembs)
    slen_y = slen_x
    bembs_dep = dy.dropout(dy.concatenate(bembs, 1), pdrop_bunsetsu)
    bembs_head = dy.dropout(dy.concatenate(bembs, 1), pdrop_bunsetsu)
    input_size = MLP_HIDDEN_DIM

    bembs_dep = leaky_relu(dep_MLP * bembs_dep + dep_MLP_bias)
    bembs_head = leaky_relu(head_MLP * bembs_head + head_MLP_bias)

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


def char_embds(char_seq, bipos_seq, word_ranges):
    lp_c = params["lp_c"]

    cembs = [lp_c[char_seq[i]] for i in range(len(char_seq))]

    bidir, l2r_outs, r2l_outs = inputs2lstmouts(l2rlstm_char, r2llstm_char, cembs, pdrop)

    ret = []

    for wr in word_ranges[1: -1]:
        start = wr[0]
        end = wr[1]
        if end < len(l2r_outs) or True:
            ret.append(dy.concatenate([l2r_outs[end] - l2r_outs[start + 1], r2l_outs[start] - r2l_outs[end - 1]]))
    if tanh:
        ret = [dy.tanh(r) for r in ret]
    return ret


def word_embds(char_seq, bipos_seq, pos_seq, pos_sub_seq, wif_seq, wit_seq, word_ranges):
    lp_c = params["lp_c"]
    lp_w = params["lp_w"]
    lp_p = params["lp_p"]

    cembs = char_embds(char_seq, bipos_seq, word_ranges)

    wembs = []

    # for idx, wr in enumerate(word_ranges[1:-1]):
    for idx, wr in enumerate(word_ranges[1: -1]):
        str = ""
        tmp_char = []
        # idx += 1

        for c in char_seq[wr[0]: wr[1]]:
            str += cd.i2x[c]
            tmp_char.append(dy.dropout(lp_c[c], pdrop))

        pos_lp = lp_p[pos_seq[wr[0]]]
        rnd_pos = np.random.randint(0, 3)
        rnd_word = np.random.randint(0, 3)

        if not TRAIN:
            rnd_word = rnd_pos = 1

        if str in wd.x2i:
            word_form = lp_w[wd.x2i[str]]
        else:
            word_form = pos_lp

        if rnd_pos == 0:
            pos_lp = pos_lp - pos_lp
        if rnd_word == 0:
            word_form = word_form - word_form

        if rnd_pos == 0:
            pos_lp = word_form
        if rnd_word == 0:
            word_form = pos_lp

        wembs.append(dy.concatenate([word_form, pos_lp, cembs[idx]]))

    return wembs


def bunsetsu_embds(l2r_outs, r2l_outs, bunsetsu_ranges, aux_position):

    ret = []

    lp_func = params["lp_func"]

    for br, aux_idx in zip(bunsetsu_ranges[2: -2], aux_position[2: -2]):
        start = br[0] - 1
        # start = br[0]
        end = br[1] - 1

        if not cont_aux_separated:
            ret.append(dy.concatenate([l2r_outs[end] - l2r_outs[start + 1],
                                       r2l_outs[start] - r2l_outs[end - 1]]))
            # if end + 1 < len(l2r_outs):
            #     ret.append(dy.concatenate([l2r_outs[end] - l2r_outs[start - 1], r2l_outs[start] - r2l_outs[end + 1]]))
            # elif end - start <= 1:
            #     ret.append(dy.concatenate([l2r_outs[-1], r2l_outs[-1]]))
            # else:
            #     ret.append(dy.concatenate([l2r_outs[-1] - l2r_outs[start - 1], r2l_outs[start] - r2l_outs[-1]]))
        elif aux_idx != -1:
            ret.append(dy.concatenate([l2r_outs[start + aux_idx] - l2r_outs[start + 1],
                                       r2l_outs[start] - r2l_outs[start + aux_idx - 1],
                                       l2r_outs[end] - l2r_outs[start + aux_idx + 1],
                                       r2l_outs[start + aux_idx] - r2l_outs[end - 1]]))
        else:
            ret.append(dy.concatenate([l2r_outs[end] - l2r_outs[start + 1],
                                       r2l_outs[start] - r2l_outs[end - 1],
                                       lp_func[0], lp_func[1]]))
                                       # l2r_outs[end] - l2r_outs[start],
                                       # r2l_outs[start] - r2l_outs[end]]))
            # elif end - start <= 1:
            #     ret.append(dy.concatenate([l2r_outs[-1],
            #                                r2l_outs[-1],
            #                                lp_func[0], lp_func[1]]))
            # else:
            #     ret.append(dy.concatenate([l2r_outs[-1] - l2r_outs[start + 1],
            #                                r2l_outs[start] - r2l_outs[-2],
            #                                lp_func[0], lp_func[1]]))
                                           # l2r_outs[-1] - l2r_outs[start],
                                           # r2l_outs[start] - r2l_outs[-1]]))
    if tanh:
        ret = [dy.tanh(r) for r in ret]

    return ret


def segment_embds(l2rlstm, r2llstm, inputs, ranges):
    ret = []

    for r in ranges:
        start = r[0]
        end = r[1]

        lstmouts, _, _ = inputs2lstmouts(l2rlstm, r2llstm, inputs[start: end], pdrop)
        ret.append(dy.concatenate([lstmouts[0], lstmouts[-1]]))

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


def aux_position(bunsetsu_ranges, pos_seq):
    ret = []

    for br in bunsetsu_ranges:
        ret.append(-1)
        for widx in range(br[1] - br[0]):
            ch1 = (td.i2x[pos_seq[br[0] + widx]])[0]
            ch2 = (td.i2x[pos_seq[br[0] + widx]])[-1]

            if ch1 == '助' or ch1 == '判' or ch2 == '点':
                ret[-1] = widx
                break

    return ret




cor_parsed_count = defaultdict(int)
done_sents = set()

def train(char_seqs, bipos_seqs, bi_b_seqs):
    losses_bunsetsu = []
    losses_arcs = []
    prev = time.time()



    print(pdrop)
    print(pdrop_bunsetsu)
    print("done_sents", len(done_sents))

    for it in range(train_iter):
        print("iteration: ", it)
        num_tot_bunsetsu_dep = 0
        num_tot_cor_bunsetsu_dep = 0
        num_tot_cor_bunsetsu_dep_not_argmax = 0
        tot_loss_in_iter = 0

        for i in (range(len(char_seqs))):
            if i % batch_size == 0:
                losses_bunsetsu = []
                losses_arcs = []

                dy.renew_cg()

            if random_pickup:  idx = i if not TRAIN else np.random.randint(len(char_seqs))
            else: idx = i

            if scheduled_learning and cor_parsed_count[idx] > 0:
                done_sents.add(idx)
                revise = np.random.randint(0, 10)
                if revise < 8:
                    continue

            if len(char_seqs[idx]) == 0 or len(bi_b_seqs[idx]) == 0:
                continue

            bi_w_seq = [0 if (bpd.i2x[b])[0] == 'B' else 1 for b in bipos_seqs[idx]]
            bi_w_seq = word_bi(bi_w_seq, bi_b_seqs[idx])

            word_ranges = word_range(bipos_seqs[idx])
            word_pos_seqs = word_pos(train_pos_seqs[idx], word_ranges)

            wembs = word_embds(char_seqs[idx], train_word_bipos_seqs[idx], train_pos_seqs[idx],
                               train_pos_sub_seqs[idx], train_wif_seqs[idx], train_wit_seqs[idx], word_ranges)

            wembs, l2r_outs, r2l_outs = inputs2lstmouts(l2rlstm_word, r2llstm_word, wembs, pdrop)

            loss_bi_bunsetsu, bi_bunsetsu_preds, _ = bi_bunsetsu_wembs(wembs, bi_w_seq)
            losses_bunsetsu.append(loss_bi_bunsetsu)

            bunsetsu_ranges = bunsetsu_range(bi_w_seq)
            aux_positions = aux_position(bunsetsu_ranges, word_pos_seqs)

            bembs = bunsetsu_embds(l2r_outs, r2l_outs, bunsetsu_ranges, aux_positions)

            bembs, _, _ = inputs2lstmouts(l2rlstm_bunsetsu, r2llstm_bunsetsu, bembs, pdrop_bunsetsu)

            arc_loss, arc_preds, arc_preds_not_argmax = dep_bunsetsu(bembs)

            if scheduled_learning or show_acc:
                num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds, train_chunk_deps[idx]))
                num_tot_cor_bunsetsu_dep_not_argmax += np.sum(np.equal(arc_preds_not_argmax[1:], train_chunk_deps[idx]))

            num_tot_bunsetsu_dep += len(bembs) - 1

            if scheduled_learning:
                if len(bi_w_seq) == len(bi_bunsetsu_preds) and \
                    np.sum(np.equal(arc_preds, train_chunk_deps[idx])) == len(train_chunk_deps[idx]) and \
                    np.sum(np.equal(bi_bunsetsu_preds, bi_w_seq)) == len(bi_w_seq):
                    cor_parsed_count[idx] += 1
                else:
                    if idx in done_sents:
                        cor_parsed_count[idx] = 0
                        done_sents.remove(idx)

            # losses_arcs.append(dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, train_chunk_deps[idx])))


            losses_arcs.append(dy.sum_batches(dy.pickneglogsoftmax_batch(arc_loss, train_chunk_deps[idx])))

            global global_step
            if i % batch_size == 0 and i != 0:

                losses_arcs.extend(losses_bunsetsu)

                sum_losses_arcs = dy.esum(losses_arcs)
                sum_losses_arcs_value = sum_losses_arcs.value()

                sum_losses_arcs.backward()
                update_parameters()
                global_step += 1
                tot_loss_in_iter += sum_losses_arcs_value

            if i % show_loss_every == 0 and i != 0:
                if show_acc:
                    print("dep accuracy: ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)
                    print("dep accuracy not argmax: ", num_tot_cor_bunsetsu_dep_not_argmax / num_tot_bunsetsu_dep)
        print("time in this iter: ", time.time() - prev)
        prev = time.time()
        print("total loss in this iter: ", tot_loss_in_iter)


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

        num_tot += len(char_seqs[i])

        word_ranges = word_range(bipos_seqs[i])
        word_pos_seq = word_pos(dev_pos_seqs[i], word_ranges)

        wembs = word_embds(char_seqs[i], dev_word_bipos_seqs[i], dev_pos_seqs[i],
                           dev_pos_sub_seqs[i], dev_wif_seqs[i], dev_wit_seqs[i], word_ranges)
        wembs, l2r_outs, r2l_outs = inputs2lstmouts(l2rlstm_word, r2llstm_word, wembs, pdrop)

        loss_bi_b, preds_bi_b, num_cor_bi_b = bi_bunsetsu_wembs(wembs, bi_w_seq)
        num_tot_bi_b += len(wembs)
        num_tot_cor_bi_b += num_cor_bi_b
        if i % show_acc_every == 0 and i != 0:
            print("accuracy chunking: ", num_tot_cor_bi_b / num_tot_bi_b)
            print("loss chuncking: ", loss_bi_b.value())
        gold_bunsetsu_ranges = bunsetsu_range(bi_w_seq)

        if chunker:
            failed_chunk = []
            for bidx, br in enumerate(gold_bunsetsu_ranges[1:]):
                start = br[0]
                end = br[1]
                if end == len(gold_bunsetsu_ranges[1:]):
                    end = - 1
                if np.sum(np.equal(bi_w_seq[start: end], preds_bi_b[start: end])) != len(bi_w_seq[start: end]):
                    failed_chunk.append(bidx)

            remains = [True] * len(gold_bunsetsu_ranges[1:])
            for fc in failed_chunk:
                remains[fc]
                dev_chunk_deps[i]
                chunks_excluded += np.sum(np.equal(dev_chunk_deps[i], fc)) + remains[fc]
                remains = [r * (1 - d) for r, d in zip(remains, np.equal(dev_chunk_deps[i], fc))]
                remains[fc] = False
        aux_positions = aux_position(gold_bunsetsu_ranges, word_pos_seq)
        bembs = bunsetsu_embds(l2r_outs, r2l_outs, gold_bunsetsu_ranges, aux_positions)

        bembs, _, _ = inputs2lstmouts(l2rlstm_bunsetsu, r2llstm_bunsetsu, bembs, pdrop_bunsetsu)

        if i % show_acc_every == 0 and i != 0:
            loss_bi_b_value = loss_bi_b.value()
            print(i, " bi_b loss")
            print(loss_bi_b_value)
            if chunker and num_tot_bi_b > 0:
                print(i, " accuracy chunking ", num_tot_cor_bi_b / num_tot_bi_b)
            if num_tot_bunsetsu_dep > 0:
                print(i, " accuracy dep ", num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)
                print(i, " accuracy dep ", num_tot_cor_bunsetsu_dep_not_argmax / num_tot_bunsetsu_dep)
            print("time: ", time.time() - prev)
            prev = time.time()
        if len(wembs) == num_cor_bi_b:
            complete_chunking += 1

        arc_loss, arc_preds, arc_preds_not_argmax = dep_bunsetsu(bembs)

        num_tot_bunsetsu_dep += len(bembs) - 1

        if len(arc_preds) != len(dev_chunk_deps[i]):
            failed_chunking += 1
            continue

        num_tot_cor_bunsetsu_dep += np.sum(np.equal(arc_preds, dev_chunk_deps[i]))
        if chunker:
            num_tot_cor_bunsetsu_dep += np.sum([r * d for r, d in zip(remains, np.equal(arc_preds, dev_chunk_deps[i]))])
            num_tot_cor_bunsetsu_dep_not_argmax += np.sum([r * d for r, d in zip(remains, np.equal(arc_preds_not_argmax[1:], dev_chunk_deps[i]))])
        num_tot_cor_bunsetsu_dep_not_argmax += np.sum(
            np.equal(arc_preds_not_argmax[1:], dev_chunk_deps[i]))

    global best_acc
    global update
    global early_stop_count
    if best_acc < num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep:
        best_acc = num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep
        update = True
        early_stop_count = 0
    if output_result:
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write(str(i) + " accuracy dep " + str(num_tot_cor_bunsetsu_dep / num_tot_bunsetsu_dep)+ '\n')
            f.write("total arc loss: " + str(total_loss) + '\n')
            if chunker:
                f.write(str(i) + " accuracy chunking " + str(num_tot_cor_bi_b / num_tot_bi_b) + '\n')
                f.write("complete chunking rate: " + str(complete_chunking / len(char_seqs))+ '\n')
                f.write("failed_chunking rate: " + str(failed_chunking / len(char_seqs))+ '\n')
                f.write("complete chunking: " + str(complete_chunking)+ '\n')
                f.write("failed_chunking: " + str(failed_chunking)+ '\n')

    print("total arc loss: " + str(total_loss))
    if chunker:
        print("complete_chunking rate: " + str(complete_chunking / len(char_seqs)))
        print("failed_chunking rate: " + str(failed_chunking / len(char_seqs)))
        print("complete chunking: " + str(complete_chunking))
        print("failed_chunking: " + str(failed_chunking))
        print("chunks_excluded: ", chunks_excluded)
    return


prev = time.time()

if LOAD:
    pc.populate(load_file)
    print("loaded from: ", load_file)

prev_epoc = 0

for e in range(epoc):

    if e * train_iter >= change_train_iter and train_iter != 1:
        train_iter = 1

    print("epoc: ", prev_epoc)
    prev_epoc += train_iter

    if output_result:
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write("time: " + str(prev) + '\n')
            f.write("epoc: " + str(prev_epoc) + '\n')

    TRAIN = True
    global pdrop
    global pdrop_bunsetsu
    pdrop = pdrop_stash
    pdrop_bunsetsu = pdrop_bunsetsu_stash
    if not TEST:
        train(train_char_seqs, train_word_bipos_seqs, train_chunk_bi_seqs)

    pdrop = 0.0
    pdrop_bunsetsu = 0.0
    TRAIN = False
    update = False

    dev(dev_char_seqs, dev_word_bipos_seqs, dev_chunk_bi_seqs)

    global early_stop_count
    if not update:
        early_stop_count += train_iter

    print("time: ", time.time() - prev)
    prev = time.time()

    if SAVE and not TEST and update:
        pc.save(save_file)
        print("saved into: ", save_file)

    if early_stop_count > early_stop:
        print("best_acc: ", best_acc)
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write("best_acc: " + str(best_acc))

        break
    else:
        print("best_acc: ", best_acc)
        print("early_stop_count: ", early_stop_count)
        with open(result_file, mode='a', encoding='utf-8') as f:
            f.write("best_acc: " + str(best_acc) + '\n')
            f.write("early_stop_count: " + str(early_stop_count) + '\n')



