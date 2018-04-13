# -*- coding: UTF-8 -*-
import dynet as dy
import numpy as np


# from data import Vocab
from tarjan import Tarjan

import config

def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc):
    builder = dy.VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc)
    for layer, params in enumerate(builder.get_parameters()):
        W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (
            lstm_hiddens if layer > 0 else input_dims))  # the first layer takes prev hidden and input vec
        W_h, W_x = W[:, :lstm_hiddens], W[:, lstm_hiddens:]
        params[0].set_value(np.concatenate([W_x] * 4, 0))
        params[1].set_value(np.concatenate([W_h] * 4, 0))
        b = np.zeros(4 * lstm_hiddens, dtype=np.float32)
        b[lstm_hiddens:2 * lstm_hiddens] = -1.0  # fill second quarter of bias vec with -1.0
        params[2].set_value(b)
    return builder


def biLSTM(builders, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
    for fb, bb in builders:
        f, b = fb.initial_state(), bb.initial_state()
        fb.set_dropouts(dropout_x, dropout_h)
        bb.set_dropouts(dropout_x, dropout_h)
        if batch_size is not None:
            fb.set_dropout_masks(batch_size)
            bb.set_dropout_masks(batch_size)
        fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
        inputs = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
    return inputs


def leaky_relu(x):
    return dy.bmax(.1 * x, x)



def biED(x, V_r, V_i, y, seq_len, num_outputs, bias_x=None, bias_y=None):

    W_r = dy.concatenate_cols([V_r] * seq_len)
    W_i = dy.concatenate_cols([V_i] * seq_len)

    input_size = x.dim()[0][0]

    x_r, x_i = x[:- input_size // 2], x[input_size // 2:]
    y_r, y_i = y[:- input_size // 2], y[input_size // 2:]

    B = dy.inputTensor(np.zeros((seq_len * num_outputs, seq_len), dtype=np.float32))

    if bias_x:
        bias = dy.reshape(dy.concatenate_cols([bias_x] * seq_len), (seq_len * num_outputs, input_size // 2))
        B += bias * (x_r + x_i)
    if bias_y:
        bias = dy.reshape(dy.concatenate_cols([bias_y] * seq_len), (seq_len * num_outputs, input_size // 2))
        B += bias * (y_r + y_i)

    y_r = dy.concatenate([y_r] * num_outputs)
    y_i = dy.concatenate([y_i] * num_outputs)

    # X = dy.concatenate([x_r, x_i, x_r, -x_i])
    # Y = dy.concatenate([y_r, y_i, y_i, -y_r])
    # W = dy.concatenate([W_r, W_r, W_i, -W_i])
    # WY = dy.reshape(dy.cmult(W, Y), (input_size // 2 * 4, seq_len * num_outputs))

    if config.anti_symmetric:
        X = dy.concatenate([x_r, -x_i])
        Y = dy.concatenate([y_i, -y_r])
        W = dy.concatenate([W_i, -W_i])
        WY = dy.reshape(dy.cmult(W, Y), (input_size, seq_len * num_outputs))
    elif config.comp:
        X = dy.concatenate([x_r, x_i, x_r, -x_i])
        Y = dy.concatenate([y_r, y_i, y_i, -y_r])
        W = dy.concatenate([W_r, W_r, W_i, -W_i])
        WY = dy.reshape(dy.cmult(W, Y), (input_size // 2 * 4, seq_len * num_outputs))
    elif config.symmetric or config.cancel_lower:
        X = dy.concatenate([x_r, x_i])
        Y = dy.concatenate([y_r, y_i])
        W = dy.concatenate([W_r, W_i])
        WY = dy.reshape(dy.cmult(W, Y), (input_size, seq_len * num_outputs))
    else:
        raise ValueError('choose one mode from these options: symmetric, anti_symmetric and comp.')

    blin = dy.transpose(X) * WY + dy.reshape(B, (seq_len, seq_len * num_outputs))


    # W_r = dy.reshape(W_r, (seq_len * num_outputs, input_size // 2))
    # W_i = dy.reshape(W_i, (seq_len * num_outputs, input_size // 2))
    # x_r = dy.reshape(x_r, (seq_len * num_outputs, input_size // 2))
    # x_i = dy.reshape(x_i, (seq_len * num_outputs, input_size // 2))
    #
    # blin_rrr = x_r * dy.cmult(W_r, y_r)
    # blin_rii = x_i * dy.cmult(W_r, y_i)
    # blin_iri = x_r * dy.cmult(W_i, y_i)
    # blin_iir = x_i * dy.cmult(W_i, y_r)
    #
    # blin = blin_rrr + blin_rii + blin_iri - blin_iir + B

    if num_outputs > 1:
        blin = dy.reshape(blin, (seq_len, num_outputs, seq_len))

    return blin


def check_nega_posi(M):
    nega_count = 0
    elem_count = 0

    for i in range(0, M.shape[0]):
        for j in range(i + 1, M.shape[1]):
            elem_count += 1
            if M[i][j] < 0:
                nega_count += 1

    return nega_count, elem_count


# def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
#     # x,y: (input_size x seq_len) x batch_size
#     if bias_x:
#         x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
#     if bias_y:
#         y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
#
#     nx, ny = input_size + bias_x, input_size + bias_y
#     # W: (num_outputs x ny) x nx
#     lin = W * x
#     if num_outputs > 1:
#         lin = dy.reshape(lin, (ny, num_outputs * seq_len), batch_size=batch_size)
#     blin = dy.transpose(y) * lin
#     if num_outputs > 1:
#         blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size=batch_size)
#     # seq_len_y x seq_len_x if output_size == 1
#     # seq_len_y x num_outputs x seq_len_x else
#     return blin


def orthonormal_initializer(output_size, input_size):
    """
	adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
	"""
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        # for i in xrange(100):
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def left_arc_mask(N, transpose=True):
    mask = np.array([])
    for i in range(1, N + 1):
        ones = np.ones(i)
        zeros = np.zeros(N - i)
        one_zero = np.concatenate((ones, zeros))

        mask = np.concatenate((mask, one_zero), axis=0)
    mask = np.reshape(mask, (N, N))

    if transpose:
        mask = np.transpose(mask)

    return mask

def arc_argmax(parse_probs, length, tokens_to_keep, ensure_tree=True):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
    """
    if ensure_tree:
        I = np.eye(len(tokens_to_keep))
        # block loops and pad heads
        arc_masks = left_arc_mask(length)
        parse_probs = parse_probs * arc_masks
        parse_probs = np.reshape((parse_probs), (len(tokens_to_keep), len(tokens_to_keep)))
        tmp = parse_probs[-1:, :]
        parse_probs = np.concatenate((parse_probs[:-1, -1:], parse_probs[:-1, :-1]), axis=1)
        parse_probs = np.concatenate((tmp, parse_probs))

        parse_probs = parse_probs * tokens_to_keep * (1 - I)
        parse_preds = np.argmax(parse_probs, axis=1)
        tokens = np.arange(1, length) #original
        # tokens = np.arange(length) #modified
        # root_idx = len(tokens_to_keep) - 1
        root_idx = 0

        roots = np.where(parse_preds[tokens] == root_idx)[0] + 1 #original
        # roots = np.where(parse_preds[tokens] == 0)[0] #modified
        # ensure at least one root
        if len(roots) < 1:
            # global root_0
            # root_0 += 1

            # The current root probabilities
            root_probs = parse_probs[tokens, root_idx]
            # The current head probabilities
            old_head_probs = parse_probs[tokens, parse_preds[tokens]]
            # Get new potential root probabilities
            new_root_probs = root_probs / old_head_probs
            # Select the most probable root
            new_root = tokens[np.argmax(new_root_probs)]
            # Make the change
            parse_preds[new_root] = root_idx
        # ensure at most one root
        elif len(roots) > 1:
            # global root_more_than_1
            # root_more_than_1 += 1

            # The probabilities of the current heads
            root_probs = parse_probs[roots, root_idx]
            # Set the probability of depending on the root zero
            parse_probs[roots, root_idx] = 0
            # Get new potential heads and their probabilities
            new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) + 1 # original line
            # new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) # modified line
            new_head_probs = parse_probs[roots, new_heads] / root_probs
            # Select the most probable root
            new_root = roots[np.argmin(new_head_probs)]
            # Make the change
            parse_preds[roots] = new_heads
            parse_preds[new_root] = root_idx
        # remove cycles
        tarjan = Tarjan(parse_preds, tokens)
        cycles = tarjan.SCCs
        for SCC in tarjan.SCCs:
            # global circle_count
            # circle_count += 1

            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) + 1 #original
                # new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) #modified
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds
    else:
        # block and pad heads
        parse_probs = parse_probs * tokens_to_keep
        parse_preds = np.argmax(parse_probs, axis=1)
        return parse_preds

#
#
# def rel_argmax(rel_probs, length, ensure_tree=True):
#     """
# 	adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py
# 	"""
#     if ensure_tree:
#         rel_probs[:, Vocab.PAD] = 0
#         root = Vocab.ROOT
#         tokens = np.arange(1, length)
#         rel_preds = np.argmax(rel_probs, axis=1)
#         roots = np.where(rel_preds[tokens] == root)[0] + 1
#         if len(roots) < 1:
#             rel_preds[1 + np.argmax(rel_probs[tokens, root])] = root
#         elif len(roots) > 1:
#             root_probs = rel_probs[roots, root]
#             rel_probs[roots, root] = 0
#             new_rel_preds = np.argmax(rel_probs[roots], axis=1)
#             new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
#             new_root = roots[np.argmin(new_rel_probs)]
#             rel_preds[roots] = new_rel_preds
#             rel_preds[new_root] = root
#         return rel_preds
#     else:
#         rel_probs[:, Vocab.PAD] = 0
#         rel_preds = np.argmax(rel_probs, axis=1)
#         return rel_preds
