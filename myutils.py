import numpy as np
import dynet as dy


def plot_loss(plt, loss, num_epoc, subplot_idx, xlim, ylim, ylim_lower):
    x = np.arange(0, num_epoc)
    y = np.array(loss)
    plt.subplot(2, 2, subplot_idx)
    plt.plot(x, y)
    plt.xlim(0, xlim)
    plt.ylim(ylim_lower, ylim)

    return

def attention_toprecur(embs, W_attn, bias_attn, x_attn):
    cell_boundary = embs[0].dim()[0][0] // 2

    if type(embs) is list:
        embs = dy.concatenate_cols(embs)
    embs = embs[:-cell_boundary]
    attn = dy.softmax(dy.transpose(embs) * (W_attn * x_attn + bias_attn))

    return embs * attn


def attention_ranges(embs, ranges, W_attn, bias_attn, X_attn, reverse=False):
    cell_boundary = embs[0].dim()[0][0] // 2

    ret = []
    for idx, r in enumerate(ranges):
        if r[1] - r[0] == 1:
            ret.append(embs[r[0] + 1])
        else:
            ret.append(
                dy.concatenate([embs[r[1] if not reverse else r[0] + 1][-cell_boundary:],
                                attention_toprecur(embs[r[0] + 1: r[1] + 1], W_attn, bias_attn, X_attn[idx])])
                # dy.concatenate([X_attn[idx], attention_toprecur(embs[r[0] + 1: r[1] + 1], W_attn, X_attn[idx])])
                # attention_toprecur(embs[r[0] + 1: r[1] + 1][:-cell_boundary], X_attn[idx])
            )
    return ret

