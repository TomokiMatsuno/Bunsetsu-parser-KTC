import codecs
import re
import config


def make_line_chunks(lines, trigger, end_marker=None):
    lines_idx = 0
    line_chuncks = []
    pattern = trigger
    repattern = re.compile(pattern)

    while not repattern.match(lines[lines_idx][0]):
        lines_idx += 1

    while lines_idx < len(lines) and lines[lines_idx][0]:
        if lines[lines_idx][0] == end_marker:
            break

        empty_list = []
        if repattern.match(lines[lines_idx][0]):
            line_chuncks.append(empty_list)
            line_chuncks[-1].append(lines[lines_idx])
            lines_idx += 1
            continue
        line_chuncks[-1].append(lines[lines_idx])
        lines_idx += 1

    return line_chuncks


class DataFrame:
    def __init__(self):
        return

    def file2lines(self, input_file, delimiter, encoding='euc-jp'):
        with codecs.open(input_file, "r", encoding) as f:
            data = f.read()
            f.close()

        lines = data.split('\n')

        sents = []
        for l in lines:
            tokens = l.split(delimiter)
            sents.append(tokens)
        return sents

    def line2tokens(self, line, indices=None):
        tokens = []

        if indices is None:
            for idx in range(len(line)):
                tokens.append(tokens[idx])
            return

        else:
            for idx in indices:
                tokens.append(line[idx])
            return tokens


class Char:
    def __init__(self, form, bipos, chunk_bi):
        self.form = form
        self.word_bipos = bipos
        self.chunk_bi = chunk_bi


class Word:
    def __init__(self, line, chunk_bi):
        self.feats = []
        self.chars = []
        self.chunk_bi = chunk_bi
        for token in line:
            self.feats.append(token)

        for i in range(len(line[0])):
            if i == 0:
                bi = 'B'
            else:
                bi = 'I'

            if config.p_ps_separated:
                bipos = bi + '_' + line[3]
            else:
                bipos = bi + '_' + line[3] + '_' + line[4]
            if i == 0:
                self.chars.append(Char(line[0][i], bipos, chunk_bi))
            else:
                self.chars.append(Char(line[0][i], bipos, 'I'))       #set chunk bi to be 'I' if it is not the beginning of a chunk


class Chunk:
    def __init__(self, lines):
        self.id = int(lines[0][1])
        self.head = int(lines[0][2][0:-1]) + 1#if int(lines[0][2][0:-1]) >= 0 else 0
        self.rel = lines[0][2][-1]
        self.words = []
        chunk_bi = 'B'
        for l in lines[1:]:
            self.add_word(l, chunk_bi)
            chunk_bi = 'I'

    def add_word(self, line, chunk_bi):
        self.words.append(Word(line, chunk_bi))


class Sent:
    def __init__(self, lines):
        self.id = lines[0][1]
        self.chunks = []
        self.word_forms = []
        self.word_inflection_types = []
        self.word_inflection_forms = []
        self.pos = []
        self.pos_sub = []
        self.char_forms = []
        self.word_biposes = []
        self.chunk_bis = []

        self.make_chunks(lines)
        for ch in self.chunks:
            for word in ch.words:
                self.word_forms.append(word.feats[0])
                self.pos.append(word.feats[3])
                self.pos_sub.append(word.feats[4])
                self.word_inflection_forms.append(word.feats[5])
                self.word_inflection_types.append(word.feats[6])
                self.chunk_bis.append(word.chunk_bi)
                for char in word.chars:
                    self.char_forms.append(char.form)
                    self.word_biposes.append(char.word_bipos)
                    # self.chunk_bis.append(char.chunk_bi)



    def make_chunks(self, lines):
        chunk_lines = make_line_chunks(lines, '\*', 'EOS')

        for cls in chunk_lines:
            self.chunks.append(Chunk(cls))

class Dict:
    def __init__(self, doc, initial_entries=None):
        self.i2x = {}
        self.x2i = {}
        self.appeared_x2i = {}
        self.appeared_i2x = {}
        self.freezed = False
        self.initial_entries = initial_entries

        if initial_entries is not None:
            for ent in initial_entries:
                self.add_entry(ent)

        for line in doc:
            for ent in line:
                self.add_entry(ent)

    # def add_entry(self, ent):
    #     if ent not in self.x2i or self.x2i[ent] == self.x2i["UNK"]:
    #         if ((not self.freezed) and ent in self.appeared_x2i) or ent in self.initial_entries:
    #             if ent in self.initial_entries:
    #                 self.i2x[len(self.i2x)] = ent
    #                 self.x2i[ent] = len(self.x2i)
    #                 self.appeared_x2i[ent] = len(self.appeared_x2i)
    #                 self.appeared_i2x[len(self.appeared_i2x)] = ent
    #             else:
    #                 self.i2x[self.appeared_x2i[ent]] = ent
    #                 self.x2i[ent] = self.appeared_x2i[ent]#len(self.x2i)
    #         else:
    #             self.appeared_x2i[ent] = len(self.appeared_x2i)
    #             self.appeared_i2x[len(self.appeared_i2x)] = ent
    #             self.x2i[ent] = self.x2i["UNK"]

    def add_entry(self, ent):
        if ent not in self.x2i:
            if not self.freezed:
                self.i2x[len(self.i2x)] = ent
                self.x2i[ent] = len(self.x2i)
            else:
                self.x2i[ent] = self.x2i["UNK"]


    def freeze(self):
        self.freezed = True


class DataFrameUD(DataFrame):
    def __init__(self, lines):
        self.sents = self.lines2sents(lines)

    def lines2sents(self, lines):
        ret = []

        sent_lines = make_line_chunks(lines, '#', '\n')

        for sl in sent_lines:
            tmp = [sl[0]]
            # tmp.append(["ROOT", "ROOT", 0, "ROOT"])
            for l in sl[1:]:
                tmp.append(self.line2tokens(self, l, [1, 3, 6, 7]))
        ret.append(tmp)

        return ret

    def sents2dicts(self, sents, initial_entries=None):
        doc_word_forms = []
        doc_biupos_forms = []
        doc_head = []
        doc_rel = []

        for sent in sents:
            B = True
            for s in sent[1:]:
                doc_word_forms.append(s[0])
                if B:
                    doc_biupos_forms.append("B_" + s[1])
                else:
                    doc_biupos_forms.append("I_" + s[1])
                doc_head.append(s[2])
                doc_rel.append(s[3])
                B = False
        if initial_entries is None:
            initial_entries = ["NULL", "UNK", "ROOT"]
        wd = Dict(doc_word_forms, initial_entries)
        bupd = Dict(doc_biupos_forms, initial_entries)
        rd = Dict(doc_rel, ["NULL"])

        return wd, bupd, rd




class DataFrameKtc(DataFrame):
    def __init__(self, lines):
        self.sents = self.lines2sents(lines)

    def lines2sents(self, lines):
        ret = []
        sent_lines = make_line_chunks(lines, '#')

        for sl in sent_lines:
            ret.append(Sent(sl))

        return ret


    def sents2dicts(self, sents, initial_entries=None):
        doc_word_forms = []
        doc_char_forms = []
        doc_word_biposes = []
        doc_chunk_bis = []
        doc_pos = []
        doc_pos_sub = []
        doc_word_inflection_forms = []
        doc_word_inflection_types = []

        for sent in sents:
            doc_word_forms.append(sent.word_forms)
            doc_char_forms.append(sent.char_forms)
            doc_word_biposes.append(sent.word_biposes)
            doc_chunk_bis.append(sent.chunk_bis)
            doc_pos_sub.append(sent.pos_sub)
            doc_word_inflection_forms.append(sent.word_inflection_forms)
            doc_word_inflection_types.append(sent.word_inflection_types)

            poss = []
            for bp in sent.word_biposes:
                poss.append(bp[2:])
            doc_pos.append(poss)

        if initial_entries is None:
            initial_entries = ["NULL", "UNK", "ROOT", "BOS", "EOS"]
        if config.insert_dummy:
            initial_entries = initial_entries + ["DUMMY"]
        wd = Dict(doc_word_forms, initial_entries)
        cd = Dict(doc_char_forms, initial_entries)
        bpd = Dict(doc_word_biposes, ["NULL", "UNK", "B_ROOT", "B_BOS", "B_EOS", "B_DUMMY"])
        td = Dict(doc_pos, initial_entries)
        tsd = Dict(doc_pos_sub, initial_entries)
        wif = Dict(doc_word_inflection_forms, initial_entries)
        wit = Dict(doc_word_inflection_types, initial_entries)

        return wd, cd, bpd, td, tsd, wif, wit


    def sents2ids(dicts, sents):
        wd = dicts[0]
        cd = dicts[1]
        bpd = dicts[2]
        td = dicts[3]
        tsd = dicts[4]
        wifd = dicts[5]
        witd = dicts[6]

        word_seqs = []
        char_seqs = []
        word_bipos_seqs = []
        chunk_bi_seqs = []
        chunk_deps = []
        pos_seqs = []
        bi_seqs = []

        pos_sub_seqs = []
        word_inflection_form_seqs = []
        word_inflection_types_seqs = []


        for sent in sents:
            tmp_word = []
            tmp_char = []
            tmp_word_bipos = []
            tmp_chunk_bi = []
            tmp_chunk_dep = []
            tmp_pos = []
            tmp_bi = []
            tmp_pos_sub = []
            tmp_wif = []
            tmp_wit = []

            # if config.BOS_EOS:
            #     # tmp_word.append(wd.x2i["BOS"])
            #     tmp_char.append(cd.x2i["BOS"])
            #     tmp_word_bipos.append(bpd.x2i["B_BOS"])
            #     tmp_chunk_bi.append(0)
            #     tmp_pos.append(td.x2i["BOS"])
            #     tmp_bi.append(0)
                # tmp_pos_sub.append(tsd.x2i["BOS"])
                # tmp_wif.append(wifd.x2i["BOS"])
                # tmp_wit.append(witd.x2i["BOS"])

            if config.BOS_EOS:
                tmp_word.append(wd.x2i["BOS"])
                tmp_char.append(cd.x2i["BOS"])
                tmp_word_bipos.append(bpd.x2i["B_BOS"])
                # tmp_chunk_bi.append(0)
                tmp_pos.append(td.x2i["BOS"])
                tmp_bi.append(0)
                tmp_pos_sub.append(tsd.x2i["BOS"])
                tmp_wif.append(wifd.x2i["BOS"])
                tmp_wit.append(witd.x2i["BOS"])

            if config.root:
                tmp_word.append(wd.x2i["ROOT"])
                tmp_char.append(cd.x2i["ROOT"])
                tmp_word_bipos.append(bpd.x2i["B_ROOT"])
                tmp_chunk_bi.append(0)
                tmp_pos.append(td.x2i["ROOT"])
                tmp_bi.append(0)
                tmp_pos_sub.append(tsd.x2i["ROOT"])
                tmp_wif.append(wifd.x2i["ROOT"])
                tmp_wit.append(witd.x2i["ROOT"])
                # tmp_chunk_dep.append(0)
            if config.insert_dummy and config.root:
                tmp_word.append(wd.x2i["DUMMY"])
                tmp_char.append(cd.x2i["DUMMY"])
                tmp_word_bipos.append(bpd.x2i["B_DUMMY"])
                tmp_chunk_bi.append(1)
                tmp_pos.append(td.x2i["DUMMY"])
                tmp_bi.append(0)
                tmp_pos_sub.append(tsd.x2i["DUMMY"])
                tmp_wif.append(wifd.x2i["DUMMY"])
                tmp_wit.append(witd.x2i["DUMMY"])

            contains_func = False
            func_words_pos = ["助", "判", "句", "読"]

            chunk_bi_word = [0 if cbi == 'B' else 1 for cbi in sent.chunk_bis]
            # chunk_bi_word = []
            # for wbipos, chbi in zip(sent.word_biposes, sent.chunk_bis):
            #     if wbipos[0] == 'B':
            #         if chbi == 'B':
            #             chunk_bi_word.append(0)
            #         else:
            #             chunk_bi_word.append(1)

            if config.insert_dummy:
                for pidx in range(len(sent.pos)):
                    if pidx == len(sent.pos) - 1:
                        # tmp_pos.append(td.x2i[sent.pos[pidx]])
                        # tmp_word.append(wd.x2i[sent.word_forms[pidx]])
                        tmp_pos.append(td.x2i[sent.pos[pidx]])
                        tmp_word.append(wd.x2i[sent.word_forms[pidx]])
                        for w in sent.word_forms[pidx]:
                            for c in w:
                                tmp_char.append(cd.x2i[c])
                        BI = 'B_'
                        for bpidx in range(len(sent.word_forms[pidx])):
                            tmp_word_bipos.append(bpd.x2i[BI + sent.pos[pidx]])
                            tmp_bi.append(0 if BI[0] == 'B' else 1)
                            BI = 'I_'

                        tmp_pos_sub.append(tsd.x2i[sent.pos_sub[pidx]])
                        tmp_wif.append(wifd.x2i[sent.word_inflection_forms[pidx]])
                        tmp_wit.append(witd.x2i[sent.word_inflection_types[pidx]])

                        tmp_chunk_bi.append(0 if sent.chunk_bis[pidx] == 'B' else 1)
                        break


                    tmp_pos.append(td.x2i[sent.pos[pidx]])
                    tmp_word.append(wd.x2i[sent.word_forms[pidx]])
                    for w in sent.word_forms[pidx]:
                        for c in w:
                            tmp_char.append(cd.x2i[c])

                    tmp_chunk_bi.append(0 if sent.chunk_bis[pidx] == 'B' else 1)

                    BI = 'B_'
                    for bpidx in range(len(sent.word_forms[pidx])):
                        tmp_word_bipos.append(bpd.x2i[BI + sent.pos[pidx]])
                        tmp_bi.append(0 if BI[0] == 'B' else 1)
                        BI = 'I_'

                    tmp_pos_sub.append(tsd.x2i[sent.pos_sub[pidx]])
                    tmp_wif.append(wifd.x2i[sent.word_inflection_forms[pidx]])
                    tmp_wit.append(witd.x2i[sent.word_inflection_types[pidx]])

                    if (sent.pos[pidx][0] in func_words_pos \
                            or sent.pos_sub[pidx][0] in func_words_pos):
                        contains_func = True
                    if chunk_bi_word[pidx + 1] == 0:
                        if not contains_func:
                            tmp_pos.append(td.x2i["DUMMY"])
                            tmp_word.append(wd.x2i["DUMMY"])
                            tmp_pos_sub.append(tsd.x2i["DUMMY"])
                            tmp_wif.append(wifd.x2i["DUMMY"])
                            tmp_wit.append(witd.x2i["DUMMY"])
                            tmp_char.append(cd.x2i["DUMMY"])
                            tmp_word_bipos.append(bpd.x2i["B_DUMMY"])
                            tmp_bi.append(0)
                            tmp_chunk_bi.append(1)

                        contains_func = False



            else:
                for wf in sent.word_forms:
                    tmp_word.append(wd.x2i[wf])
                for cf in sent.char_forms:
                    tmp_char.append(cd.x2i[cf])
                for bp in sent.word_biposes:
                    tmp_word_bipos.append(bpd.x2i[bp])
                    # tmp_pos.append(td.x2i[bp[2:]])
                    tmp_bi.append(0 if bp[0] == 'B' else 1)

                for cbi in sent.chunk_bis:
                    tmp_chunk_bi.append(0 if cbi == 'B' else 1)
                for p in sent.pos:
                    tmp_pos.append(td.x2i[p])

                for ps in sent.pos_sub:
                    tmp_pos_sub.append(tsd.x2i[ps])
                for wif in sent.word_inflection_forms:
                    tmp_wif.append(wifd.x2i[wif])
                for wit in sent.word_inflection_types:
                    tmp_wit.append(witd.x2i[wit])

            for ch in sent.chunks:
                tmp_chunk_dep.append(ch.head)

            if config.BOS_EOS:
                tmp_word.append(wd.x2i["EOS"])
                tmp_char.append(cd.x2i["EOS"])
                tmp_word_bipos.append(bpd.x2i["B_EOS"])
                # tmp_chunk_bi.append(0)
                tmp_pos.append(td.x2i["EOS"])
                tmp_bi.append(0)
                tmp_pos_sub.append(tsd.x2i["EOS"])
                tmp_wif.append(wifd.x2i["EOS"])
                tmp_wit.append(witd.x2i["EOS"])

            word_seqs.append(tmp_word)
            char_seqs.append(tmp_char)
            word_bipos_seqs.append(tmp_word_bipos)
            chunk_bi_seqs.append(tmp_chunk_bi)
            # chunk_bi_seqs.append(chunk_bi_word)

            # chunk_deps.append([0] + tmp_chunk_dep)
            chunk_deps.append(tmp_chunk_dep)
            pos_seqs.append(tmp_pos)
            bi_seqs.append(tmp_bi)
            pos_sub_seqs.append(tmp_pos_sub)
            word_inflection_form_seqs.append(tmp_wif)
            word_inflection_types_seqs.append(tmp_wit)

            # if config.BOS_EOS:
            #     # tmp_word.append(wd.x2i["EOS"])
            #     tmp_char.append(cd.x2i["EOS"])
            #     tmp_word_bipos.append(bpd.x2i["B_EOS"])
            #     tmp_chunk_bi.append(0)
            #     tmp_pos.append(td.x2i["EOS"])
            #     tmp_bi.append(0)
            #     # tmp_pos_sub.append(tsd.x2i["EOS"])
            #     # tmp_wif.append(wifd.x2i["EOS"])
            #     # tmp_wit.append(witd.x2i["EOS"])

        return word_seqs, char_seqs, word_bipos_seqs, \
               chunk_bi_seqs, chunk_deps, pos_seqs, bi_seqs, \
               pos_sub_seqs, word_inflection_form_seqs, word_inflection_types_seqs

