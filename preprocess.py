import glob
import config
import paths
import file_reader as fr
import get_pret_embs


class Vocab(object):
    def __init__(self):
        # select file split according to the setting
        self.train_dev_boundary = -1
        files = glob.glob(paths.path2KTC + 'syn/*.*')

        if config.CABOCHA_SPLIT:
            files = glob.glob(paths.path2KTC + 'syn/95010[1-9].*')
            self.train_dev_boundary = -1

        if config.STANDARD_SPLIT:
            files = glob.glob(paths.path2KTC + 'syn/95010[1-9].*')
            files.extend(glob.glob(paths.path2KTC + 'syn/95011[0-1].*'))
            files.extend(glob.glob(paths.path2KTC + 'syn/950[1-8]ED.*'))
            if config.TEST:
                files.extend(glob.glob(paths.path2KTC + 'syn/95011[4-7].*'))
                files.extend(glob.glob(paths.path2KTC + 'syn/951[0-2]ED.*'))
                self.train_dev_boundary = -7
            else:
                files.extend(glob.glob(paths.path2KTC + 'syn/95011[2-3].*'))
                files.extend(glob.glob(paths.path2KTC + 'syn/9509ED.*'))
                self.train_dev_boundary = -3

        if config.JOS:
            files = [paths.path2KTC + 'just-one-sentence.txt', paths.path2KTC + 'just-one-sentence.txt']

        if config.MINI_SET:
            files = [paths.path2KTC + 'miniKTC_train.txt', paths.path2KTC + 'miniKTC_dev.txt']

        save_file = 'KTC'
        split_name = ""

        if config.CABOCHA_SPLIT:
            split_name = "_CABOCHA"
        elif config.STANDARD_SPLIT:
            split_name = "_STANDARD"
        elif config.MINI_SET:
            split_name = "_MINISET"

        self.save_file = config.save_file_directory + save_file + split_name

        print(files)


        # prepare dictionaries and sequences of feature ids

        df = fr.DataFrameKtc

        self.train_sents = []
        for file in files[0:self.train_dev_boundary]:
            print('[train] reading this file: ', file)
            lines = df.file2lines(df, file, ' ', config.encoding)
            self.train_sents.extend(df.lines2sents(df, lines))
        self.wd, self.cd, self.bpd, self.td, self.tsd, self.wifd, self.witd = df.sents2dicts(df, self.train_sents)

        self.wd.freeze()
        self.cd.freeze()
        self.bpd.freeze()
        self.td.freeze()
        self.tsd.freeze()
        self.wifd.freeze()
        self.witd.freeze()

        self.dev_sents = []
        for file in files[self.train_dev_boundary:]:
            print('[dev] reading this file: ', file)
            lines = df.file2lines(df, file, ' ', config.encoding)
            self.dev_sents.extend(df.lines2sents(df, lines))

        for sent in self.dev_sents:
            for w in sent.word_forms:
                self.wd.add_entry(w)
            for c in sent.char_forms:
                self.cd.add_entry(c)
            for bp in sent.word_biposes:
                self.bpd.add_entry(bp)
                self.td.add_entry(bp[2:])
            for ps in sent.pos_sub:
                self.tsd.add_entry(ps)
            for wif in sent.word_inflection_forms:
                self.wifd.add_entry(wif)
            for wit in sent.word_inflection_types:
                self.witd.add_entry(wit)


        self.train_word_seqs, self.train_char_seqs, self.train_word_bipos_seqs, \
        self.train_chunk_bi_seqs, self.train_chunk_deps, self.train_pos_seqs, self.train_word_bi_seqs, \
        self.train_pos_sub_seqs, self.train_wif_seqs, self.train_wit_seqs \
            = df.sents2ids([self.wd, self.cd, self.bpd, self.td, self.tsd, self.wifd, self.witd], self.train_sents)


        self.dev_word_seqs, self.dev_char_seqs, self.dev_word_bipos_seqs, \
        self.dev_chunk_bi_seqs, self.dev_chunk_deps, self.dev_pos_seqs, self.dev_word_bi_seqs, \
        self.dev_pos_sub_seqs, self.dev_wif_seqs, self.dev_wit_seqs \
            = df.sents2ids([self.wd, self.cd, self.bpd, self.td, self.tsd, self.wifd, self.witd], self.dev_sents)

        self.train_ids = df.sents2ids([self.wd, self.cd, self.bpd, self.td, self.tsd, self.wifd, self.witd], self.train_sents)


        self.dev_ids = df.sents2ids([self.wd, self.cd, self.bpd, self.td, self.tsd, self.wifd, self.witd], self.dev_sents)

        if config.add_pret_embs:
            self._pret_embs = get_pret_embs.get_pret_embs()
            self._pret_vocab = set(self._pret_embs.index2word)

        return


