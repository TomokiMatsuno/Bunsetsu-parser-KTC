import config
import utils
import dynet as dy

class Parser(object):
    def __init__(self,
                 LAYERS_character,
                 LAYERS_word,
                 LAYERS_bunsetsu,
                 WORDS_SIZE,
                 CHARS_SIZE,
                 BIPOS_SIZE,
                 POS_SIZE,
                 POSSUB_SIZE,
                 WIF_SIZE,
                 WIT_SIZE
                 ):

        self._pc = dy.ParameterCollection()
        if not config.use_annealing:
            self._trainer = dy.AdadeltaTrainer(self._pc)
        else:
            self._trainer = dy.AdamTrainer(self._pc, config.learning_rate, config.beta_1, config.beta_2, config.epsilon)

        if not config.orthonormal:
            self._l2rlstm_char = dy.VanillaLSTMBuilder(LAYERS_character, config.INPUT_DIM * 1, config.INPUT_DIM // 2, self._pc, config.layer_norm)
            self._r2llstm_char = dy.VanillaLSTMBuilder(LAYERS_character, config.INPUT_DIM * 1, config.INPUT_DIM // 2, self._pc, config.layer_norm)

            self._l2rlstm_word = dy.VanillaLSTMBuilder(LAYERS_word, config.INPUT_DIM * ((2 * (config.use_cembs) + 2) + config.use_wif_wit * 2), config.word_HIDDEN_DIM, self._pc, config.layer_norm)
            self._r2llstm_word = dy.VanillaLSTMBuilder(LAYERS_word, config.INPUT_DIM * ((2 * (config.use_cembs) + 2) + config.use_wif_wit * 2), config.word_HIDDEN_DIM, self._pc, config.layer_norm)

            if config.bembs_average_flag:
                self._l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, config.INPUT_DIM * ((2 * (config.use_cembs) + 2) + config.use_wif_wit * 2), config.bunsetsu_HIDDEN_DIM, self._pc, config.layer_norm)
                self._r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, config.INPUT_DIM * ((2 * (config.use_cembs) + 2) + config.use_wif_wit * 2), config.bunsetsu_HIDDEN_DIM, self._pc, config.layer_norm)
            else:
                self._l2rlstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, config.word_HIDDEN_DIM, config.bunsetsu_HIDDEN_DIM, self._pc, config.layer_norm)
                self._r2llstm_bunsetsu = dy.VanillaLSTMBuilder(LAYERS_bunsetsu, config.word_HIDDEN_DIM, config.bunsetsu_HIDDEN_DIM, self._pc, config.layer_norm)

        else:
            self._l2rlstm_char = utils.orthonormal_VanillaLSTMBuilder(LAYERS_character, config.INPUT_DIM * 1, config.INPUT_DIM // 2, self._pc)
            self._r2llstm_char = utils.orthonormal_VanillaLSTMBuilder(LAYERS_character, config.INPUT_DIM * 1, config.INPUT_DIM // 2, self._pc)

            self._l2rlstm_word = utils.orthonormal_VanillaLSTMBuilder(LAYERS_word, config.INPUT_DIM * ((2 * (config.use_cembs) + 2) + config.use_wif_wit * 2), config.word_HIDDEN_DIM, self._pc)
            self._r2llstm_word = utils.orthonormal_VanillaLSTMBuilder(LAYERS_word, config.INPUT_DIM * ((2 * (config.use_cembs) + 2) + config.use_wif_wit * 2), config.word_HIDDEN_DIM, self._pc)

            if config.bembs_average_flag:
                self._l2rlstm_bunsetsu = utils.orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, config.INPUT_DIM * ((2 * (config.use_cembs) + 2) + config.use_wif_wit * 2), config.bunsetsu_HIDDEN_DIM, self._pc)
                self._r2llstm_bunsetsu = utils.orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, config.INPUT_DIM * ((2 * (config.use_cembs) + 2) + config.use_wif_wit * 2), config.bunsetsu_HIDDEN_DIM, self._pc)
            else:
                self._l2rlstm_bunsetsu = utils.orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, config.word_HIDDEN_DIM, config.bunsetsu_HIDDEN_DIM, self._pc)
                self._r2llstm_bunsetsu = utils.orthonormal_VanillaLSTMBuilder(LAYERS_bunsetsu, config.word_HIDDEN_DIM, config.bunsetsu_HIDDEN_DIM, self._pc)

        self._params = dict()
        self._params["lp_w"] = self._pc.add_lookup_parameters(
            (WORDS_SIZE + 1, config.INPUT_DIM),
            init=dy.ConstInitializer(0.))
        self._params["lp_c"] = self._pc.add_lookup_parameters(
            (CHARS_SIZE + 1, config.INPUT_DIM),
            init=dy.ConstInitializer(0.))
        self._params["lp_bp"] = self._pc.add_lookup_parameters(
            (BIPOS_SIZE + 1, config.INPUT_DIM),
            init=dy.ConstInitializer(0.))
        self._params["lp_p"] = self._pc.add_lookup_parameters(
            (POS_SIZE + 1, (config.INPUT_DIM // 10) * 5 * (1 + config.use_wif_wit)),
            init=dy.ConstInitializer(0.))
        self._params["lp_ps"] = self._pc.add_lookup_parameters((
            POSSUB_SIZE + 1, (config.INPUT_DIM // 10) * 5 * (1 + config.use_wif_wit)),
            init=dy.ConstInitializer(0.))
        self._params["lp_wif"] = self._pc.add_lookup_parameters(
            (WIF_SIZE + 1, (config.INPUT_DIM // 10) * 5),
            init=dy.ConstInitializer(0.))
        self._params["lp_wit"] = self._pc.add_lookup_parameters(
            (WIT_SIZE + 1, (config.INPUT_DIM // 10) * 5),
            init=dy.ConstInitializer(0.))
        self._params["root_emb"] = self._pc.add_lookup_parameters(
            (1, config.bunsetsu_HIDDEN_DIM * 2), init=dy.ConstInitializer(0.))

        self._params["R_bi_b"] = self._pc.add_parameters(
            (2, config.word_HIDDEN_DIM * 2))
        self._params["bias_bi_b"] = self._pc.add_parameters(
            (2),
            init=dy.ConstInitializer(0.))

        if config.cont_aux_separated:
            self._params["cont_MLP"] = self._pc.add_parameters(
                (config.word_HIDDEN_DIM, config.word_HIDDEN_DIM // (2 - config.wemb_lstm) * 2))
            self._params["cont_MLP_bias"] = self._pc.add_parameters(
                (config.word_HIDDEN_DIM), init=dy.ConstInitializer(0.))

            self._params["func_MLP"] = self._pc.add_parameters(
                (config.word_HIDDEN_DIM, config.word_HIDDEN_DIM // (2 - config.wemb_lstm) * 2))
            self._params["func_MLP_bias"] = self._pc.add_parameters(
                (config.word_HIDDEN_DIM),
                init=dy.ConstInitializer(0.))

        if not config.TEST:
            W = utils.orthonormal_initializer(
                config.MLP_HIDDEN_DIM, 2 * config.bunsetsu_HIDDEN_DIM)
            self._params["head_MLP"] = self._pc.parameters_from_numpy(W)
            self._params["head_MLP_bias"] = self._pc.add_parameters((
                config.MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

            self._params["dep_MLP"] = self._pc.parameters_from_numpy(W)
            self._params["dep_MLP_bias"] = self._pc.add_parameters((
                config.MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))
        else:
            self._params["head_MLP"] = self._pc.add_parameters((
                config.MLP_HIDDEN_DIM, config.bunsetsu_HIDDEN_DIM * 2))
            self._params["head_MLP_bias"] = self._pc.add_parameters((
                config.MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

            self._params["dep_MLP"] = self._pc.add_parameters((
                config.MLP_HIDDEN_DIM, config.bunsetsu_HIDDEN_DIM * 2))
            self._params["dep_MLP_bias"] = self._pc.add_parameters((
                config.MLP_HIDDEN_DIM), init=dy.ConstInitializer(0.))

        self._params["R_bunsetsu_biaffine"] = self._pc.add_parameters((
            config.MLP_HIDDEN_DIM + config.biaffine_bias_y, config.MLP_HIDDEN_DIM + config.biaffine_bias_x),
            init=dy.ConstInitializer(0.))
