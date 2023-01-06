from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction
from drain3.persistence_handler import PersistenceHandler
import configparser
import ast
import re
import json
import joblib
import pathlib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from common import constants
import os
import time
from datetime import datetime


class DrainConfigParamParser(TemplateMinerConfig):
    '''
    drain3 사용 시, ini 파일 없이 구성하기 위해 별도의 config 내 param 데이터를 parsing하는 클래스
    '''
    def __init__(self, logger):
        super(DrainConfigParamParser, self).__init__()
        self.parser = configparser.ConfigParser()
        self.logger = logger

    def set_params(self, config_dict):
        self.parser.read_dict(config_dict)

        section_profiling = 'PROFILING'
        self.profiling_enabled = self.parser.getboolean(section_profiling, 'enabled', fallback=self.profiling_enabled)
        self.profiling_report_sec = self.parser.getint(section_profiling, 'report_sec', fallback=self.profiling_report_sec)

        section_snapshot = 'SNAPSHOT'
        self.snapshot_interval_minutes = self.parser.getint(section_snapshot, 'snapshot_interval_minutes', fallback=self.snapshot_interval_minutes)
        self.snapshot_compress_state = self.parser.getboolean(section_snapshot, 'compress_state', fallback=self.snapshot_compress_state)

        section_drain = 'DRAIN'
        drain_extra_delimiters_str = self.parser.get(section_drain, 'extra_delimiters', fallback=str(self.drain_extra_delimiters))
        self.drain_extra_delimiters = ast.literal_eval(drain_extra_delimiters_str)
        self.drain_sim_th = self.parser.getfloat(section_drain, 'sim_th', fallback=self.drain_sim_th)
        self.drain_depth = self.parser.getint(section_drain, 'depth', fallback=self.drain_depth)
        self.drain_max_children = self.parser.getint(section_drain, 'max_children', fallback=self.drain_max_children)
        self.drain_max_clusters = self.parser.getint(section_drain, 'max_clusters', fallback=self.drain_max_clusters)
        self.parametrize_numeric_tokens = self.parser.getboolean(section_drain, 'parametrize_numeric_tokens', fallback=self.parametrize_numeric_tokens)

        section_masking = 'MASKING'
        masking_instructions_str = self.parser.get(section_masking, 'masking', fallback=str(self.masking_instructions))
        self.mask_prefix = self.parser.get(section_masking, 'mask_prefix', fallback=self.mask_prefix)
        self.mask_suffix = self.parser.get(section_masking, 'mask_suffix', fallback=self.mask_suffix)
        self.parameter_extraction_cache_capacity = self.parser.get(section_masking, 'parameter_extraction_cache_capacity', fallback=self.parameter_extraction_cache_capacity)

        masking_instructions = []
        for mi in ast.literal_eval(masking_instructions_str):
            instruction = MaskingInstruction(mi['regex_pattern'], mi['mask_with'])
            masking_instructions.append(instruction)
        self.masking_instructions = masking_instructions


class FilePersistence(PersistenceHandler):
    def __init__(self, file_path):
        self.file_path = file_path

    def save_state(self, state):
        pathlib.Path(self.file_path).write_bytes(state)

    def load_state(self):
        if not os.path.exists(self.file_path):
            return None

        return pathlib.Path(self.file_path).read_bytes()


class Log2Template:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.param_parser = DrainConfigParamParser(logger)
        self.param_parser.set_params(constants.DEFAULT_DRAIN3_CONFIG_DICT)

        template_miner_path = os.path.join(self.config['model_dir'], f"{constants.MODEL_S_LOGSEQ}/template_miner.pkl")
        self.template_miner = TemplateMiner(FilePersistence(template_miner_path), self.param_parser)
        self.mined_period = None#{'from': '', 'to': ''}
        self.n_templates = None

        self.template_dict = {}

        self.template2vec = None

    def regex_filtering(self, lines):
        # datetime
        lines = list(map(lambda x: re.sub(r"\d{4}\-\d{2}\-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", '<DATETIME>', x), lines))
        lines = list(map(lambda x: re.sub(r"\[\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}\]", '<DATETIME> ', x), lines))
        lines = list(map(lambda x: re.sub(r"\s{2,}", ' ', x), lines))
        lines = list(map(lambda x: x.strip(), lines))
        self.logger.info('[Log2Template] regex_filtering - Datetime filtered')

        lines = list(map(lambda x: re.sub(r"(?<=statement )\[.+\]", '<SQL-STATEMENT>', x), lines))
        self.logger.info('[Log2Template] regex_filtering - SQL statement filtered')

        lines = list(map(lambda x: re.sub(r"at .*", '<ERROR-Traceback>', x), lines))
        lines = list(map(lambda x: re.sub(r"Caused by: .*", '<ERROR-Traceback>', x), lines))
        lines = list(map(lambda x: re.sub(r"\.{3} \d+ more", '<ERROR-Traceback>', x), lines))
        lines = list(map(lambda x: re.sub(r"\s{2,}", ' ', x), lines))
        self.logger.info('[Log2Template] regex_filtering - Error Traceback filtered')

        return lines

    def log2tidx(self, log_df, fitting=False):
        if fitting:
            self.fit(log_df)
        return list(map(self.transform, self.regex_filtering(log_df['msg'].values)))

    def fit(self, log_df):
        # template_miner
        if self.load_template_miner(self.config['model_dir']):
            self.logger.info(f"[Log2Template] incremental mining")
            lines_to_mine = self.preprocess_mining_period(log_df)
        else:
            self.mined_period = {'from': self.config['date'][0], 'to': self.config['date'][-1]}
            lines_to_mine = log_df['msg'].values

        self.logger.info(f"[Log2Template] N_lines to mine = {len(lines_to_mine)}")
        if len(lines_to_mine) == 0:
            self.logger.info(f"[Log2Template] No lines to mine. => skip mining")
        else:
            # _ = map(lambda line: self.template_miner.add_log_message(line.strip()), filtered_lines)
            self.logger.info(f"[Log2Template] Mining start")
            time_mining_s = time.time()
            for line in self.regex_filtering(lines_to_mine):
                self.template_miner.add_log_message(line.strip())
            self.logger.info(f"[Log2Template] Mining end (elapsed = {time.time() - time_mining_s:.3f}s)")

        self.n_templates = len(self.template_miner.drain.clusters)
        self.logger.info(f"[Log2Template] Mining finished => n_templates = {self.n_templates}")

        self.logger.debug("="*20 + f" SORTED CLUSTERS " + "="*20)
        sorted_clusters = sorted(self.template_miner.drain.clusters, key=lambda c: c.size, reverse=True)
        for cluster in sorted_clusters:
            self.logger.debug(cluster)
        self.logger.debug(f"=" * 50)

        # template2vec
        self.logger.info(f"[Log2Template] creating Template2Vec model")
        self.make_template2vec()
        self.logger.info(f"[Log2Template] Template2Vec model created")

    def preprocess_mining_period(self, log_df):
        # Filtering already mined data & update mined_period
        dt_from_mined, dt_to_mined = [datetime.strptime(dt_str, '%Y%m%d') for dt_str in [self.mined_period['from'], self.mined_period['to']]]
        dt_from_df, dt_to_df = [datetime.strptime(dt_str, '%Y%m%d') for dt_str in [self.config['date'][0], self.config['date'][-1]]]

        lines_to_mine = []
        if dt_from_mined > dt_from_df:
            # datetime based data filtering
            lines_to_mine.extend(log_df[log_df['_time'] < self.mined_period['from']]['msg'].values)
            self.logger.info(f"[Log2Template] Update mined_period 'from' : {self.mined_period['from']} => {self.config['date'][0]}")
            self.mined_period['from'] = self.config['date'][0]

        if dt_to_mined < dt_to_df:
            # datetime based data filtering
            lines_to_mine.extend(log_df[log_df['_time'] > self.mined_period['to']]['msg'].values)
            self.logger.info(f"[Log2Template] Update mined_period 'to' : {self.mined_period['to']} => {self.config['date'][-1]}")
            self.mined_period['to'] = self.config['date'][-1]

        self.logger.info(f"[Log2Template] mined_period = {self.mined_period['from']} ~ {self.mined_period['to']}")

        return lines_to_mine

    def make_template2vec(self):
        tokenized_templates = [list(filter(lambda x: re.search('[a-zA-Z가-힣]', x) is not None, cluster.get_template().split())) for cluster in self.template_miner.drain.clusters]
        tagged_sents = [TaggedDocument(s, [i]) for i, s in enumerate(tokenized_templates)]
        self.template2vec = Doc2Vec(tagged_sents, vector_size=100, window=2, min_count=1, epochs=500)

    def transform(self, filtered_line):
        matched_template = self.template_miner.match(filtered_line)
        tidx = self.get_most_similar_template(filtered_line) if matched_template is None else matched_template.cluster_id
        return tidx

    def get_most_similar_template(self, line, for_training=False):
        sent_template = self.template_miner.drain.get_content_as_tokens(line)
        sent_filtered = list(filter(lambda x: re.search('[a-zA-Z가-힣]', x) is not None, sent_template))
        sent_vec = self.template2vec.infer_vector(sent_filtered)
        return self.template2vec.docvecs.most_similar(positive=[sent_vec])[0][0] + 1

    def tidx2template(self, pred):
        return [self.template_dict[str(int(c_idx)+1)].get_template() for c_idx in pred]

    def save(self, model_dir):
        self.template_miner.save_state('model_save')
        self.logger.info(f"[Log2Template] template_miner saved (1/4)")

        template2vec_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/template2vec.model")
        self.template2vec.save(template2vec_path)
        self.logger.info(f"[Log2Template] template2vec saved (2/4)")

    def load(self, model_dir):
        try:
            return True if self.load_template_miner(model_dir) and self.load_template2vec(model_dir) else False
        except Exception as e:
            self.logger.info(f"[Log2Template] Error log while Load() : {e}")
            return False

    def load_template_miner(self, model_dir):
        model_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/template_miner.pkl")
        etc_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/etc_info.pkl")
        if os.path.exists(model_path):
            self.template_miner.load_state()
            self.template_dict = dict([(str(c.cluster_id), c) for c in list(self.template_miner.drain.clusters)])
            self.logger.info(f"[Log2Template] template_miner loaded")
            if os.path.exists(etc_path):
                etc_info = joblib.load(etc_path)
                self.mined_period = etc_info['mined_period']
                self.logger.info(f"[Log2Template] mined_period = {self.mined_period['from']} ~ {self.mined_period['to']}")
            return True
        else:
            self.logger.info(f"[Log2Template] template_miner not found")
            return False

    def load_template2vec(self, model_dir):
        model_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/template2vec.model")
        if os.path.exists(model_path):
            self.template2vec = Doc2Vec.load(model_path)
            return True
        else:
            self.logger.info(f"[Log2Template] template2vec_model not found")
            return False