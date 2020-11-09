#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

import torch

from fairseq import options, utils
from fairseq_cli.prepare_generation import prepare_iterator_task_generator_models

from fairseq.tasks.translation_with_lm import TranslationLanguageModelTask
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.tasks.translation import TranslationTask

from copy import deepcopy


def main(args):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger('fairseq_cli.generate')

    use_cuda = torch.cuda.is_available() and not args.cpu

    lm_args = deepcopy(args)
    lm_args.path = args.language_models_paths
    lm_args.task = 'language_modeling'
    lm_iterator, lm_task, lm_generator, lm_models = prepare_iterator_task_generator_models(lm_args, logger, use_cuda)

    tm_args = deepcopy(args)
    tm_args.path = args.translation_models_paths
    tm_args.task = 'translation'
    tm_iterator, tm_task, tm_generator, tm_models = prepare_iterator_task_generator_models(tm_args, logger, use_cuda)

    lm_hs = []
    tm_hs = []
    stats_data = []
    for _, sample_tm in zip(lm_iterator, tm_iterator):
        sample_lm = deepcopy(sample_tm)
        sample_lm['net_input']['src_tokens'] = sample_tm['net_input']['prev_output_tokens']
        sample_lm['net_input']['src_lengths'] = sample_tm['tgt_lengths']
        del sample_lm['net_input']['prev_output_tokens']
        sample_lm = utils.move_to_cuda(sample_lm) if use_cuda else sample_lm
        sample_tm = utils.move_to_cuda(sample_tm) if use_cuda else sample_tm
        lm_hypos = lm_task.inference_step(lm_generator, lm_models, sample_lm)
        tm_hypos = tm_task.inference_step(tm_generator, tm_models, sample_tm)

        for i, sample_id in enumerate(sample_tm['id'].tolist()):
            assert len(lm_hypos[i][0]['pmfs']) == len(tm_hypos[i][0]['pmfs'])
            for tstep in range(len(lm_hypos[i][0]['pmfs'])):
                stats_data_it = {
                    'sentid': sample_id,
                    'tokid': lm_hypos[i][0]['tokens'][tstep].cpu().data.numpy(),
                    'pos': tstep,
                    'mode_lm': torch.argmax(lm_hypos[i][0]['pmfs'][tstep].probs).cpu().data.numpy(),
                    'mode_tm': torch.argmax(tm_hypos[i][0]['pmfs'][tstep].probs).cpu().data.numpy(),
                    'entropy_lm': lm_hypos[i][0]['pmfs'][tstep].entropy().cpu().data.numpy(),
                    'entropy_tm': tm_hypos[i][0]['pmfs'][tstep].entropy().cpu().data.numpy(),
                }
                stats_data.append(stats_data_it)
                lm_hs.append(stats_data_it['entropy_lm'])
                tm_hs.append(stats_data_it['entropy_tm'])
    sns.distplot(lm_hs)
    sns.distplot(tm_hs)
    plt.savefig(os.path.join(args.analysis_dir, 'entropy.png'))
    plt.clf()

    df = pd.DataFrame(stats_data)
    df.to_csv(os.path.join(args.analysis_dir, 'stats_data.tsv'), sep='\t')


def cli_main():
    parser = options.get_analysis_parser()
    TranslationLanguageModelTask.add_args(parser)
    LanguageModelingTask.add_args(parser, ignore_common_arguments=True)
    TranslationTask.add_args(parser, ignore_common_arguments=True)
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
