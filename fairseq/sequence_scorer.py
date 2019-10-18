# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import sys

from fairseq import utils
from scipy.stats import entropy
from matplotlib import pyplot


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, summarize_softmax_distribution=None, plot_softmax=None):
        self.pad = tgt_dict.pad()
        self.softmax_batch = softmax_batch or sys.maxsize
        self.summarize_softmax_distribution = summarize_softmax_distribution
        self.plot_softmax = plot_softmax
        self.n_plot = 0
        self.max_plot = 10
        assert self.softmax_batch > 0

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        softmax_distribution = []
        for model in models:
            model.eval()
            decoder_out = model.forward(**net_input)
            attn = decoder_out[1]

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            softmax_distribution = []
            for bd, tgt, is_single in batched:
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data
                bsz, tsz, vb = curr_prob.shape
                if self.summarize_softmax_distribution is not None:
                    softmax_distribution.extend(self._summarize_softmax(curr_prob, bsz, tgt, self.summarize_softmax_distribution, plot_softmax=self.plot_softmax))
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i, start_idxs[i]:]
                _, alignment = avg_attn_i.max(dim=0)
            else:
                avg_attn_i = alignment = None
            res = {
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
            }
            if softmax_distribution:
                softmax_distribution_i = softmax_distribution[i][start_idxs[i]:start_idxs[i] + tgt_len]
                res.update({'positional_statistics': softmax_distribution_i})
            hypos.append([res])
        return hypos

    def _summarize_softmax(self, softmax_probas, bsz, tgt_idxs, method, plot_softmax=None):
        def _step(step_probas):
            if method == 'std':
                return step_probas.numpy().std()
            elif method == 'var':
                return step_probas.numpy().var()
            elif method == 'entr':
                return entropy(np.exp(step_probas))
            else:
                raise ValueError
        output = []
        for i in range(bsz):
            plot = False
            if plot_softmax and not self.n_plot > self.max_plot:
                plot = True
                self.n_plot += 1
            segment_output = []
            for t, idx in enumerate(tgt_idxs[i]):
                if idx == 1:  # ignore padding
                    continue
                proba_copy = softmax_probas[i][t].cpu()
                segment_output.append(_step(proba_copy))
                if plot:
                    self._plot_softmax(proba_copy, self.n_plot, t, self.plot_softmax)
            output.append(segment_output)
        return output

    @staticmethod
    def _plot_softmax(probas, sent_num, word_num, prefix):
        pyplot.hist(probas, bins=50)
        pyplot.savefig('{}.sent{}.word{}.png'.format(prefix, sent_num, word_num))
        pyplot.clf()
