# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys

from fairseq import utils

from torch import distributions


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, retain_dropout_k=None, eos=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.retain_dropout_k = retain_dropout_k

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

        def softmax_entropy(probs):
            return distributions.Categorical(probs=probs).entropy().data

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_probs_v = None
        avg_attn = None

        h_before_avg = None

        model_idx_iter = range(self.retain_dropout_k) if self.retain_dropout_k is not None else range(len(models))

        for model_idx_ in model_idx_iter:
            model_idx = 0 if self.retain_dropout_k is not None else model_idx_
            models[model_idx].eval()
            decoder_out = models[model_idx](**net_input)
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                assert is_single
                sample['target'] = tgt
                curr_prob = models[model_idx].get_normalized_probs(bd, log_probs=len(model_idx_iter) == 1, sample=sample).data  # [B, T, V]
                curr_entr = softmax_entropy(curr_prob)
                if avg_probs_v is None:
                    avg_probs_v = curr_prob
                else:
                    avg_probs_v.add_(curr_prob)
                if h_before_avg is None:
                    h_before_avg = curr_entr
                else:
                    h_before_avg.add_(curr_entr)
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
        if len(model_idx_iter) > 1:
            h_before_avg.div_(len(model_idx_iter))
            avg_probs.div_(len(model_idx_iter))
            avg_probs_v.div_(len(model_idx_iter))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(model_idx_iter))

        h_after_avg = softmax_entropy(avg_probs_v)

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

            unc_data_ij = h_before_avg[i][start_idxs[i]:start_idxs[i] + tgt_len]
            unc_total_ij = h_after_avg[i][start_idxs[i]:start_idxs[i] + tgt_len]
            unc_data_i = unc_data_ij.sum() / tgt_len
            unc_total_i = unc_total_ij.sum() / tgt_len

            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'positional_unc_data': unc_data_ij.tolist(),
                'positional_unc_total': unc_total_ij.tolist(),
                'positional_unc_model': (unc_total_ij - unc_data_ij).tolist(),
                'unc_data': unc_data_i,
                'unc_total': unc_total_i,
                'unc_model': unc_total_i - unc_data_i,
            }])
        return hypos
