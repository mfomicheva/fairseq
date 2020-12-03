# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
from fairseq import utils


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(
        self,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
    ):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.unk = tgt_dict.unk()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )

    def batch_for_softmax(self, dec_out, target):
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

    def gather_target_probs(self, probs, target):
        probs = probs.gather(
            dim=2,
            index=target.unsqueeze(-1),
        )
        return probs

    def compute_scores(self, sample, models, sample_from_model_at_step=1):

        net_input = sample["net_input"]
        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_probs_v = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            batched = self.batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=False, sample=sample
                ).data

                if avg_probs_v is None:
                    avg_probs_v = curr_prob
                else:
                    avg_probs_v.add_(curr_prob)

                if is_single:
                    probs = self.gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = self.gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs.div_(len(models))
        avg_probs.log_()
        if avg_attn is not None:
            avg_attn.div_(len(models))

        return avg_probs, avg_probs_v, avg_attn

    def prepare_hypotheses(self, sample, bsz, avg_probs, avg_probs_v, avg_attn):
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]: start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len

            argmax_probs, indices = torch.max(avg_probs_v[i, :, :], dim=1)
            argmax_probs = argmax_probs[start_idxs[i]: start_idxs[i] + tgt_len]
            argmax_accs = indices == sample["target"][i]
            argmax_accs = argmax_accs.long()
            argmax_accs = argmax_accs[start_idxs[i]: start_idxs[i] + tgt_len]

            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypo_data = {
                "tokens": ref,
                "score": score_i,
                "attention": avg_attn_i,
                "alignment": alignment,
                "positional_scores": avg_probs_i,
                "argmax_probs": argmax_probs,
                "argmax_accs": argmax_accs,
            }
            if "replaced" in sample:
                hypo_data.update({"replaced": sample["replaced"][i][start_idxs[i]: start_idxs[i] + tgt_len]})
            hypos.append([hypo_data])
        return hypos

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        avg_probs, avg_probs_v, avg_attn = self.compute_scores(sample, models)
        bsz = avg_probs.size(0)
        return self.prepare_hypotheses(sample, bsz, avg_probs, avg_probs_v, avg_attn)


class SequenceScorerSampling(SequenceScorer):

    def __init__(self,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
        replacement_probability=0.3,
        max_replacement_steps=100,
    ):
        super().__init__(
            tgt_dict, softmax_batch=softmax_batch, compute_alignment=compute_alignment, eos=eos,
            symbols_to_strip_from_output=symbols_to_strip_from_output,)
        self.replacement_probability = replacement_probability
        self.max_replacement_steps = max_replacement_steps

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        avg_probs, avg_probs_v, avg_attn = self.compute_scores(sample, models)
        device = avg_probs.device
        bsz = avg_probs.size(0)
        tgt_len = sample["target"].ne(self.pad).long().sum(axis=1)

        sample["replaced"] = torch.zeros(sample["target"].size()).long().to(device)
        num_replaced_tokens = torch.zeros(bsz).long().to(device)
        num_tokens_to_replace = (tgt_len * self.replacement_probability).long()

        for step in range(self.max_replacement_steps):
            avg_probs, avg_probs_v, avg_attn = self.compute_scores(sample, models)
            bsz = avg_probs_v.size(0)
            for i in range(bsz):
                if num_replaced_tokens[i] == num_tokens_to_replace[i]:
                    continue
                tgt_idx = torch.randint(tgt_len[i] - 1, [1]).item()
                sampled_token = self.pad
                while sampled_token == self.pad or sampled_token == self.eos or sampled_token == self.unk \
                    or sampled_token == sample["target"][i][tgt_idx]:
                    sampled_token = torch.multinomial(avg_probs_v[i, tgt_idx, :], 1, replacement=True)
                sample["net_input"]["prev_output_tokens"][i][tgt_idx + 1] = sampled_token
                sample["target"][i][tgt_idx] = sampled_token
                sample["replaced"][i][tgt_idx] = 1
                num_replaced_tokens[i] += 1
            if torch.equal(num_tokens_to_replace, num_replaced_tokens):
                break

        return self.prepare_hypotheses(sample, bsz, avg_probs, avg_probs_v, avg_attn)