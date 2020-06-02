# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from tests.test_sequence_generator import get_dummy_task_and_parser
from fairseq.models.transformer import TransformerModel


class TestInferenceDropout(unittest.TestCase):

    def setUp(self):
        self.task, self.parser = get_dummy_task_and_parser()
        TransformerModel.add_args(self.parser)
        self.args = self.parser.parse_args([])
        self.args.encoder_layers = 2
        self.args.decoder_layers = 1
        self.args.retain_dropout = False
        self.args.exclude_dropout_modules = None

    def test_sets_inference_dropout_to_true(self):
        self.args.retain_dropout = True
        self.transformer_model = TransformerModel.build_model(self.args, self.task)
        assert self.transformer_model.encoder.dropout.apply_during_inference
        assert self.transformer_model.decoder.dropout.apply_during_inference
        for layer in self.transformer_model.encoder.layers:
            assert layer.dropout.apply_during_inference

    def test_inference_dropout_false_by_default(self):
        self.transformer_model = TransformerModel.build_model(self.args, self.task)
        assert not self.transformer_model.encoder.dropout.apply_during_inference
        assert not self.transformer_model.decoder.dropout.apply_during_inference
        for layer in self.transformer_model.encoder.layers:
            assert not layer.dropout.apply_during_inference
        for layer in self.transformer_model.decoder.layers:
            assert not layer.dropout.apply_during_inference

    def test_applies_training_mode(self):
        self.transformer_model = TransformerModel.build_model(self.args, self.task)
        assert self.transformer_model.encoder.dropout.training
        for layer in self.transformer_model.encoder.layers:
            assert layer.dropout.training

        self.transformer_model.eval()
        assert not self.transformer_model.decoder.dropout.training
        for layer in self.transformer_model.encoder.layers:
            assert not layer.dropout.training

    def test_excludes_modules(self):
        self.args.retain_dropout = True
        self.args.exclude_dropout_modules = ['TransformerEncoder', 'TransformerEncoderLayer']
        self.transformer_model = TransformerModel.build_model(self.args, self.task)
        assert not self.transformer_model.encoder.dropout.apply_during_inference
        assert self.transformer_model.decoder.dropout.apply_during_inference
        for layer in self.transformer_model.encoder.layers:
            assert not layer.dropout.apply_during_inference
