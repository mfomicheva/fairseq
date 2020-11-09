#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

import numpy as np

from fairseq import checkpoint_utils, tasks, utils


def prepare_iterator_task_generator_models(args, logger, use_cuda):

    utils.import_user_module(args)

    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set inference dropout
    overrides = eval(args.model_overrides)
    gen_overrides = {
        'retain_dropout': args.retain_dropout,
        'exclude_dropout_modules': args.exclude_dropout_modules,
        'num_stochastic_passes': args.num_stochastic_passes
    }
    if overrides is not None:
        overrides.update(gen_overrides)
    else:
        overrides = gen_overrides

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=overrides,
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    generator = task.build_generator(models, args)

    return itr, task, generator, models
