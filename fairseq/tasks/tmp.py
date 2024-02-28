def make_maskedlm_dataset(dataset, args, source_dictionary):
    dataset = maybe_shorten_dataset(
        dataset,
        split,
        args.shorten_data_split_list,
        args.shorten_method,
        args.tokens_per_sample,
        args.seed,
    )

    # create continuous blocks of tokens
    dataset = TokenBlockDataset(
        dataset,
        dataset.sizes,
        args.tokens_per_sample - 1,  # one less for <s>
        pad=source_dictionary.pad(),
        eos=source_dictionary.eos(),
        break_mode=args.sample_break_mode,
    )
    # logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

    # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
    dataset = PrependTokenDataset(dataset, source_dictionary.bos())

    # create masked input and targets
    mask_whole_words = (
        get_whole_word_mask(args, source_dictionary)
        if args.mask_whole_words
        else None
    )

    src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
        dataset,
        source_dictionary,
        pad_idx=source_dictionary.pad(),
        mask_idx=mask_idx,
        seed=args.seed,
        mask_prob=args.mask_prob,
        leave_unmasked_prob=args.leave_unmasked_prob,
        random_token_prob=args.random_token_prob,
        freq_weighted_replacement=args.freq_weighted_replacement,
        mask_whole_words=mask_whole_words,
        mask_multiple_length=args.mask_multiple_length,
        mask_stdev=args.mask_stdev,
    )

    with data_utils.numpy_seed(args.seed + epoch):
        shuffle = np.random.permutation(len(src_dataset))

    return SortDataset(
        NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=source_dictionary.pad(),
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": RightPadDataset(
                    tgt_dataset,
                    pad_idx=source_dictionary.pad(),
                ),
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        ),
        sort_order=[
            shuffle,
            src_dataset.sizes,
        ],
    )

