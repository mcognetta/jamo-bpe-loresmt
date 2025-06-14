# Intro

This is the implementation for *Jamo-Level Subword Tokenization in Low-Resource Korean Machine Translation* (found [here](https://aclanthology.org/2025.loresmt-1.8/)).

We use `fairseq` and `sentencepiece` as our core implementation details, and only slightly modify `fairseq` (to add jamo-level pre/post-processing for validation and testing; see `fairseq/data/jamo_utils.py` and `fairseq/data/data_utils.py#post_process`). The remaining implementation isn't code, but rather how the data is prepared. We prepare different corpora for each representation type (`syllable`, `byte`, `positional`, and `compat`) and then let `fairseq` work as normal, but with the correct postprocessing to make sure that all the data is validated at the syllable level.

## Installing and Running

This depends on `torch`(we use >=2), `fairseq`, `sentencepiece`, and `sacrebleu`.

To properly install `fairseq`, navigate to the `fairseq` directory and run `pip install --editable ./`.

We include the preprocessed Jeju/Korean corpus from https://aclanthology.org/2020.lrec-1.318/ (and https://github.com/kakaobrain/jejueo, which is licensed under an Apache 2.0 license) to assist you. These are present in `fairseq/examples/kr_translation/orig` and are placed in different subdirectories depending on the representation (note that we use the suffix `_no_end` for `positional` and `compat` jamo to signal that they don't use an explicit empty 종성 character: e.g., `한 -> ㅎㅏㄴ` but `무 -> ㅁㅜ`, not `ㅁㅜ(x)`). 

To add your own corpora, make a similar directory structure, and use `jamo_utils.py` to preprocess the Korean corpus into the correct representation. Non-Korean corpora can be left alone or preprocessed in whatever format you want.

To run an experiment, use the `fairseq/examples/kr_translation/train_<src>_<tgt>_sentencepiece.sh` script (we include `kr<->je` for you). 

- Invocation options:
    - `bash train_kr_je_sentencepiece.sh --experiment-name <EXPERIMENT_NAME> --jamo-type <JAMO_TYPE> --src-bpe-tokens <STOKENS> --tgt-bpe-tokens <TTOKENS> --src-dropout <SDROPOUT> --target-dropout <TDROPOUT> --seed <SEED>`
    - `<EXPERIMENT_NAME>` is any string identifier
    - `<JAMO_TYPE>` is one of `jeju_syllable`, `jeju_syllable_byte`, `jeju_compat_no_end`, `jeju_positional_no_end`
    - `<STOKENS>` and `<TTOKENS>` are the size of the source and target tokens (default 8k each)
    - `<SDROPOUT>` and `<TDROPOUT>` are dropout parameters for the tokenizer they should be between 0.0 and 1.0 (default is 0.0, which means no dropout)
    - `<SEED>` is the random seed for training
    - The output folder name is `<EXPERIMENT_NAME>_VOCAB_<STOKENS>_<TTOKENS>_jamo_type_<JAMO_TYPE>_dropout_<SDROPOUT>_<TDROPOUT>_seed_<SEED>.<LANG_PAIR>` in the experiment outputs directory

An example invocation is:

```
bash train_kr_je_sentencepiece.sh --experiment-name example --jamo-type jeju_compat_no_end --src-bpe-tokens 4000 --tgt-bpe-tokens 4000 --src-dropout 0.1 --tgt-dropout 0.1 --seed 100 --device 0
```

This will make a log file in the `fairseq/experimental_outputs/` directory under a new directory that is built from your run configuration. This directory will hold the training log, the best checkpoint, and, after training, the translated test corpus, and metrics (BLEU, CHRF, etc.).
