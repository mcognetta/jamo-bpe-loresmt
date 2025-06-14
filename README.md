# Intro

This is the implementation for *Jamo-Level Subword Tokenization in Low-Resource Korean Machine Translation* (found [here](https://aclanthology.org/2025.loresmt-1.8/)).

We use `fairseq` and `sentencepiece` as our core implementation details, and only slightly modify `fairseq` (to add jamo-level pre/post-processing for validation and testing; see `fairseq/data/jamo_utils.py` and `fairseq/data/data_utils.py#post_process`). The remaining implementation isn't code, but rather how the data is prepared. We prepare different corpora for each representation type (`syllable`, `byte`, `positional`, and `compat`) and then let `fairseq` work as normal, but with the correct postprocessing to make sure that all the data is validated at the syllable level.

## Installing and Running

This depends on `torch`(we use >=2), `fairseq`, `sentencepiece`, and `sacrebleu`.

We include the preprocessed Jeju/Korean corpus from https://aclanthology.org/2020.lrec-1.318/ (and https://github.com/kakaobrain/jejueo, which is licensed under an Apache 2.0 license) to assist you. These are present in `fairseq/examples/kr_translation/orig` and are placed in different subdirectories depending on the representation (note that we use the suffix `_no_end` for `positional` and `compat` jamo to signal that they don't use an explicit empty 종성 character: e.g., `한 -> ㅎㅏㄴ` but `무 -> ㅁㅜ`, not `ㅁㅜ(x)`). 

To add your own corpora, make a similar directory structure, and use `jamo_utils.py` to preprocess the Korean corpus into the correct representation. Non-Korean corpora can be left alone or preprocessed in whatever format you want.

To run an experiment, use the `fairseq/examples/kr_translation/train_<src>_<tgt>_sentencepiece.sh` script (we include `kr<->je` for you). A sample invocation is `bash train_kr_je_sentencepiece.sh --experiment-name example --jamo-type jeju_compat_no_end --src-bpe-tokens 4000 --tgt-bpe-tokens 4000 --src-dropout 0.1 --tgt-dropout 0.1 --seed 100 --device 0`. This will make a log file in the `fairseq/experimental_outputs/` directory under a new directory that is built from your run configuration (i.e., it is built from `experiment-name`, `src/tgt-bpe-tokens`, etc.). This directory will hold the training log, the best checkpoint, and, after training, the translated test corpus, and metrics (BLEU, CHRF, etc.).
