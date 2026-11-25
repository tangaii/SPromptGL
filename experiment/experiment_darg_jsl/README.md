# DARG + JSL

This experiment tests Jacobian-guided Spectral Learning without Disease Semantic Evidence. The DARG graph model, CAM144 representation, original classifier, frozen Original O spectral basis, 16-dimensional spectral space, and 128-dimensional orthogonal complement are retained. The JSL controller uses whitened spectral tokens, the current classifier margin, and the two Jacobian norms. There are no prompt embeddings, text queries, Gaussian banks, EMA statistics, or DSE outputs in the executable JSL path.

The ten folds start from the read-only E61 epoch-10 full state registered by DPSL. The inherited DARG optimizers and RNG are restored through the DPSL historical loader. New JSL parameters use an isolated CPU RNG and a separate AdamW optimizer. The historical classifier and relation losses remain active. JSL losses are configured in `config.json`.

Historical protocol limitations include 19 intentional training-test overlaps, `val=test` checkpoint selection, pre-CV feature selection and preprocessing, and a transductive patient graph. Results must not be described as independent generalization estimates.

Run `python3.12 -I -B cli.py preflight` before `python3.12 -I -B cli.py train --all`. Training is refused unless the preflight receipt is PASS and matches the frozen implementation hashes.
