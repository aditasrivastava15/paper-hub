# Questions

Conceptual questions about **Neural Machine Translation by Jointly Learning to Align and Translate**. Try them, then check the key at the bottom.

**1.** Why do Bahdanau et al. say a basic encoder–decoder struggles on long sentences?

- A. French has more words than English, so the decoder always runs out of steps
- B. The whole source is squashed into one fixed-length vector, which becomes a bottleneck
- C. Softmax cannot handle more than 30 words
- D. Beam search only works on short sentences

**2.** In the baseline RNN encoder–decoder, the same context \(c\) is used at every decoder step. What is the main change in RNNsearch?

- A. \(c\) is replaced by a different context \(c_i\) for each target word
- B. They delete the encoder and only keep the decoder
- C. They use hard alignment from GIZA++ instead of a neural net
- D. They translate one word at a time with no memory

**3.** What does \(\alpha_{ij}\) mean?

- A. The BLEU score of sentence pair \((i, j)\)
- B. How much source annotation \(h_j\) is used when producing target word \(i\)
- C. The hidden size of the encoder
- D. The probability that word \(i\) is unknown

**4.** Why is the alignment called **soft**?

- A. The weights are rounded to 0 or 1 after training
- B. Each target step can look at many source positions at once, with weights that add to 1
- C. Only the diagonal of the alignment matrix is allowed
- D. Humans draw the alignment by hand, then freeze it

**5.** “Jointly learning to align and translate” means:

- A. First run IBM Model 4, then train the neural net
- B. Alignment is a separate latent variable inferred by EM
- C. One training objective (better \(p(y \mid x)\)) updates the encoder, the decoder, and the alignment scorer together
- D. The aligner is trained on German, the translator on French

**6.** Why a **bidirectional** encoder?

- A. So the model can translate English→French and French→English in one net
- B. So each annotation \(h_j\) sees left and right context around source word \(j\)
- C. So decoding can run backwards from the last target word
- D. So unknown words get copied from the source

**7.** \(c_i = \sum_j \alpha_{ij} h_j\) is best described as:

- A. Concatenating every source hidden state
- B. Picking the single source word with the largest \(\alpha_{ij}\)
- C. A weighted average (expected annotation) of the source notes
- D. The last forward hidden state \(h_{T_x}\)

**8.** In the “the man” → “l’ homme” example, why does **soft** alignment help?

- A. French always copies English word order
- B. To choose *le / la / l’*, the model needs to see the noun after “the,” not only “the”
- C. Softmax cannot align “the” to “l’”
- D. Moses already aligned those two words, so the net copies it

**9.** How is this attention different from Graves (2013) handwriting alignment?

- A. Graves is faster because it scores every input at every step; this paper does not
- B. Graves’s attention mostly moves forward; this model scores every source word at every target step, so reordering is allowed
- C. Graves uses a bidirectional RNN; this paper uses only a forward RNN
- D. There is no difference except the language pair

**10.** On WMT 2014 English→French, what did they actually find?

- A. RNNencdec beat RNNsearch on long sentences
- B. RNNsearch beat RNNencdec at all lengths and held up much better as sentences got longer
- C. Moses was useless once neural nets existed
- D. Unknown words were fully solved

**11.** Before this paper, Bengio et al. (2003) style nets in machine translation were mostly used to:

- A. Generate a translation directly from the source
- B. Provide one extra feature inside statistical MT, or re-rank a statistical n-best list
- C. Replace the phrase table with attention
- D. Align words with softmax

**12.** If you freeze \(c_i = \overrightarrow{h}_{T_x}\) for every step, the model:

- A. Becomes a Transformer
- B. Collapses back to the basic RNN encoder–decoder
- C. Can no longer use gated hidden units
- D. Must use hard alignment

**13.** What open problem do the authors still flag?

- A. Softmax is undefined
- B. Attention cannot look at adjectives
- C. Unknown and rare words
- D. English–French is too monotonic to learn

## Answers

1. **B.** One fixed-length vector must hold a sentence of any length. That is the bottleneck. Cho et al. had already shown quality dropping with length.
2. **A.** Equation (4): each target word has its own \(c_i\).
3. **B.** Equation (6): \(\alpha_{ij}\) is the softmax weight on source position \(j\) for target step \(i\).
4. **B.** Weights are in \([0,1]\) and sum to 1. There is no hard single match.
5. **C.** \(a(s_{i-1}, h_j)\) is a neural net. Gradients from the translation loss flow through \(\alpha_{ij}\).
6. **B.** \(h_j = [\overrightarrow{h}_j; \overleftarrow{h}_j]\). Forward sees the left. Backward sees the right.
7. **C.** Equation (5). A soft mix, not a hard pick and not a concatenation of everything with equal weight.
8. **B.** Gender and elision depend on the noun. Hard “the” → “l’” is the wrong story. Looking at both words is the right one.
9. **B.** Graves is monotonic, a mixture of Gaussians that advance. Machine translation needs jumps, for example “European Economic Area” → “zone économique européenne.”
10. **B.** Table 1 and Figure 2: RNNsearch-50 BLEU 26.75 versus RNNencdec-50 at 17.82. Long sentences stay strong. Known-word BLEU is near Moses, which also had extra monolingual data.
11. **B.** Neural nets were a helper score or a re-ranker. This paper wants a standalone neural translator.
12. **B.** The appendix sanity check: one fixed \(c\) is the old model.
13. **C.** They say better handling of unknown words is needed to match strong statistical systems in all settings.
