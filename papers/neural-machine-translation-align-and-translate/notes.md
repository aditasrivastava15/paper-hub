# Notes

These notes are only for **Neural Machine Translation by Jointly Learning to Align and Translate**.

## The one idea that matters

Older neural machine translation squashes a whole source sentence into one fixed-length vector. That bottleneck fails on long sentences. This paper lets the decoder look at a different mix of source words for each target word. That look is attention, also called soft alignment, and it is trained together with translation.

## Picture box

- **Sticky note** = the old fixed-length vector
- **Peeking / flashlight** = attention (look more at some words, less at others)
- **Two-way walker** = the bidirectional encoder (reads left-to-right and right-to-left)
- **One team, one grade** = joint training (alignment and translation learn together)

The story in one breath: old machines smash a whole English sentence onto one sticky note, then write French from that note. A long sentence does not fit, so details fall off. This paper keeps a row of notes, one near each English word, and lets the writer peek at the bits that matter for the next French word.

## Paper map

1. **Neural machine translation.** One network, one goal: maximize \(p(y \mid x)\), not a pipeline of separately tuned statistical pieces.
2. **Encoder–decoder baseline.** The encoder RNN ends at one vector \(c = h_{T_x}\). The decoder generates every target word from that same \(c\).
3. **Attention.** Each target step \(i\) gets its own \(c_i = \sum_j \alpha_{ij} h_j\). \(\alpha_{ij}\) is a softmax over how well decoder state \(s_{i-1}\) matches source annotation \(h_j\).
4. **Bidirectional encoder.** Each \(h_j\) concatenates a forward and a backward hidden state, so it sees left and right context around word \(j\).
5. **Joint training.** Alignment is a small neural net, not a hidden variable inferred on the side. The translation loss backpropagates through the attention weights.
6. **Results (WMT 2014 English→French).** RNNsearch-50 BLEU **26.75** versus the baseline **17.82** on all sentences. On sentences with only known words, **34.16**, near Moses at **35.63**. Quality holds on long sentences. The baseline collapses.
7. **Open problem they name.** Unknown and rare words.

## Section 5, the attention part

When the machine writes the next French word, it does not use one sticky note for the whole English sentence. It makes a fresh little note for this word, by peeking more at some English words and less at others.

1. The reader first makes a row of notes, one near each English word. Each note knows that word plus some of the words around it, because the two-way walker has gone left-to-right and right-to-left.
2. Before writing French word \(i\), the writer looks at what it has already said (\(s_{i-1}\)) and scores every English note: “How useful are you right now?”
3. Those scores become shares that add up to 100% (softmax). The share for English word \(j\) is \(\alpha_{ij}\).
4. The fresh note is a mix: \(c_i = \sum_j \alpha_{ij} h_j\). A word with a big share counts more in the mix.
5. The writer uses that mix, plus the last French word, to write the next word.

Why this is the heart of the paper:

- The old machine used one note \(c\) for every French word. Here each French word gets its own note \(c_i\).
- Peeking is soft. It can look at “the” and “man” at once, which helps pick “l’” versus “le.”
- Peeking is trained with translation. Nobody draws the matches by hand. The same “did I write good French?” grade teaches the peeking too.

## The history paragraph, in easy language

Neural nets were already used in translation, but only as a helper, not as the whole translator.

Bengio et al. (2003) built a neural language model. It answers “what word comes next?” A neural net reads a fixed number of previous words (a short window, not the whole sentence) and outputs chances for the next word. Words became lists of numbers, so similar words could share meaning instead of living in separate count bins.

After that, people put neural nets inside the old translation factory in two common ways:

1. **One extra feature.** The statistical system still scores a translation with many knobs. The net adds one more score. The old system still runs the show.
2. **Re-rank a list.** The old system writes, say, the 100 best French drafts. A neural net re-orders that list. The net never writes the sentence from scratch.

Bahdanau, Cho, and Bengio want the opposite: one network that reads the source and generates the translation.

## What I still don't understand

-
