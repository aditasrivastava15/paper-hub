# Notes

The paper is [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), by Ilya Sutskever, Oriol Vinyals, and Quoc V. Le (NIPS 2014).

## The picture

Picture two people and one small card.

The first person reads an English sentence and writes the whole meaning on the card. The card is the same size no matter how long the sentence is. The second person never sees the English. They only see the card, and they write the French from it, one word at a time, until they decide the sentence is finished.

A short sentence fits easily. A long sentence has to be squeezed onto the same card. The surprise in this paper is that a careful reader can still squeeze a long sentence well enough to beat a strong older translation system. The trick that makes the squeeze work is reading the English backwards, so the first English words sit next to the first French words.

## The one idea

One network reads the whole English sentence and keeps a single fixed list of numbers. A second network writes the French from that list and stops when it writes an end marker. Reversing the English puts the start of the English next to the start of the French, so learning the link is a short step instead of a long one.

Four names for the pieces of that picture:

- **The card** is the fixed list of numbers. The paper calls it \(v\). It is whatever the reader remembers after the last English word.
- **Two desks** are two separate networks. One reads English. One writes French. They do not share the same network.
- **Reading backwards** means reversing the English only. The French is still written in normal order.
- **The stop sign** is a special word, `<EOS>`, meaning "end of sentence." It lets the French be shorter or longer than the English.

## Words you will see

- **Sequence.** A list of words in order. A sentence is a sequence. The English sequence and the French sequence can have different lengths.
- **Neural network.** A model that learns a job from many examples. Here the job is "given this English sentence, write the French."
- **LSTM.** A kind of network that is good at remembering things it saw earlier in a sequence. LSTM stands for long short-term memory. You can think of it as a reader with a notebook.
- **Layer.** One pass of the network. These authors stack four layers, so the notebook is filled four times, each time a little more abstract.
- **Encoder.** The reader. It walks through the English and leaves the card.
- **Decoder.** The writer. It starts from the card and writes French words.
- **Perplexity.** A score of how surprised the model is by the true French. Lower is better.
- **BLEU.** A score of how close the model's French is to a human translation. Higher is better. It is the number people quote for this paper.

## What problem they are solving

Many neural networks need the input and the answer to be lists of a known, fixed size. Translation is not like that. The English and the French can have different lengths, and you are not told which English word matches which French word.

A plain network can turn one sequence into another when you already know which input step matches which output step. Translation does not give you that match. So this model does not try to line the words up. It reads the whole source into one vector, then writes the target from that vector.

They try it on English to French, using the WMT 2014 data set.

## How a translation is made

```mermaid
flowchart TD
  Flip[Flip the English word order] --> Read[A four-layer LSTM reads it, one word at a time]
  Read --> Card[Keep only the last state: 8000 numbers]
  Card --> Start[Start a second four-layer LSTM from that state]
  Start --> Word[Pick the next French word]
  Word --> More{End token?}
  More -->|no| Word
  More -->|yes| Done[Finished sentence]
```

In symbols, the reader sees words \(x_1, \ldots, x_T\) and keeps only its last memory, a fixed vector \(v\). The writer then produces

\[p(y_1, \ldots, y_{T'} \mid x) = \prod_t p(y_t \mid v, y_1, \ldots, y_{t-1})\]

Read that as: the chance of the whole French sentence is the chance of the first French word, times the chance of the second given the first, and so on. At every step the writer can see the card and the French words it has already written. It cannot look back at the English.

## Why they reverse the English

Suppose the English is "a b c" and the French is "α β γ". If the model reads in normal order, the word *a* is far away in time from *α*. That long gap is hard to learn.

They reverse only the source. The model sees "c b a" and then writes "α β γ". Now *a* sits next to *α*. The average distance between matched words does not change. What changes is the shortest gap. The beginnings of the two sentences can "talk" to each other while the model is learning.

That one change mattered a lot:

- Test perplexity fell from 5.8 to 4.7.
- BLEU for one model rose from 25.9 to 30.6.

They expected only the early French words to get better, and the late ones to get worse. Long sentences got better too. They read that as the network using its memory more carefully.

## What they trained

- **Data.** WMT 2014 English to French. A selected subset of 12 million sentences: 304 million English words and 348 million French words. Same subset as the public baseline they compare against.
- **Vocabulary.** 160,000 English words and 80,000 French words. Any other word is replaced by `UNK`, short for unknown.
- **Size.** Four layers, 1000 cells per layer, and word embeddings of 1000 numbers. The card is the four layers' memories joined together, so it is 8000 numbers. About 384 million parameters. A parameter is one number the model learns. 64 million of those are the connections that carry memory from one word to the next.
- **The French word choice.** At every step the model scores all 80,000 French words and picks among them. They did not use a shortcut to avoid that big list.
- **Training.** The model is trained to make the true French probable given the English. They use plain SGD, no momentum, learning rate 0.7. After epoch 5 the learning rate is halved every half epoch. Training runs 7.5 epochs. A batch has 128 sentences of similar length. If the gradient gets too large (its norm exceeds 5), it is scaled back down. Weights start as small random numbers between −0.08 and 0.08.
- **Time.** About 10 days on 8 GPUs. Four GPUs hold the four layers. Four hold the final word scores.
- **Depth.** A shallow net was worse. Each extra layer cut perplexity by nearly 10%.
- **The LSTM itself.** They use the version described by Graves (2013). You do not need that paper to follow this one. It is the recipe for how the notebook updates itself.

## How they pick a translation at test time

During training, the true French is known. At test time it is not. The model has to choose.

Beam search keeps a few partial translations at once. It extends each one by one word, then throws away all but the best \(B\) partial sentences. \(B\) is the beam width.

- A partial sentence that writes `<EOS>` leaves the beam and becomes a finished candidate.
- A beam of 1 already works. That means "always pick the single best next word."
- A beam of 2 captures most of the gain.
- A beam of 12 is what they use for the headline number.

## The numbers

| Setup | BLEU |
|---|---|
| One model, English not reversed, beam 12 | 26.17 |
| One reversed model, beam 12 | 30.59 |
| Five reversed models, beam 1 | 33.00 |
| Five reversed models, beam 2 | 34.50 |
| Five reversed models, beam 12 | **34.81** |
| The phrase-based baseline they compare to | 33.30 |
| The five models rescoring that baseline's 1000-best list | 36.5 |
| The best WMT 2014 system, scored their way | 37.0 |

A few of those rows need a plain-language note.

- **Five models** means they trained the same network five times with different random starts, then averaged them. That is an ensemble.
- **Rescoring** means the old translation system first writes 1000 possible French sentences. The neural model then helps pick the best one from that list. The neural model is not writing freely there. It is judging drafts the old system already made.
- **Direct decoding** is the other way: the neural model writes the French itself, with beam search. That score is 34.81. It beats the phrase-based baseline of 33.30. It is still below rescoring (36.5) and just below the best system of that year (37.0).
- An oracle that could magically pick the best of those 1000 drafts would score about 45. So the 1000-best list still contains better translations than any of these systems actually pick.
- BLEU is hurt whenever the human reference uses a French word outside the 80,000-word list. The model cannot emit that word, so it cannot match the reference.

Long sentences do not collapse. Their Figure 3 shows little drop for sentences under about 35 words, and only a small drop on the longest ones.

## The card really is a meaning note

They plot the 8000-number cards with PCA, a way to squash many numbers down to a picture you can look at. In that plot, sentences with similar meaning land near each other. Word order matters. "John respects Mary" sits near "Mary is admired by John": the active sentence and the passive sentence are close, which a bag of words would not arrange so cleanly.

## Where this paper sits

Before this, a neural net usually helped an older translation system. It rescored a list the old system had already written, or it added one extra score inside that system.

- Kalchbrenner and Blunsom (2013) already turned a whole sentence into a vector and generated from it. Their reader was a convolutional net, which throws word order away.
- Cho et al. (2014) used an encoder and a decoder much closer to this paper, and mostly plugged it into a statistical system.
- Graves had shown a network that can look back at parts of its input. That look-back is called attention.
- Bahdanau, Cho, and Bengio turn attention into the translator, because encoder–decoder models had been dropping long sentences. They report 28.45 BLEU in this paper's comparison table.

Sutskever, Vinyals, and Le stay with the single card. Their bet is that a deep LSTM, plus English read backwards, can remember a long sentence without looking back, and that the same network can write the translation itself.

## What they still cannot do well

- **Unknown words.** If a word is not on the fixed list, the model cannot really translate it. The lists are larger than in many earlier neural translators, and they are still closed.
- **The model is rough.** The authors call it "relatively unoptimized." A plain word-score over 80,000 words, a simple beam, and no extra French-only or English-only text all leave room to improve.
- **Rescoring still needs the old system.** The direct writer does not, and it scores lower: 34.81 versus 36.5.

## One line to remember

A deep LSTM can read a sentence onto one card and write another sentence from that card. Reversing the source is what makes that memory easy enough to learn.

## What is still confusing

- Why reversing helps the end of a long French sentence, not only the beginning. The late French words are still far from their English partners.
- How much of the 8000-number card is actually used. It is four layers of 1000 cells joined into one state, and the paper does not say which parts matter.
- Why a beam of 2 is almost as good as a beam of 12, even though many French words are still possible at each step.
