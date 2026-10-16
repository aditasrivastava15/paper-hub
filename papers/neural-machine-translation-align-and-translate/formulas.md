# Formulas

Every main formula from **Neural Machine Translation by Jointly Learning to Align and Translate**, said slowly. Where the paper does not write a derivation, the step is marked as implied algebra. The 3-word example at the end is a toy, not a paper result.

## 0. What the machine is trying to do

\[
\hat{y} = \arg\max_y \, p(y \mid x)
\]

- \(x\) = English sentence
- \(y\) = a possible French sentence
- \(p(y \mid x)\) = how likely this French is, given that English
- \(\arg\max\) = pick the French with the biggest chance
- \(\hat{y}\) = that pick

Training makes \(p(y \mid x)\) big on real pairs. The paper does not write a loss line. At test time they use beam search: keep a few good drafts, not every possible sentence.

## 1. Encoder step — equation (1)

\[
h_t = f(x_t, h_{t-1})
\]

- \(x_t\) = English word \(t\), as numbers
- \(h_{t-1}\) = memory after the previous word
- \(f\) = mix this word with old memory
- \(h_t\) = new memory

Read one word. Update the backpack. Repeat.

Then the old model squishes the whole backpack history into one note:

\[
c = q(\{h_1,\ldots,h_{T_x}\})
\]

The usual choice, from Sutskever and colleagues, is \(q\) = “keep only the last backpack,” so \(c = h_{T_x}\). Short or long English, the note is the same size. That is the jam.

## 2. A sentence chance is a chain — equation (2)

\[
p(y) = \prod_{t=1}^{T_y} p(y_t \mid y_1,\ldots,y_{t-1}, c)
\]

The chance of the whole French sentence is the chance of word 1, times the chance of word 2 given word 1, times the chance of word 3 given the first two, and so on.

Why multiply? The chain rule: \(p(a,b,c) = p(a)\,p(b \mid a)\,p(c \mid a,b)\). The paper writes the product and does not expand this. That expansion is implied algebra.

**Old decoder — equation (3)**

\[
p(y_t \mid \ldots) = g(y_{t-1}, s_t, c)
\]

- \(s_t\) = the writer’s memory at step \(t\)
- \(c\) = the same English note every time
- \(g\) = turn that into chances over the next French word

If \(c\) forgot something, every later word is in trouble.

## 3. New decoder — equation (4)

\[
p(y_i \mid y_1,\ldots,y_{i-1}, x) = g(y_{i-1}, s_i, c_i)
\]

\[
s_i = f(s_{i-1}, y_{i-1}, c_i)
\]

Same shape as before, except \(c\) became \(c_i\). Each French word gets its own English mix.

## 4. How \(c_i\) is built — equation (5)

\[
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
\]

- \(h_j\) = the note sitting near English word \(j\)
- \(\alpha_{ij}\) = how big a share that note gets for French word \(i\), a number from 0 to 1
- the sum adds those weighted notes

This is a weighted average. If \(\alpha_{i3} = 1\) and the rest are 0, then \(c_i = h_3\). If two words share 0.5 each, \(c_i\) is halfway between those two notes.

The paper also describes this as an expected note: treat \(\alpha_{i\cdot}\) as “the chance I am looking at word \(j\),” then \(c_i = \mathbb{E}[h_j]\). Same formula, probability language.

## 5. Where the shares come from — equation (6)

First a raw score. How well does “what I still need to say” match “this English spot”?

\[
e_{ij} = a(s_{i-1}, h_j)
\]

- \(s_{i-1}\) = the writer’s memory before writing French word \(i\)
- \(a\) = a small scoring network, the alignment model

Then turn scores into shares that add to 1:

\[
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
\]

\(\exp\) makes every share positive. Dividing by the sum makes them add to 1. That is softmax. After this, \(\alpha_{i\cdot}\) is a pile of peeking weights.

What \(a\) actually is, from the appendix (not numbered in the main text):

\[
a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j)
\]

1. Stretch the writer memory: \(W_a s_{i-1}\)
2. Stretch the English note: \(U_a h_j\)
3. Add them
4. Squash with \(\tanh\), which keeps numbers in a bounded range
5. \(v_a^\top\) turns that list into one number, the energy \(e_{ij}\)

\(U_a h_j\) does not depend on which French word is being written, so it can be computed once per English sentence.

## 6. Two-way reader — equation (7)

A normal RNN only sees words to the left. They run two:

- Forward: \(\overrightarrow{h}_j\) = left context plus this word
- Backward: \(\overleftarrow{h}_j\) = right context plus this word

\[
h_j = \begin{bmatrix} \overrightarrow{h}_j \\ \overleftarrow{h}_j \end{bmatrix}
\]

Glue the two lists into one longer list. Now each \(h_j\) knows both sides of word \(j\). That is the note attention looks at.

## 7. Writer memory with gates

From the appendix:

\[
z_i = \sigma(W_z e(y_{i-1}) + U_z s_{i-1} + C_z c_i)
\]

\[
r_i = \sigma(W_r e(y_{i-1}) + U_r s_{i-1} + C_r c_i)
\]

\[
\tilde{s}_i = \tanh\bigl(W e(y_{i-1}) + U[r_i \circ s_{i-1}] + C c_i\bigr)
\]

\[
s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i
\]

- \(e(y_{i-1})\) = the last French word, as numbers
- \(\sigma\) = squash to (0, 1), like a dimmer switch
- \(\circ\) = multiply slot by slot
- \(z_i\) = update gate. Near 0, keep old memory. Near 1, write the new candidate.
- \(r_i\) = reset gate. Near 0, ignore old memory when inventing the candidate.
- \(\tilde{s}_i\) = the candidate new memory
- The last line blends old and new, controlled by the dimmer.

The encoder uses the same gates but drops \(c_i\). The reader is only reading. It is not peeking.

Start of writing:

\[
s_0 = \tanh(W_s \overleftarrow{h}_1)
\]

\(\overleftarrow{h}_1\) has already walked the English from the end to the start, so it has seen the whole sentence. That wakes the writer up. Each later \(c_i\) is a local peek.

Sanity check the paper gives: if you freeze \(c_i = \overrightarrow{h}_{T_x}\) (always the last forward backpack), the model falls back to the old encoder–decoder.

## 8. Next-word chances

From the appendix:

\[
\tilde{t}_i = U_o s_i + V_o E y_{i-1} + C_o c_i
\]

Mix the writer memory, the last word, and this peek.

Maxout: each output slot is the bigger of a pair of numbers in \(\tilde{t}_i\). That is a simple “pick the stronger signal” hidden layer.

Then

\[
p(y_i \mid \ldots) \propto \exp(y_i^\top W_o t_i)
\]

and a softmax over the 30,000-word list so the chances add to 1. Here \(y_i\) is one-hot: a 1 in the slot of that word.

## 9. Toy arithmetic

Made-up numbers, not from the paper. English has 3 notes, \(h_1, h_2, h_3\). The writer is about to write word \(i\). Raw scores:

\[
e_{i1}=2,\quad e_{i2}=1,\quad e_{i3}=0
\]

\[
\exp(2)\approx 7.39,\quad \exp(1)\approx 2.72,\quad \exp(0)=1
\]

The sum is about 11.11. Shares:

\[
\alpha \approx (0.67,\ 0.24,\ 0.09)
\]

So \(c_i \approx 0.67 h_1 + 0.24 h_2 + 0.09 h_3\). The flashlight is mostly on word 1, a little on word 2, and almost off on word 3. That mix goes into the gates, then into the next-word softmax.

## 10. How the numbers get learned

One grade: “was the real French likely?” Because \(c_i\) is made of addition, multiplication, and softmax, the grade can flow back through the shares into \(a\), the reader, and the writer. That is “jointly learning to align and translate.” There is no extra alignment formula.

**Table 1, results rather than a derivation.** RNNsearch-50 BLEU is 26.75. The old model is 17.82, on all sentences. On sentences with no unknown words, 34.16 versus Moses at 35.63. Attention wins, and long sentences stop collapsing.
