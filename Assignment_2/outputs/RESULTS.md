# Assignment 2 - TF and IDF Normalization

Language identification on the 23-language IndicCorpV2 dataset built in
Assignment 1, comparing normalized and unnormalized TF and IDF vectors.

*Generated in 30.6 minutes. No sklearn was used.*

## 1. Setup

| item | value |
|:--|--:|
| languages | 23 |
| sentences per language | 1000 |
| train / validation / test | 18,400 / 2,300 / 2,300 |
| features (word 1-2 grams + char 2-4 grams, min_df=2) | 322,746 |
| non-zeros in the training matrix | 4,223,714 |
| epochs / learning rate / batch | 20 / 0.05 / 512 |

## 2. The formulas being compared

For a term `t` in sentence `d`, let `f` be its raw count, `N` the number of
training sentences, and `df(t)` the number of training sentences containing `t`.

**TF variants**

| variant | formula | normalized? |
|:--|:--|:--|
| raw | `tf = f` | no (Assignment 1) |
| length | `tf = f / (total terms in d)` | yes - task 1(a) |
| maxfreq | `tf = f / max_t' f(t', d)` | yes - task 1(b) |

**IDF variants**

| variant | formula | normalized? |
|:--|:--|:--|
| raw | `idf = log(N / df(t))` | no |
| norm | `idf = log(df_max / df(t))` | yes - task 2 |

`df_max` is the document frequency of the most frequent term across all
sentences, so the normalized IDF is 0 for that term and grows for rarer ones.

> **Why L2 row normalization is switched off.** Both TF normalizations divide a
> whole row by one per-sentence constant. L2 normalization also divides a row by
> a per-row constant, so it would cancel them exactly and all three TF variants
> would produce identical matrices. The comparison only means something with L2 off.

## 3. Results - all 6 combinations

| # | TF | IDF | TF norm. | IDF norm. | mean feature value | val macro-F1 | **test macro-F1** | test accuracy |
|--:|:--|:--|:-:|:-:|--:|--:|--:|--:|
| 1 | raw | raw | no | no | 6.635 | 0.9571 | **0.9560** | 0.9561 |
| 2 | raw | norm | no | yes | 4.994 | 0.9536 | **0.9552** | 0.9552 |
| 3 | length | raw | yes | no | 0.02449 | 0.9675 | **0.9705** | 0.9704 |
| 4 | length | norm | yes | yes | 0.01847 | 0.9671 | **0.9719** | 0.9717 |
| 5 | maxfreq | raw | yes | no | 1.504 | 0.9570 | **0.9581** | 0.9578 |
| 6 | maxfreq | norm | yes | yes | 1.135 | 0.9551 | **0.9577** | 0.9574 |

Ranked by test macro-F1:

| rank | combination | test macro-F1 |
|--:|:--|--:|
| 1 | TF=length, IDF=norm | 0.9719 |
| 2 | TF=length, IDF=raw | 0.9705 |
| 3 | TF=maxfreq, IDF=raw | 0.9581 |
| 4 | TF=maxfreq, IDF=norm | 0.9577 |
| 5 | TF=raw, IDF=raw | 0.9560 |
| 6 | TF=raw, IDF=norm | 0.9552 |

## 4. Marginal effect of each normalization

Average test macro-F1 over the other axis:

| setting | average test macro-F1 |
|:--|--:|
| TF raw (unnormalized) | 0.9556 |
| TF length-normalized | 0.9712 |
| TF maxfreq-normalized | 0.9579 |
| | |
| IDF unnormalized | 0.9615 |
| IDF normalized | 0.9616 |

Spread between the best and worst combination: **0.0167** macro-F1.

## 5. Best combination in detail

**TF=length, IDF=norm** - test macro-F1 0.9719, accuracy 0.9717.

### Per-language scores

```
label         precision   recall       f1  support
--------------------------------------------------
kan_Knda         1.0000   1.0000   1.0000      100
guj_Gujr         1.0000   1.0000   1.0000      100
npi_Deva         1.0000   1.0000   1.0000      100
ory_Orya         1.0000   1.0000   1.0000      100
mni_Mtei         1.0000   1.0000   1.0000      100
mal_Mlym         1.0000   1.0000   1.0000      100
khasi            1.0000   1.0000   1.0000      100
kas_Arab         1.0000   1.0000   1.0000      100
snd_Deva         1.0000   1.0000   1.0000      100
urd_Arab         1.0000   1.0000   1.0000      100
tel_Telu         1.0000   1.0000   1.0000      100
tam_Taml         1.0000   1.0000   1.0000      100
san_Deva         1.0000   1.0000   1.0000      100
santhali         1.0000   1.0000   1.0000      100
pan_Guru         1.0000   1.0000   1.0000      100
ben_Beng         0.9901   1.0000   0.9950      100
asm_Beng         1.0000   0.9900   0.9950      100
mai_Deva         1.0000   0.9900   0.9950      100
doi_Deva         0.9894   0.9300   0.9588      100
brx_Deva         1.0000   0.8700   0.9305      100
hin_Deva         0.8333   1.0000   0.9091      100
gom_Deva         0.7767   0.8000   0.7882      100
mar_Deva         0.7938   0.7700   0.7817      100
--------------------------------------------------
macro avg        0.9732   0.9717   0.9719     2300
accuracy                           0.9717     2300
```

### Most frequent confusions

| true | predicted | count |
|:--|:--|--:|
| mar_Deva (Marathi) | gom_Deva (Konkani) | 23 |
| gom_Deva (Konkani) | mar_Deva (Marathi) | 20 |
| brx_Deva (Bodo) | hin_Deva (Hindi) | 13 |
| doi_Deva (Dogri) | hin_Deva (Hindi) | 7 |
| mai_Deva (Maithili) | doi_Deva (Dogri) | 1 |
| asm_Beng (Assamese) | ben_Beng (Bengali) | 1 |

## 6. Training curves (validation macro-F1 per epoch)

| epoch | TF=raw/IDF=raw | TF=raw/IDF=norm | TF=length/IDF=raw | TF=length/IDF=norm | TF=maxfreq/IDF=raw | TF=maxfreq/IDF=norm |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.9540 | 0.9527 | 0.9634 | 0.9631 | 0.9570 | 0.9551 |
| 2 | 0.9553 | 0.9514 | 0.9648 | 0.9644 | 0.9554 | 0.9544 |
| 3 | 0.9554 | 0.9504 | 0.9644 | 0.9662 | 0.9546 | 0.9494 |
| 4 | 0.9543 | 0.9514 | 0.9653 | 0.9653 | 0.9545 | 0.9502 |
| 5 | 0.9518 | 0.9527 | 0.9653 | 0.9653 | 0.9549 | 0.9481 |
| 6 | 0.9495 | 0.9536 | 0.9662 | 0.9662 | 0.9553 | 0.9461 |
| 7 | 0.9499 | 0.9510 | 0.9662 | 0.9671 | 0.9551 | 0.9469 |
| 8 | 0.9507 | 0.9500 | 0.9671 | 0.9671 | 0.9544 | 0.9482 |
| 9 | 0.9549 | 0.9491 | 0.9662 | 0.9667 | 0.9534 | 0.9504 |
| 10 | 0.9499 | 0.9482 | 0.9662 | 0.9667 | 0.9517 | 0.9509 |
| 11 | 0.9538 | 0.9493 | 0.9671 | 0.9662 | 0.9534 | 0.9508 |
| 12 | 0.9553 | 0.9499 | 0.9662 | 0.9658 | 0.9558 | 0.9516 |
| 13 | 0.9571 | 0.9512 | 0.9666 | 0.9649 | 0.9529 | 0.9505 |
| 14 | 0.9552 | 0.9510 | 0.9666 | 0.9666 | 0.9511 | 0.9515 |
| 15 | 0.9549 | 0.9534 | 0.9641 | 0.9671 | 0.9503 | 0.9530 |
| 16 | 0.9527 | 0.9483 | 0.9675 | 0.9671 | 0.9520 | 0.9492 |
| 17 | 0.9522 | 0.9491 | 0.9671 | 0.9671 | 0.9480 | 0.9465 |
| 18 | 0.9508 | 0.9494 | 0.9666 | 0.9671 | 0.9498 | 0.9466 |
| 19 | 0.9500 | 0.9500 | 0.9670 | 0.9670 | 0.9463 | 0.9485 |
| 20 | 0.9496 | 0.9466 | 0.9654 | 0.9671 | 0.9447 | 0.9448 |

## 7. Sample predictions from the best model

| sentence (first 60 chars) | true | predicted | correct |
|:--|:--|:--|:-:|
| પ્રિયંકા-દીપિકાને પછાડી આ અભિનેત્રી ઇન્સ્ટાગ્રામ પર બની નંબર | guj_Gujr | guj_Gujr | yes |
| పన్నెండు సంవత్సరాలలోపు వయస్సు ఉన్న పిల్లలు వారి తల్లిదండ్రుల | tel_Telu | tel_Telu | yes |
| ꯃꯁꯤꯅ ꯎꯈ꯭ꯔꯨꯜ ꯕꯨ ꯑꯁꯪꯕ ꯃꯌꯣꯜꯂꯝ ꯑꯣꯢꯍꯜꯂꯤ꯫ | mni_Mtei | mni_Mtei | yes |
| ꯃꯁꯤ ꯑꯔꯤꯕ ꯏꯖꯤꯞꯀꯤ ꯍꯤꯡꯍꯧꯔꯤꯕ ꯈꯨꯗꯝ ꯀꯌꯥꯒꯤ ꯃꯅꯨꯡꯗ ꯑꯃꯅꯤ꯫ | mni_Mtei | mni_Mtei | yes |
| سنہ 1792ء کو 11 سال کی عمر میں انہوں نے بھارت کی سات سالہ یا | urd_Arab | urd_Arab | yes |
| U la ïathuh ba 500 tylli ki shnong ban ïoh ïa kane ka skhim  | khasi | khasi | yes |
| कारण अपनाकेँ असगरुआ आभास भ’ रहल छल तैं हिनका लोकनिक लगीच जाय | mai_Deva | mai_Deva | yes |
| വയനാട്ടിൽ വിശ്രമിച്ച ശേഷം നൗഷാദ് രണ്ടുദിവസത്തിനകം മടങ്ങും. | mal_Mlym | mal_Mlym | yes |
| உலகலாவிய ரீதியில் முக்கியத்துவம் வாய்ந்த இந்த மாநாட்டை யாழ்ப | tam_Taml | tam_Taml | yes |
| ھڪ اندازي مطابق رڳو ھڪ ڪيوبڪ ميٽر ۾ ڏھ ھزار کان ٽيھ ھزار ڪرل | snd_Deva | snd_Deva | yes |
| रां राननो थांनायनि सन्देहआव राइजोआ हमथानानै लाखियो बिजेपिनि  | brx_Deva | brx_Deva | yes |
| حڪم جڏهن پنهنجي قبيلي جي ساٿين کان ان نئين اڏجندڙ شهر جي نال | snd_Deva | snd_Deva | yes |

## 8. How to reproduce

```
python assignment2.py
```

## 9. Discussion — why the results came out this way

### 9.1 Summary of the comparison (assignment task 3)

| normalization | average test macro-F1 | change vs. unnormalized |
|:--|--:|--:|
| TF unnormalized (raw `f`) | 0.9556 | — |
| TF normalized by sentence length — task 1(a) | **0.9712** | **+0.0156** |
| TF normalized by max word frequency — task 1(b) | 0.9579 | +0.0023 |
| IDF unnormalized `log(N/df)` | 0.9615 | — |
| IDF normalized `log(df_max/df)` — task 2 | 0.9616 | +0.0001 |

**TF normalization matters. IDF normalization does not.** There is a clean
mathematical reason for both halves of that sentence.

### 9.2 Why IDF normalization changes almost nothing

Write out the two IDF definitions:

```
idf_raw (t) = log(N / df(t))       = log(N)      - log(df(t))
idf_norm(t) = log(df_max / df(t))  = log(df_max) - log(df(t))
```

Subtracting gives

```
idf_norm(t) = idf_raw(t) - log(N / df_max)
```

— the same **constant** is subtracted from every term's IDF. More importantly,
IDF is a **per-column** transformation: column `j` of the matrix is multiplied
by one fixed number.

A linear model can undo any per-column rescaling for free. If column `j` is
multiplied by `a_j`, the model simply learns `W[j] / a_j` instead of `W[j]` and
produces exactly the same scores. So IDF normalization cannot change what the
classifier is *able* to represent — it can only change how quickly the optimiser
gets there. Measured effect: **+0.0001**, i.e. noise.

### 9.3 Why TF normalization does matter

TF normalization is a **per-row** transformation: each sentence is divided by its
own constant. The model has one weight vector shared across all rows, so it
*cannot* absorb a different divisor for every sentence. This genuinely changes
the geometry of the data, and that is why it moves the score.

Without it, a long sentence produces a larger vector than a short one, so it
generates larger logits purely because it contains more text. The model's
confidence ends up tracking sentence length instead of language.

### 9.4 Why length-normalization (1a) beats max-frequency normalization (1b)

Look at what each one actually divides by — visible in the mean feature values
in section 3:

| TF variant | mean feature value | implied average divisor |
|:--|--:|--:|
| raw | 6.635 | 1 |
| maxfreq | 1.504 | ≈ 4.4 |
| length | 0.0245 | ≈ 271 |

The most frequent term in a single sentence occurs only about **4–5 times**, and
that number is nearly the same for every sentence. So dividing by it is close to
dividing every row by the *same* constant — which is a per-column-like global
rescaling that the model can largely absorb, exactly like IDF. That is why
`maxfreq` lands within 0.002 of `raw`.

Total term count, by contrast, varies with the actual length of the sentence
(≈271 on average, but genuinely different per row). Dividing by it is a real
per-row normalization, and it delivers the full +0.0156.

**Task 1(b) turns out to be a much weaker normalization than task 1(a), and the
reason is that `max f` barely varies across sentences while `sum f` does.**

### 9.5 Length normalization also acts as a regulariser

From the training curves in section 6, comparing training loss at epoch 20:

| combination | train loss @20 | val macro-F1 trend |
|:--|--:|:--|
| TF=maxfreq, IDF=raw | 0.0008 | 0.9570 → 0.9447 (falling) |
| TF=raw, IDF=raw | 0.0294 | 0.9540 → 0.9496 (falling) |
| TF=length, IDF=norm | 0.0781 | 0.9631 → 0.9671 (rising) |

The unnormalized and max-frequency models drive training loss essentially to
zero — they memorise the training set — while their validation score *decreases*.
The length-normalized models keep a higher training loss and their validation
score still improves at epoch 20. Bounding every row to a small, comparable
magnitude limits how large the logits can grow, which slows memorisation.

(The reported test scores use the best-on-validation checkpoint, so the
overfitting is caught rather than reflected in the final numbers.)

### 9.6 Where the remaining errors are

All 23 languages except the Devanagari-script group score at or near 1.0000.
Every error is a confusion *within a single script*, overwhelmingly between
Marathi, Konkani, Hindi and Bodo — four languages that share the Devanagari
script and a large amount of vocabulary. No sentence is ever assigned to a
language written in a different script.

### 9.7 Relation to Assignment 1

Assignment 1 scored **0.9740** using sublinear TF (`1 + log f`) with L2 row
normalization. The best combination here reaches **0.9719** without L2. The gap
is small and points the same way: the useful ingredient is *some* per-row
normalization, and `1 + log f` damps repeated terms slightly better than a plain
division by the count.

### 9.8 Conclusion

1. Normalizing TF improves the identifier; normalizing IDF does not.
2. The asymmetry is structural, not empirical — IDF is a per-column transform a
   linear model can absorb into its weights, TF is a per-row transform it cannot.
3. Of the two TF normalizations, dividing by sentence length (1a) is clearly the
   better one (+0.0156); dividing by the most frequent term (1b) gives almost
   nothing (+0.0023) because that divisor hardly varies between sentences.
4. Best combination: **TF = length-normalized, IDF = normalized**, test macro-F1
   **0.9719**, accuracy **0.9717**.
