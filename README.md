# spanish-ner
Experiments from NER task in Spanish language using CoNLL-2002 and Mexican news datasets

### CoNLL-2002 data set

The CoNLL-2002 data set is in the language Spanish and it has four classes under tagging schema IOB (Inside/Outside/Beginning).

No.| Class | IOB Schema   | Description
---|-------|--------------|:-----------
1  | ORG   | B-ORG, I-ORG | Organizations names
2  | PER   | B-PER, I-PER | People names
3  | LOC   | B-LOC, I-LOC | Location names
4  | MISC  | B-MISC, I-MISC | Miscellaneous

#### Tags distribution using IOB schema
![conll_tags_distribution](./img/conll_tags-dist.png)

Parts of data from the corpus CoNLL-2002.

No.| Partition | Sentences O | Sentences S | Tokens | Tags
---|-----------|------------:|------------:|-------:|---
1  | Test A| 1915 | 2177 | 9646 | 8
2  | Test B| 1517 | 1848 | 9086 | 8
3  | Train| 8323 | 9947 | 26099 | 8
4  | Ensemble| 11755 | 13972 | 31405 | 8

CoNLL-2002 ELmo embeddings [Download](http://148.228.13.30/spanish-ner/data/conll-2002-spanish.full.elmo.tar.gz)

In headings, **Sentences O** are the original sentences length. **Sentences S** were splitted to length 50 (tokens).


#### Sentence histogram

![conll_sentences](./img/conll_sentences.png)

***

### Mx-news data set

The Mx-news data set is in the language Spanish and it has four classes under tagging schema IOBES (Inside/Outside/Beginning/End/Single).

No.| Class | IOB Schema                | Description
---|-------|---------------------------|:----------------------------------------------------------
1  |  PER | B-PER, I-PER, E-PER, S-PER | People names, aliases and abbreviations
2  |  ORG | B-ORG, I-ORG, E-ORG, S-ORG | Organizations, institutions
3  |  DAT | B-DAT, I-DAT, E-DAT, S-DAT | Dates on different formats
4  |  TIT | B-TIT, I-TIT, E-TIT, S-TIT | Title or position of persons
5  |  GPE | B-GPE, I-GPE, E-GPE, S-GPE | Country names, states, cities, municipalities
6  |  PEX | B-PEX, I-PEX, E-PEX, S-PEX | Political party names, aliases and abbreviations
7  |  TIM | B-TIM, I-TIM, E-TIM, S-TIM | Time expresions
8  |  FAC | B-FAC, I-FAC, E-FAC, S-FAC | Facility names
9  |  EVT | B-EVT, I-EVT, E-EVT, S-EVT | Event names
10 | ADD | B-ADD, I-ADD, E-ADD, S-ADD | Addresses expressions, URLs and Twitter users
11 | MNY | B-MNY, I-MNY, E-MNY, ----- | Monetary amounts
12 | DOC | B-DOC, I-DOC, E-DOC, S-DOC | Documents, laws, rules
13 | PRO | B-PRO, I-PRO, E-PRO, S-PRO | Product names, brands, application names
14 | PRC | B-PRC, I-PRC, E-PRC, ----- | Percentage expressions
15 | DEM | B-DEM, -----, E-DEM, S-DEM | Geographical or racial origin of people
16 | AGE | B-AGE, I-AGE, E-AGE, ----- | People age
17 | LOC | B-LOC, I-LOC, E-LOC, S-LOC | Locations about regions, rivers, lakes

#### Tags distribution using IOB schema
![conll_tags_distribution](./img/mx_tags-dist.png)

Parts of data from the corpus Mx-news.

No.| Partition | Sentences O | Sentences S | Tokens | Tags
---|---------|------|------|------|---
1  | Split I | 1295 | 1666 | 7628 | 63
2  | Split II | 1295 | 1677 | 7726 | 63
3  | Split III | 1297 | 1661 | 7664 | 63
4  | Ensemble | 3888 | 5004 | 13273 | 65

Mx-news ELMo embeddings [Download](http://148.228.13.30/spanish-ner/data/mx-news.spanish.full.elmo.tar.gz)

#### Sentence histogram

![mx-news_sentneces](./img/mx_sentences.png)





















