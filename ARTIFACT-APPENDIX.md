# Artifact Appendix

Paper title: dX-Privacy for Text and the Curse of Dimensionality

Requested Badge(s): **Available**

## Description
This repository is associated with the following paper: 
H. J. Asghar, R. Carpentier, B. Z. H. Zhao, and D. Kaafar, “dX-privacy for text and the curse of dimensionality.”, accepted at Proceeding on Privacy Enhancement Technologies, 2026.

It contains the code to reproduce the results of the Figures and Tables of the paper. In a nutshell:
- We process the FastText, GloVe and Word2Vec word-embeddings vocabularies
- We apply the dx-privacy mechanism from (Feyisetan et al., 2020) on samples from the vocabularies
- We compute distances between elements of the vocabularies to shed light on the problem exposed in the paper
- We propose a fix for the dx-privacy mechanism and compare it with the mechanism from (Yue et al., 2021)

### Security/Privacy Issues and Ethical Concerns

To the best of our knowledge, this artifact does not pose any Security/Privacy Issues nor Ethical Concerns.

## Basic Requirements
The code leverages [cupy](https://cupy.dev/) which requires an Nvidia GPU and a CUDA driver enabled.

## Environment

### Accessibility
To access this code repository, go to [https://github.com/r-carpentier/dx-privacy-curse/tree/main](https://github.com/r-carpentier/dx-privacy-curse/tree/main).

### Set up the environment
1. Install the dependencies with either conda or pip, e.g. `conda env create -n dx-privacy-curse --file environment.yml`
2. Activate the conda environment if needed `conda activate dx-privacy-curse`
3. Make a directory to host the vocabularies and write its absolute path to config.py. In the directory, download the following files:

| Name          | Direct Link                                                  | Homepage                                                     | Extracted files                                              |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GloVe 6B      | [Link](https://nlp.stanford.edu/data/glove.6B.zip)           | [Link](https://nlp.stanford.edu/projects/glove/), file "glove.6B.zip" | glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt and glove.6B.300d.txt |
| GloVe Twitter | [Link](https://nlp.stanford.edu/data/glove.twitter.27B.zip)  | [Link](https://nlp.stanford.edu/projects/glove/), file "glove.twitter.27B.zip" | glove.twitter.27B.25d.txt, glove.twitter.27B.50d.txt, glove.twitter.27B.100d.txt, glove.twitter.27B.200d.txt |
| FastText      | [Link](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec) | [Link](https://fasttext.cc/docs/en/pretrained-vectors.html), under "English: text" | wiki.en.vec                                                  |
| Word2vec      | [Link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) | [Link](https://code.google.com/archive/p/word2vec/), look for "The archive is available here:" | GoogleNews-vectors-negative300.bin                           |

4. Then, run the PreProcessing.ipynb file contained in each of the code folders named fasttext, glove and word2vec.

## Artifact Evaluation
### Main Results and Claims
There are three files in each of folders named fasttext, glove and word2vec:
- NeighborDistances.ipynb generates the results used in Table 2, which involves computing distances between words and their k-th neighbor within the vocabulary. The code in the glove folder contains an additional section to generate the data for Figure 6.
- CloseNeighborsDxFrequencies.ipynb generates the results used in Figure 2, which involves applying $d_x$-privacy to random words and identifying the rank (i.e., the "k" in k-th neighbor) of the word which was chosen as the replacement by the mechanism. The code in the glove and word2vec folders additionaly include the post-processing fix proposed in the paper and used for Figure 8.
- ExponentialMechanism.ipynb generates the data used for Figure 9, where the privacy mechanism applied is the one from (Yue et al., 2021).

Note that the glove folder contains an additional file named textSanitization.ipynb which was used to produce the sanitization example of a small paragraph in Section 2.

## Notes on Reusability
The code of this repository can be reused to experiment with text sanitization on other word-embeddings vocabularies or textual datasets. In particular the notebook glove/textSanitization.ipynb gives an example of sanitizing a paragraph, with the main functions for the privacy mechanism being detailed in utils/dx.py.
