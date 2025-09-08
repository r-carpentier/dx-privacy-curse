# Artifact Appendix

Paper title: dX-Privacy for Text and the Curse of Dimensionality


Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**

## Description
This repository is associated with the following paper: 
H. J. Asghar, R. Carpentier, B. Z. H. Zhao, and D. Kaafar, “dX-privacy for text and the curse of dimensionality.”, accepted at Proceeding on Privacy Enhancement Technologies, 2026.

It contains the code to reproduce the results of the Figures and Tables of the paper. In a nutshell:
- We process the FastText, GloVe and Word2Vec word-embeddings vocabularies
- We apply the dx-privacy mechanism from (Feyisetan et al., 2020) on word samples from the vocabularies
- We compute distances between elements of the vocabularies to shed light on the problem exposed in the paper
- We propose a fix for the dx-privacy mechanism and compare it with the mechanism from (Yue et al., 2021)

### Security/Privacy Issues and Ethical Concerns

To the best of our knowledge, this artifact does not pose any Security/Privacy Issues nor Ethical Concerns.

## Basic Requirements

### Hardware Requirements

- NVIDIA GPU card

The machine used to run the experiments has: an Nvidia H100, 1TB of RAM. 
### Software Requirements

- Docker Engine 28.3.3 (should work with any recent version)
- git 2.39.5 (any version should work)
- NVIDIA Driver 575.57.08 or higher (matching GPU card)
- NVIDIA Container Toolkit 1.17.8 (any recent version should work)

The experiments were run on Ubuntu 24.04 (should work with any OS having Docker). The interpreter used is Python 3.11.9. All Python dependencies can be found in the `environment.yml` file.

The following datasets are used for the experiments (the Docker image contains all of them):

| Name          | Direct Link                                                  | Homepage                                                     | Extracted files                                              |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| GloVe 6B (Wiki)     | [Link](https://nlp.stanford.edu/data/glove.6B.zip)           | [Link](https://nlp.stanford.edu/projects/glove/), file "glove.6B.zip" | glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt and glove.6B.300d.txt |
| GloVe Twitter | [Link](https://nlp.stanford.edu/data/glove.twitter.27B.zip)  | [Link](https://nlp.stanford.edu/projects/glove/), file "glove.twitter.27B.zip" | glove.twitter.27B.25d.txt, glove.twitter.27B.50d.txt, glove.twitter.27B.100d.txt, glove.twitter.27B.200d.txt |
| FastText      | [Link](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec) | [Link](https://fasttext.cc/docs/en/pretrained-vectors.html), under "English: text" | wiki.en.vec                                                  |
| Word2vec      | [Link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) | [Link](https://code.google.com/archive/p/word2vec/), look for "The archive is available here:" | GoogleNews-vectors-negative300.bin                           |

### Estimated Time and Storage Consumption

- The overall compute time required to run all the artifact's code to reproduce all experiments is expected to be 24 hours.
- The overall disk space consumed by the artifact is around 60GB (30GB of Docker image + 30GB of processed datasets).

## Environment

### Accessibility
To access this code repository, go to [https://github.com/r-carpentier/dx-privacy-curse/tree/main](https://github.com/r-carpentier/dx-privacy-curse/tree/main).

### Set up the environment
1. Clone the repository and download the Docker image with:
```bash
git clone https://github.com/r-carpentier/dx-privacy-curse.git
cd dx-privacy-curse
docker pull ghcr.io/r-carpentier/dx-privacy-curse:latest
```

2. Launch the Docker container with
```bash
docker run --rm -it -p 8888:8888 -v ${PWD}:/dx-privacy-curse -w /dx-privacy-curse --gpus all --entrypoint bash ghcr.io/r-carpentier/dx-privacy-curse
```

3. Then, start a jupyter notebook within the Docker container:
```bash
conda run --no-capture-output -n dx-privacy jupyter notebook --ip 0.0.0.0 --no-browser
```
4. Click on the link mentionned in the output of the jupyter command to open your web browser. The link is similar to `http://127.0.0.1:8888/tree?token=...`

5. Run the PreProcessing.ipynb notebooks contained within each of the code folders named fasttext, glove and word2vec.

### Testing the Environment
To test the environment after setup, open the notebook glove/textSanitization.ipynb. Execute alls cells one by one. This notebook displays an example of how to sanitize a short text sample. Because the process is randomized, the exact output will be different.

Output example: 'maria gonzalez , a patients at riverside abortions , was diagnosed with depression on november 6 , ersguterjunge . she currently illiterate at 19,300 oak drive , san barbara . pedro whose there bizarre constraint and is undergoing weekly therapy sessions .'

## Artifact Evaluation
### Main Results and Claims
#### Main Result 1: Close Neighbors are not chosen

Our paper claim that the sanitization mechanism presented in (Feyisetan et al, 2020) does not preserve utility as close neighbors of a word are rarely chosen. We report these results in Figure 2.

#### Main Result 2: Probability Masses
We claim that, because of how the distances behave in high dimensions, the probability masses favor the original word being chosen. We report these results in Table 2 and in Figure 6.

#### Main Result 3: Our fix solves the problem we expose
We claim that our fix solves the problem exposed, by selecting more close neighbors. We report these results in Figure 8.

#### Main Result 4: Exponential Mechanism 
We claim that the exponential mechanism from (Yue et al., 2021) does not solve the problem. We report these results in Figure 9.

### Experiments
#### Experiment 1: Close Neighbors Dx Frequencies
- Time: 5 human-minutes +
  - FastText: 20 compute-minutes
  - GloVe: 2 compute-minutes
  - Word2vec: 20 compute-minutes

In each of the folders named "fasttext", "glove" and "word2vec" there is a notebook named CloseNeighborsDxFrequencies.ipynb. It involves applying the $d_x$-privacy mechanism from (Feyisetan et al, 2020) to random words of the vocabularies and identifying the rank (i.e., the "k" in k-th neighbor) of the word which was chosen as the replacement. In the paper, we used the code for Glove (version 6B, 300d) and word2vec vocabularies in the Figure 2.

Note: We provide the code for the fasttext vocabulary even though the results are not shown in the paper.

**How to run**: Run each cell of the notebook one after the other. The plots generated for the GloVe (version 6B, 300d) and word2vec vocabularies should be the same as the Figure 2 of the paper.

#### Experiment 2: Neighbor Distances
- Time: 5 human-minutes +
  - FastText: 20 compute-minutes
  - GloVe: 40 compute-minutes
  - Word2vec: 30 compute-minutes

In each of the folders named "fasttext", "glove" and "word2vec" there is a notebook named NeighborDistances.ipynb. It generates the results used in Table 2, which involves computing distances between words and their k-th neighbor within the vocabulary (Note that this experiment does not involve $d_x$-privacy). The code in the glove folder contains an additional section to generate the data for Figure 6.

**How to run**: Run each cell of the notebook one after the other. For glove/NeighborDistances.ipynb, change the parameter "glove_variant" at the start of the notebook to switch between Glove-Wiki and Glove-Twitter version.
The results printed at the end of each notebook should match Table 2. Additionally, the plot at the end of glove/NeighborDistances.ipynb should match Figure 6.

#### Experiment 3: Post-processing fix
- Time: 5 human-minutes +
  - GloVe: 60 compute-minutes
  - Word2vec: 15 compute-hours

In the folders named "glove" and "word2vec" there is a notebook named Fix.ipynb. It involves applying the $d_x$-privacy mechanism, including our post-processing fix, to random words of the vocabularies and identifying the rank (i.e., the "k" in k-th neighbor) of the word which was chosen as the replacement. In the paper, we used the code for Glove (version 6B, 300d) and word2vec vocabularies in the Figure 8.

**How to run**: Run each cell of the notebook one after the other. The plots generated for the GloVe (version 6B, 300d) and word2vec vocabularies should be the same as the Figure 8 of the paper.

#### Experiment 4: Exponential Mechanism 
- Time: 5 human-minutes +
  - FastText: 80 compute-minutes
  - GloVe: 20 compute-minutes
  - Word2vec: 4 compute-hours

In each of the folders named "fasttext", "glove" and "word2vec" there is a notebook named ExponentialMechanism.ipynb. It involves applying the mechanism of (Yue et al., 2021) to random words of the vocabularies and identifying the rank (i.e., the "k" in k-th neighbor) of the word which was chosen as the replacement. In the paper, we used the code for Glove (version 6B, 300d) and word2vec vocabularies in the Figure 9.

Note: We provide the code for the fasttext vocabulary even though the results are not shown in the paper.

**How to run**: Run each cell of the notebook one after the other. The plots generated for the GloVe (version 6B, 300d) and word2vec vocabularies should be the same as the Figure 9 of the paper.

## Limitations
None, all results of the paper should be reproducible using the artifact. 

## Notes on Reusability
The code of this repository can be reused to experiment with text sanitization on other word-embeddings vocabularies or textual datasets. In particular the notebook glove/textSanitization.ipynb gives an example of sanitizing a paragraph, with the main functions for the sanitization mechanism being detailed in utils/dx.py.

## References in the code:
- (Feyisetan et al, 2020): O. Feyisetan, B. Balle, T. Drake, and T. Diethe, “Privacy- and utility-preserving textual analysis via calibrated multivariate perturbations,” in Proceedings of the 13th international conference on web search and data mining, in WSDM ’20. New York, NY, USA: Association for Computing Machinery, 2020, pp. 178–186. doi: [10.1145/3336191.3371856](https://doi.org/10.1145/3336191.3371856).

- (Qu et al., 2021): C. Qu, W. Kong, L. Yang, M. Zhang, M. Bendersky, and M. Najork, “Natural language understanding with privacy-preserving BERT,” in Proceedings of the 30th ACM international conference on information & knowledge management, in CIKM ’21. New York, NY, USA: Association for Computing Machinery, 2021, pp. 1488–1497. doi: [10.1145/3459637.3482281](https://doi.org/10.1145/3459637.3482281).

- (Yue et al., 2021) X. Yue, M. Du, T. Wang, Y. Li, H. Sun, and S. S. M. Chow, “Differential privacy for text analytics via natural text sanitization,” in Findings of the association for computational linguistics: ACL-IJCNLP 2021, Aug. 2021, pp. 3853–3866. doi: [10.18653/v1/2021.findings-acl.337](https://doi.org/10.18653/v1/2021.findings-acl.337).