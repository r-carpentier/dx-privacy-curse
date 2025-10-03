import numpy as np
from secrets import randbits
import cupy as cp
from cupyx.scipy.spatial import distance
from utils.tools import best_uint_type, rank_neighbors


def sample_noise_vectors(
    dimension: int,
    shape1: int,
    shape2: int,
    epsilon: float,
    dtype: np.dtype = np.float32,
    seed: int = -1,
) -> np.ndarray:
    """Sample a noise vector of shape (shape1, shape2, dimension) as a numpy array."""
    if seed == -1:
        rng = np.random.default_rng(randbits(128))
    else:
        rng = np.random.default_rng(seed)

    # Generate an array of noise vectors sampled from the multivariate normal distribution
    # https://numpy.org/doc/1.26/reference/random/generated/numpy.random.Generator.multivariate_normal.html
    # mean: Mean of the N-dimensional distribution. Chosen as the origin following (Feyisetan et al., 2020, Sec. 2.6)
    # cov: The covariance matrix of the distribution. Chosen as the identity matrix following (Feyisetan et al., 2020, Sec. 2.6)
    # size: Shape of the ouput. Set to the number of noise vectors we need.
    # check_valid: raise error if the covariance matrix is not positive semidefinite.
    # tol: Tolerance when checking the singular values in covariance matrix. Unset, default 1e-8.
    # method: Method for computing an intermediate matrix. Only impacts performances. "cholesky" is the fastest.
    origin = np.full(dimension, 0)
    cov_matrix = np.identity(dimension)
    noises = rng.multivariate_normal(
        mean=origin,
        cov=cov_matrix,
        size=(shape1, shape2),
        check_valid="raise",
        method="cholesky",
    ).astype(dtype)

    # Normalize each noise by dividing each vector by its norm.
    # https://numpy.org/doc/1.26/reference/generated/numpy.linalg.norm.html
    # x: The vector to be normalized
    # ord: Order of the norm. None uses the Frobenius matrix norm, which, applied on vectors, results in the Euclidean/L2 norm.
    # axis: Specifies the axis of x along which to compute the vector norms. We want each single vector to be normalized thus choosing the last axis i.e. -1
    # keepdims: The normed axis are left in the result as dimensions with size one.
    noises /= np.linalg.norm(noises, ord=None, axis=-1, keepdims=True).astype(dtype)

    # Generate an array of magnitude scalars.
    # https://numpy.org/doc/1.26/reference/random/generated/numpy.random.Generator.gamma.html
    # shape: Shape of the gamma distribution, often noted "k". Set to the embeddings' dimension following (Feyisetan et al., 2020, Sec. 2.6) and (Qu et al., 2021, Sec. 3.2.3)
    # scale: Scale of the distribution, often noted theta. Set to 1/epsilon following (Feyisetan et al., 2020, Sec. 2.6) and (Qu et al., 2021, Sec. 3.2.3)
    # size: Shape of the ouput. Set to the number of magnitude scalars we need.
    magnitudes = rng.gamma(
        shape=dimension, scale=1.0 / epsilon, size=(shape1, shape2)
    ).astype(dtype)

    noises *= magnitudes[..., np.newaxis]

    return noises


def noisy_embeddings_to_ids(
    embeddings: np.ndarray,
    vocabulary: np.ndarray,
    distance_metric: str = "euclidean",
    chunk_size: int = -1,
) -> np.ndarray:
    """Performs a nearest neighbor search of the embeddings against the vocabulary.
    Returns a numpy array of shape (embeddings.shape[0], vocabulary.shape[0]) where
    array[i][j] contains the distance between the i-th embedding and the j-th vocabulary
    element.
    We leverage cupy and performs the computation chunk-by-chunk to avoid VRAM overload.
    """
    number_of_words = embeddings.shape[0]

    if chunk_size == -1:
        # Calculate the ideal chunk_size such that storing the distances before the argmin
        # uses at most 2GB of VRAM. Note that cupyx.scipy.spatial.distance.cdist(x1, x2)
        # first allocates an array of shape (x1.shape[0], x2.shape[0]) with np.float64 precision (8 bytes).
        chunk_size = round(2e9 / (vocabulary.shape[0] * 8))

    # Copy the vocabulary to GPU. Returns a reference to the original array if it is already on GPU.
    vocab_cp = cp.asarray(vocabulary)

    # Declare the result
    noisy_words_ids = np.empty(
        shape=(number_of_words), dtype=best_uint_type(vocabulary.shape[0])
    )

    # Compute the distances and find the nearest neighbor, chunk-by-chunk
    for i in range(0, number_of_words, chunk_size):
        j = min(i + chunk_size, number_of_words)
        distances = distance.cdist(embeddings[i:j], vocab_cp, distance_metric)
        noisy_words_ids[i:j] = distances.argmin(axis=-1).get()

    return noisy_words_ids


def noisy_embeddings_to_ids_with_post_processing_fix(
    embeddings: np.ndarray,
    vocabulary: np.ndarray,
    dx_constant: float,
    epsilon: int,
    distance_metric: str = "euclidean",
) -> np.ndarray:
    """Applies the post-processing fix proposed in the paper.

    Args:
        embeddings (np.ndarray): A two-dimensional array containing the embeddings we want to process.
        vocabulary (np.ndarray): A two-dimensional array containing all the embeddings of the vocabulary
            to compute the fix against.
        dx_constant (float): The c constant in the formula of the paper.
        epsilon (int): The epsilon value in the dx-privacy formula.
        distance_metric (str, optional): The distance metric to use for ranking elements of the vocabulary. Defaults to "euclidean".

    Returns:
        np.ndarray: A one-dimensional numpy array containing the ids of the sampled replacements.
    """
    input_size = embeddings.shape[0]
    vocab_size = vocabulary.shape[0]

    # We first find the nearest neighbors of each of the noisy embeddings, called the "pivots" here
    pivot_word_ids = noisy_embeddings_to_ids(embeddings, vocabulary, distance_metric)
    pivot_word_embeddings = vocabulary[pivot_word_ids]

    del pivot_word_ids  # Save RAM

    # Rank the elements of the vocabulary according to their distance with each of the pivot embeddings
    embeddings_neighbors_ranked = rank_neighbors(
        pivot_word_embeddings, vocabulary, distance_metric
    )

    # Compute probabilities
    probabilities = np.exp(-dx_constant * epsilon * embeddings_neighbors_ranked)

    # Normalize probabilities along the last axis
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    # Sample one element for each embedding and return their ids
    noisy_ids = np.empty((input_size), dtype=best_uint_type(vocab_size))
    for i in range(input_size):
        noisy_ids[i] = np.random.choice(vocab_size, size=1, p=probabilities[i]).item()

    return noisy_ids
