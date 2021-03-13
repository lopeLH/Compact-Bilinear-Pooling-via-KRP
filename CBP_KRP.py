import torch
from torch import nn

import numpy as np
from numba import jit, prange
from scipy.sparse import csr_matrix


class CBP_KRP(nn.Module):
    """ Compact Bilinear Pooling via Kernelized Random Projection (CBP-KRP)
    
        This implementation of CBP-KRP uses a redundant random vectors matrix representation
        where the unique p random vectors are repeated as needed to ease inference in massively paralell
        devices such as the GPU. This means this implementation stores duplicated copies of the
        random vectors and performs redundat computations, but in turn avoids the complex indexig
        in the main algorithm loop which would be very inefficient when running on GPU.
        
        One should use this implementation at training time to take advantage of GPU acceleration,
        and use the class CBP_KRP_cpu_inference at test time for memory and computation efficient
        inference in low computational power devides such as the raspberry pi. Furthermore, the
        inference with CBP_KRP_cpu_inference will take advantage of sparse matrix multiplication
        routines if the sparsity level of the random vectors is high enough.
    """
    
    def __init__(self, input_dim, k, t=2, s=100, p=5000, epsilon=1e-9):
        """
        Initialize internal randomized variables:
        Agrs:
            input_dim: Dimensionality of the input feature vectors. The input to this layer must be given
                       as a feature map of shape (batch_size, input_dim, height, width).
            k: Dimensionality of the output feature descriptor.
            t: Number of vectors to sum to apply the Central Limit Theorem.
            s: Sparsity level in Achlioptas' distribution for the random vectors
            p: Total number of unique random projection vectors to generate.
            epsilon: Replace zeros in the projection vectors by this value to avoid zero division in gradient
                     computation.
        """
        super().__init__()

        random_vectors = torch.Tensor(np.random.choice((-1, epsilon, 1), size=(p, input_dim), p=[1./(2*s), 1-1./s, 1./(2*s)]))
        
        assert random_vectors.shape == (p, input_dim), "Should contain p random vectors of dimension input_dim"
        
        redundant_random_vectors = torch.cat([random_vectors for _ in range(2*t*k // p)] \
                                             + [random_vectors[:2*t*k % p]], axis=0)
        
        # random_indexes indicates which unique random vector from random_vectors is used at each position
        # of the redundant_random_vectors matrix
        random_indexes = np.concatenate([range(p) for _ in range(2*t*k // p)] + [range(2*t*k % p)], axis=0)
        
        # Shuffle the contents of redundant_random_vectors, which at this point contains
        # several consequitive copies of random_vectors
        shuffle_idxs = torch.randperm(2*t*k)
        redundant_random_vectors = redundant_random_vectors[shuffle_idxs]
        random_indexes = random_indexes[shuffle_idxs]

        assert redundant_random_vectors.shape == (2*t*k, input_dim), "Should contain 2*t*k vectors of dimension input_dim"
        assert random_indexes.shape == (2*t*k,), "Should contain 2*t*k indexes"
        
        # This constant can be factored out the be applied at the end, so that the projection matrix
        # can stay composed of {-1, 0, 1}
        self.constant = np.sqrt(s) ** 2 / np.sqrt(k * t)
        
        self.redundant_random_vectors = redundant_random_vectors.T
        self.random_vectors = random_vectors
        self.random_indexes = random_indexes.reshape(2, t, k).astype(np.int)
        self.k, self.t, self.s, self.p = k, t, s, p
        self.expected_input_dim = input_dim

        
    def forward(self, x):
        assert len(x.shape) in [3, 4], \
            "Input tensor should be one of:" \
            " - A batch of 2D feature maps with shape (batch_size, input_dim, height, width)" \
            " - A batch of 1D feature maps with shape (batch_size, input_dim, n_descriptors)"
        
        if len(x.shape) == 4:
            x = x.permute(0,2,3,1)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        else:
            x = x.permute(0,2,1)
            
        # At this point x is of shape (batch_size, n_descriptors, features)
        batch_size, n_descriptors, input_dim = x.shape
        assert self.expected_input_dim == input_dim

        x = torch.matmul(x, self.redundant_random_vectors)
        assert x.shape == (batch_size, n_descriptors, 2 * self.t * self.k)
        
        x = x.reshape(batch_size, n_descriptors, 2, self.t, self.k)
        
        x = x[:,:,0,:,:] * x[:,:,1,:,:]
        assert x.shape == (batch_size, n_descriptors, self.t, self.k)
        
        # Sum for the CLT
        x = torch.sum(x, axis=2)
        assert x.shape == (batch_size, n_descriptors, self.k)
        
        # Global sum pooling
        x = torch.sum(x, axis=1)
        assert x.shape == (batch_size, self.k)
        
        return self.constant * x
    
    
class CBP_KRP_cpu_inference():
    
    random_vectors = None
    random_indexes = None
    
    def __init__(self, random_vectors, random_indexes, s, use_sparse_matrix_multiplication=True):
        """
        Compact Bilinear Pooling via Kernelized Random Projection (CBP-KRP)
        
        Class intended for fast CBP-KRP inference in low computational power devices. Takes advantage
        of sparse matrix multiplication routines.
        
        Agrs:
            random_vectors: Set of unique random vectors used by the CBP_KRP layer, as a numpy array
                            of shape (n_random_vectors (p), input_dim).
            
            s: Sparsity level in Achlioptas' distribution used when generating the the random vectors.
            
            random_indexes: Matrix of indexes indicating which vectors of random_vectors are used for 
                            each computation in the main loop of the algorithm. Must contain a total
                            of 2 * t * k indexes, expressed as a numpy int array of shape (2, t, k).
                            Each index should be in range [0, p]
            
            use_sparse_matrix_multiplication: If True, store random_vectors as a csr_matrix sparse
                                              matrix and use sparse matrix multiplication functions
                                              when doing inference.
        """
        
        assert len(random_vectors.shape) == 2
        assert len(random_indexes.shape) == 3
        
        _, self.t, self.k = random_indexes.shape
        self.p, self.expected_input_dim = random_vectors.shape
        self.s = s
        
        assert np.max(random_indexes) <= self.p, "Indexes cannot be higher than the number of unique random vectors"
        assert np.min(random_indexes) >= 0, "Indexes cannot be lower than zero"
        
        self.random_vectors = random_vectors.T
        self.random_indexes = random_indexes

        if use_sparse_matrix_multiplication:
            self.random_vectors[np.abs(self.random_vectors) < 0.01] = 0
            self.random_vectors = csr_matrix(self.random_vectors)

        
    def forward(self, x):        
        assert len(x.shape) in [3, 4], \
            "Input tensor should be one of:" \
            " - A batch of 2D feature maps with shape (batch_size, features, height, width)" \
            " - A batch of 1D feature maps with shape (batch_size, features, n_descriptors)"

        if len(x.shape) == 4:
            x = x.transpose(0,2,3,1)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        else:
            x = x.transpose(0,2,1)
            
        # At this point x is of shape (batch_size, n_descriptors, input_dim)
        batch_size, n_descriptors, input_dim = x.shape
        assert input_dim == self.expected_input_dim
            
        x = x.reshape(-1, x.shape[-1])
        assert x.shape == (batch_size * n_descriptors, input_dim)

        K = np.dot(x, self.random_vectors) if isinstance(self.random_vectors, np.ndarray) \
            else x * self.random_vectors
        assert K.shape == (batch_size * n_descriptors, self.p), K.shape

        K = K.reshape((batch_size, n_descriptors, self.p))
        constant = np.sqrt(self.s) **2 / np.sqrt(self.k * self.t)

        return optimized_looping(K, self.random_indexes, self.t, self.p, self.k, constant)


@jit(nopython=True, cache=True, parallel=True)
def optimized_looping(K, random_indexes, t, p, k, constant):
    global_descriptor = np.zeros((K.shape[0], k)).astype(K.dtype)
    
    for sample_idx in prange(K.shape[0]):
        for t_idx in range(t):
            for location_idx in range(K.shape[1]):
                for output_feature_idx in range(k):
                    global_descriptor[sample_idx, output_feature_idx] += \
                        K[sample_idx, location_idx, random_indexes[0, t_idx, output_feature_idx]] * \
                        K[sample_idx, location_idx, random_indexes[1, t_idx, output_feature_idx]]

    return constant * global_descriptor   