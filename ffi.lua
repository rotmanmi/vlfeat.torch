local ffi = require 'ffi'

-- host specific typedef-s
if ffi.abi('64bit') then
  ffi.cdef[[
typedef long long unsigned  vl_uint64;
typedef int       unsigned  vl_uint32;
typedef vl_uint64           vl_size;
]]
else
  ffi.cdef[[
typedef int       unsigned  vl_uint32;
typedef vl_uint32           vl_size;
]]
end

ffi.cdef[[
typedef vl_uint32 vl_type;
typedef int                 vl_bool ;

enum {
  VL_TYPE_FLOAT = 1,
  VL_TYPE_DOUBLE
};

typedef float (*VlFloatVectorComparisonFunction)(
  vl_size dimension, float const * X, float const * Y
);
typedef double (*VlDoubleVectorComparisonFunction)(
  vl_size dimension, double const * X, double const * Y
);

typedef enum _VlKMeansAlgorithm {
  VlKMeansLloyd,
  VlKMeansElkan,
  VlKMeansANN
} VlKMeansAlgorithm;

typedef enum _VlKMeansInitialization {
  VlKMeansRandomSelection,
  VlKMeansPlusPlus
} VlKMeansInitialization;

enum _VlVectorComparisonType {
  VlDistanceL1,
  VlDistanceL2,
  VlDistanceChi2,
  VlDistanceHellinger,
  VlDistanceJS,
  VlDistanceMahalanobis,
  VlKernelL1,
  VlKernelL2,
  VlKernelChi2,
  VlKernelHellinger,
  VlKernelJS
};

typedef enum _VlVectorComparisonType VlVectorComparisonType;

typedef struct _VlKMeans
{

  vl_type dataType;
  vl_size dimension;
  vl_size numCenters;
  vl_size numTrees;
  vl_size maxNumComparisons;
  VlKMeansInitialization initialization;
  VlKMeansAlgorithm algorithm;
  VlVectorComparisonType distance;
  vl_size maxNumIterations;
  double minEnergyVariation;
  vl_size numRepetitions;
  int verbosity;
  void * centers;
  void * centerDistances;
  double energy;
  VlFloatVectorComparisonFunction floatVectorComparisonFn;
  VlDoubleVectorComparisonFunction doubleVectorComparisonFn;
} VlKMeans;

VlKMeans * vl_kmeans_new(
  vl_type dataType,
  VlVectorComparisonType distance
);

void vl_kmeans_delete(
  VlKMeans * self
);

double vl_kmeans_cluster(
  VlKMeans * self,
  void const * data,
  vl_size dimension,
  vl_size numData,
  vl_size numCenters
);

void vl_kmeans_set_centers(
  VlKMeans * self,
  void const * centers,
  vl_size dimension,
  vl_size numCenters
);

void vl_kmeans_init_centers_with_rand_data(
  VlKMeans * self,
  void const * data,
  vl_size dimensions,
  vl_size numData,
  vl_size numCenters
);

void vl_kmeans_init_centers_plus_plus(
  VlKMeans * self,
  void const * data,
  vl_size dimensions,
  vl_size numData,
  vl_size numCenters
);

double vl_kmeans_refine_centers(
  VlKMeans * self,
  void const * data,
  vl_size numData
);

void vl_kmeans_quantize(
  VlKMeans * self,
  vl_uint32 * assignments,
  void * distances,
  void const * data,
  vl_size numData
);
]]

ffi.cdef [[
	typedef enum _VlGMMInitialization
	{
		VlGMMKMeans, 
		VlGMMRand,   
		VlGMMCustom  
	} VlGMMInitialization ;

]]


ffi.cdef [[
	struct _VlGMM
	{
		vl_type dataType ;                  /**< Data type. */
		vl_size dimension ;                 /**< Data dimensionality. */
		vl_size numClusters ;               /**< Number of clusters  */
		vl_size numData ;                   /**< Number of last time clustered data points.  */
		vl_size maxNumIterations ;          /**< Maximum number of refinement iterations. */
		vl_size numRepetitions   ;          /**< Number of clustering repetitions. */
		int     verbosity ;                 /**< Verbosity level. */
		void *  means;                      /**< Means of Gaussian modes. */
		void *  covariances;                /**< Diagonals of covariance matrices of Gaussian modes. */
		void *  priors;                     /**< Weights of Gaussian modes. */
		void *  posteriors;                 /**< Probabilities of correspondences of points to clusters. */
		double * sigmaLowBound ;            /**< Lower bound on the diagonal covariance values. */
		VlGMMInitialization initialization; /**< Initialization option */
		VlKMeans * kmeansInit;              /**< Kmeans object for initialization of gaussians */
		double LL ;                         /**< Current solution loglikelihood */
		vl_bool kmeansInitIsOwner; /**< Indicates whether a user provided the kmeans initialization object */
	} ;
	
	typedef struct _VlGMM VlGMM ;
	
	VlGMM * 	vl_gmm_new (vl_type dataType, vl_size dimension, vl_size numComponents);
	
	void 	vl_gmm_reset (VlGMM *self);
	
	void vl_gmm_delete (VlGMM *self);
	
	vl_type vl_gmm_get_data_type (VlGMM const *self);
	
	vl_size vl_gmm_get_num_clusters (VlGMM const *self);
	
	vl_size vl_gmm_get_num_data (VlGMM const *self);
	
	double 	vl_gmm_get_loglikelihood (VlGMM const *self);
	
	int 	vl_gmm_get_verbosity (VlGMM const *self);
	
	void 	vl_gmm_set_verbosity (VlGMM *self, int verbosity);
	
	void const * 	vl_gmm_get_means (VlGMM const *self);
	
	void const * 	vl_gmm_get_covariances (VlGMM const *self);
	
	void const * 	vl_gmm_get_priors (VlGMM const *self);
	
	void const * 	vl_gmm_get_posteriors (VlGMM const *self);
	
	vl_size 	vl_gmm_get_max_num_iterations (VlGMM const *self);
	
	void 	vl_gmm_set_max_num_iterations (VlGMM *self, vl_size maxNumIterations);
	
	void 	vl_gmm_set_num_repetitions (VlGMM *self, vl_size numRepetitions);
	
	vl_size 	vl_gmm_get_dimension (VlGMM const *self);
	
	VlGMMInitialization 	vl_gmm_get_initialization (VlGMM const *self);
	
	void 	vl_gmm_set_initialization (VlGMM *self, VlGMMInitialization init);
	
	VlKMeans * 	vl_gmm_get_kmeans_init_object (VlGMM const *self);
	
	void 	vl_gmm_set_kmeans_init_object (VlGMM *self, VlKMeans *kmeans);
	
	double const * 	vl_gmm_get_covariance_lower_bounds (VlGMM const *self);
	
	void 	vl_gmm_set_covariance_lower_bounds (VlGMM *self, double const *bounds);
	
	void 	vl_gmm_set_covariance_lower_bound (VlGMM *self, double bound);
	
	VlGMM * 	vl_gmm_new_copy (VlGMM const *self);
	
	void 	vl_gmm_init_with_rand_data (VlGMM *self, void const *data, vl_size numData);
	
	void 	vl_gmm_init_with_kmeans (VlGMM *self, void const *data, vl_size numData, VlKMeans *kmeansInit);
	
	double 	vl_gmm_cluster (VlGMM *self, void const *data, vl_size numData);
	
	double 	vl_gmm_em (VlGMM *self, void const *data, vl_size numData);
	
	void 	vl_gmm_set_means (VlGMM *self, void const *means);
	
	void 	vl_gmm_set_covariances (VlGMM *self, void const *covariances);
	
	void 	vl_gmm_set_priors (VlGMM *self, void const *priors);
	
	vl_size 	vl_gmm_get_num_repetitions (VlGMM const *self);
	
	void free (void *) ;
]] 

-- RTLD_GLOBAL mode (cf. man 3 dlopen)
local global = true

local ok, C = pcall(ffi.load, 'vl', global)
if not ok then
   error('cannot load VLFeat library: ' .. C)
end
vlfeat.C = C

if not jit then
  -- see https://github.com/facebook/luaffifb#pointer-comparison
  vlfeat.NULL = ffi.C.NULL
end

vlfeat.TH_VL_TYPE = {
  ['torch.FloatTensor']  = C.VL_TYPE_FLOAT,
  ['torch.DoubleTensor'] = C.VL_TYPE_DOUBLE
}

vlfeat.VL_TH_TYPE = {
  [C.VL_TYPE_FLOAT]      = 'torch.FloatTensor',
  [C.VL_TYPE_DOUBLE]     = 'torch.DoubleTensor'
}

vlfeat.VECT_COMPARISON_TYPE = {
  L1                     = C.VlDistanceL1,
  L2                     = C.VlDistanceL2,
  CHI2                   = C.VlDistanceChi2,
  HELLINGER              = C.VlDistanceHellinger,
  JS                     = C.VlDistanceJS,
  MAHALANOBIS            = C.VlDistanceMahalanobis,
  KERNELL1               = C.VlKernelL1,
  KERNELL1               = C.VlKernelL2,
  KERNELCHI2             = C.VlKernelChi2,
  KERNELHELLINGER        = C.VlKernelHellinger,
  KERNELJS               = C.VlKernelJS
}
