local GMM = torch.class('vlfeat.GMM')
local ffi    = require 'ffi'
local C      = vlfeat.C
local NULL   = vlfeat.NULL

local VL_GMM_INIT = {
  RAND                   = C.VlGMMRand,
  KMEANS               = C.VlGMMKMeans,
	CUSTOM = C.VlGMMCustom
}

local assert_valid = function(self, tensor)
  local th = torch.type(tensor)
  local vl = self.dataType

  assert(
    vlfeat.TH_VL_TYPE[th] == vl,
    'expected ' .. vlfeat.VL_TH_TYPE[vl] .. ', got ' .. th
  )

  assert(
    tensor:dim() == 2,
    'expected 2D tensor, got ' .. tensor:dim() .. 'D tensor'
  )

  assert(
    tensor:isContiguous(),
    'expected a contiguous tensor'
  )
end

function GMM:__init(dimension, numClusters)
  if dimension and numClusters then
    assert(
      type(dimension) == 'number',
      'expected dimension to be a `number`, got `' .. type(dimension) .. '`'
    )
		assert(
      type(numClusters) == 'number',
      'expected numClusters to be a `number`, got `' .. type(numClusters) .. '`'
    )
	else
		error('GMM Requires the number of dimensions and the number of clusters.')
  end
  self.handle = ffi.gc(
    C.vl_gmm_new(C.VL_TYPE_DOUBLE, dimension, numClusters),
    C.vl_gmm_delete
  )
	self.dataType = C.VL_TYPE_DOUBLE
	self.dimension = dimension
	self.numClusters = numClusters
end

function GMM:float()
  self.dataType = C.VL_TYPE_FLOAT
  return self
end

function GMM:double()
  self.dataType = C.VL_TYPE_DOUBLE
  return self
end

function GMM:maxIter(value)
  C.vl_gmm_set_max_num_iterations(self.handle,value)
end

function GMM:initialization(inittype)
	inittype = inittype or 'RAND'
  C.vl_gmm_set_initialization (self.handle,VL_GMM_INIT[inittype])
end

function GMM:cluster(data)
	assert_valid(self, data)
  return C.vl_gmm_cluster (self.handle, data:data() , data:size(1))
end

function GMM:means()
  assert(
    self.handle.means ~= NULL,
    'NULL means'
  )
  local means = torch.getmetatable(
    vlfeat.VL_TH_TYPE[self.handle.dataType]
  ).new():resize(
    tonumber(self.handle.numClusters),
    tonumber(self.handle.dimension)
  )
	assert(means:isContiguous())
	ffi.copy(
    means:data(),
    self.handle.means,
    means:nElement() * means:storage():elementSize()
	)
  return means 
end

function GMM:covariances()
    assert(
    self.handle.covariances ~= NULL,
    'NULL covariances'
  )
  local covariances = torch.getmetatable(
    vlfeat.VL_TH_TYPE[self.handle.dataType]
  ).new():resize(
    tonumber(self.handle.numClusters),
    tonumber(self.handle.dimension)
  )
	assert(covariances:isContiguous())
	ffi.copy(
    covariances:data(),
    self.handle.covariances,
    covariances:nElement() * covariances:storage():elementSize()
	)
  return covariances 
end

function GMM:priors()
	assert(
    self.handle.priors ~= NULL,
    'NULL priors'
  )
  local priors = torch.getmetatable(
    vlfeat.VL_TH_TYPE[self.handle.dataType]
  ).new():resize(
    tonumber(self.handle.numClusters)
  )
	assert(priors:isContiguous())
	ffi.copy(
    priors:data(),
    self.handle.covariances,
    priors:nElement() * priors:storage():elementSize()
	)
  return priors 
end

function GMM:posteriors()
  	assert(
    self.handle.posteriors ~= NULL,
    'NULL posteriors'
  )
  local posteriors = torch.getmetatable(
    vlfeat.VL_TH_TYPE[self.handle.dataType]
  ).new():resize(
    tonumber(self.handle.numClusters),
		tonumber(self.handle.numData)
  )
	assert(posteriors:isContiguous())
	ffi.copy(
    posteriors:data(),
    self.handle.covariances,
    posteriors:nElement() * posteriors:storage():elementSize()
	)
  return posteriors 
end