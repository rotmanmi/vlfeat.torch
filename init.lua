require 'torch'
vlfeat = {}     -- top-level module for Torch class system (it MUST be global!)

require 'vlfeat.ffi'
require 'vlfeat.kmeans'
require 'vlfeat.gmm'

return vlfeat
