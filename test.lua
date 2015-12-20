require 'vlfeat'
-- 1000 points in 2D
numPoints = 100
dimension = 2

numClusters = 5



a = vlfeat.GMM(dimension,numClusters)
a:maxIter(1)
a:initialization()
b = torch.rand(numPoints,dimension)



a:cluster(b)
print(a:means())
print(a:covariances())
print(a:priors())
print(a:posteriors())
--print(a:covariances())
--print(a:means())


print('Done testing')
