# A simple test for the conservative piDMD (i.e. the orthogonal Procrustes
# problem)

import numpy as np
import matplotlib.pyplot as plt
from core.operation.optimization.DMD.PhysicDMD import piDMD

np.random.seed(1)  # Set random seed
n = 10  # Number of features
m = 1000  # Number of samples

# Generate an orthogonal model
trueA, _, _ = np.linalg.svd(np.random.randn(n, n))
trueVals = np.linalg.eig(trueA)[0]

# Generate random but consistent data
X = np.random.randn(n, m)
Y = trueA.dot(X)

# Make the data noisy
noiseMag = .5
Yn = Y + noiseMag * np.random.randn(n, m)
Xn = X + noiseMag * np.random.randn(n, m)

ortoDMD = piDMD(method='orthogonal')  # Energy preserving DMD
svdDMD = piDMD(method='exact')
# Train the models
piA, piVals, piEgvectors = ortoDMD.fit(Xn, Yn)  # Energy preserving DMD
exA, exVals, exactEgvectors = svdDMD.fit(Xn, Yn)  # Exact DMD

# Display the error between the learned operators
I = np.eye(n)
print('piDMD model error is     ' + str(np.linalg.norm(ortoDMD.predict(I) - trueA, 'fro') / np.linalg.norm(trueA, 'fro')))
print('exact DMD model error is ' + str(np.linalg.norm(svdDMD.predict(I) - trueA, 'fro') / np.linalg.norm(trueA, 'fro')))

# Plot some results
plt.figure(1)
# plt.plot(np.exp(1j * np.linspace(0, 2 * np.pi)), '--', color=[0.5, 0.5, 0.5], linewidth=2)
p2 = plt.plot((exVals) + 1j * np.finfo(float).eps, 'r^', linewidth=2, markersize=10)
p3 = plt.plot((piVals) + 1j * np.finfo(float).eps, 'bx', linewidth=2, markersize=10)
p4 = plt.plot((trueVals) + 1j * np.finfo(float).eps, 'o', color=[0.5, 0.5, 0.5], linewidth=2, markersize=10)
plt.axis('equal')
plt.axis(1.3 * np.array([-1, 1, -1, 1]))

plt.legend([p2, p3, p4], ['exact DMD', 'piDMD', 'truth'], fontsize=15)
plt.title('Spectrum of linear operator', fontsize=20)

plt.show()
_ = 1
