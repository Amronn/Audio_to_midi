import numpy as np
import matplotlib.pyplot as plt
x = [1,2,3]
y = [-1,1,3]

# print(np.correlate(x,y))

har = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
harmoniczne = np.zeros(85)
harmoniczne[har[:3]] = 1

chroma = [1, 13, 20, 25, 29, 32, 35, 37, 39, 40, 41, 43, 44,45,46]
# chroma = [0, 12, 19, 24, 28, 31, 34, 36, 38, 39, 40, 42, 43,44,45]
chrom = np.zeros(85)
chrom[chroma] = 1
cor = np.correlate(harmoniczne, chrom, 'same')
print(cor)
print(cor.shape)
print(np.argmax(cor))
# plt.plot(cor)
# plt.show()