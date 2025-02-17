import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_hdf('./dataset1.h5')

def getPoint(r1, r2, v1, v2):
    #Calculation of the closest point of approach
    cross_st = np.cross(v1, v2)
    cross_stnorm = np.linalg.norm(cross_st)
    vts = np.dot(v1, v2)
    if cross_stnorm < 1.0e-6 or vts < 1.0e-6:
        return False, [0, 0, 0]
    cross_sst = np.cross(v1, cross_st)
    DeltaR = r1 - r2
    xpoca2 = r2 - v2 * np.dot(DeltaR, cross_sst)/cross_stnorm**2
    xpoca1 = r1 + v1 * np.dot((xpoca2-r1), v1)/vts
    v = 0.5 * (xpoca1 + xpoca2)
    return True, v

binInfo = dict()
binInfo['thresholdx'] = 0.4
binInfo['thresholdy'] = 0.1
binInfo['limitX'] = [-10, 10]
binInfo['limitY'] = [-5, 5]
binInfo['limitZ'] = [-5, 5]

ax = []
ay = []
az = []

x = []
y = []
z = []

# loop through the rows using iterrows()
for index, row in dataset.iterrows():
    #if index > 100:
    #    break
    r1 = np.asarray([row['x1'], row['y1'], row['z1']])
    r2 = np.asarray([row['x2'], row['y2'], row['z2']])
    v1 = np.asarray([row['vx1'], row['vy1'], row['vz1']])
    v2 = np.asarray([row['vx2'], row['vy2'], row['vz2']])
    dtx = row['dthetax']
    dty = row['dthetay']

    valid = False
    ###Apply here a simple angular selection
    if abs(dtx) > binInfo['thresholdx'] or abs(dty) > binInfo['thresholdy']:
        valid, v = getPoint(r1, r2, v1, v2)
        if not valid:
            continue
        if v[0] < binInfo['limitX'][0] or v[0] > binInfo['limitX'][1]:
            continue
        if v[1] < binInfo['limitY'][0] or v[1] > binInfo['limitY'][1]:
            continue
        if v[2] < binInfo['limitZ'][0] or v[2] > binInfo['limitZ'][1]:
            continue
        ax.append(v[0])
        ay.append(v[1])
        az.append(v[2])
    else:
        continue
    x = np.asarray(ax)
    y = np.asarray(ay)
    z = np.asarray(az)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c="blue", s=1)
ax.view_init(elev=20, azim=45)
plt.show()
