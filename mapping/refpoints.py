import numpy as np

def refpoints(contour, cwall, ismlo=False):
    if not ismlo:
        rsup = 0.4800
        rinf = 0.4800
    else:
        rsup = 0.4573
        rinf = 0.3500
        
    nipple = find_nipple(contour, cwall)
    supPoint, infPoint = extremePoints(nipple, contour, rsup, rinf)
    
    return {'p0': nipple, 'p1': supPoint, 'p2': infPoint}

def extremePoints(nipple, contour, rsup, rinf):
    xc = contour['x']
    yc = contour['y']
    
    # Index of nipple
    diff = np.sqrt((xc - nipple['x'])**2 + (yc - nipple['y'])**2)
    in_idx = np.argmin(diff)
    
    x_phy0 = xc[in_idx::-1]
    y_phy0 = yc[in_idx::-1]
    x_phy1 = xc[in_idx:]
    y_phy1 = yc[in_idx:]
    
    r0 = np.cumsum(np.sqrt(np.diff(x_phy0)**2 + np.diff(y_phy0)**2))
    r1 = np.cumsum(np.sqrt(np.diff(x_phy1)**2 + np.diff(y_phy1)**2))
    
    rmax = np.sum(np.sqrt(np.diff(xc)**2 + np.diff(yc)**2))
    
    r = np.concatenate(([-val for val in r0[::-1]], [0], r1)) / rmax
    
    k1 = np.argmin(np.abs(r - (-rsup)))
    xsup = xc[k1]
    ysup = yc[k1]
    
    k0 = np.argmin(np.abs(r - rinf))
    xinf = xc[k0]
    yinf = yc[k0]
    
    return {'x': xsup, 'y': ysup}, {'x': xinf, 'y': yinf}

def find_nipple(contour, pecLine):
    xc = contour['x']
    yc = contour['y']
    
    m = pecLine['m']
    b = pecLine['b']
    
    p_distance = np.abs(m * xc - yc + b) / np.sqrt(m**2 + 1)
    
    if len(yc) > 0:
        p_distance[yc > 0.8 * np.max(yc)] = -np.inf
        
    if len(p_distance) > 0:
        id = np.argmax(p_distance)
        return {'x': xc[id], 'y': yc[id]}
    else:
        return {'x': 0, 'y': 0}
