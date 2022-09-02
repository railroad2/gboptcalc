import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint

fname = './optdata.json'

debug = 0

with open(fname) as f:
    optdata = json.load(f)

def read_data(fname, delimiter=None):
    if os.path.splitext(fname)[1] == '.csv':
        delimiter=','

    dat = np.genfromtxt(fname, delimiter=delimiter, names=True)
    return dat


def proj_thetaphi(theta, phi, degrees=True):
    if degrees:
        theta = np.radians(theta)
        phi = np.radians(phi)

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)

    return x, y


def calc(xfoc, yfoc, keyroot):
    res = 0

    if hasattr(xfoc, '__iter__') and hasattr(yfoc, '__iter__'):
        res = []
        for xi, yi in zip(xfoc, yfoc):
            res.append(calc(xi, yi, keyroot))

        return np.array(res)
    
    if debug:
        print ()
        print ('='*50)
        print (f'xfoc, yfoc = {xfoc}, {yfoc}')
        print ('-'*50)

    for key in optdata.keys():
        if keyroot in key:
            exponent = int(key.strip(keyroot))
            p = np.poly1d(optdata[key])
            res += p(yfoc) * xfoc**exponent

            if debug:
                print (f'key, exponent = {key}, {exponent}')
                print (p)
                print (f'p(yfoc) = {p(yfoc)}')
                print (f'res = {res}')
                print ('-'*50)

    return res
        

def xy2thetaphi(x,y):
    mag = (x**2+y**2)**0.5
    theta = np.arcsin(mag) 
    phi = np.tan(y/x)
    return theta, phi


def test_fullarray(fname, LTconvention=False):
    dat = read_data(fname)

    if LTconvention:
        xfocs = -dat['Yfoc']
        yfocs = dat['Xfoc']
    else:
        xfocs = dat['Xfoc']
        yfocs = dat['Yfoc']

    dpsi = dat['omtffr'] - dat['omtfoc']
    xp, yp = proj_thetaphi(dat['theta'], dat['phi'])

    xpfit = calc(xfocs, yfocs, 'cx')
    ypfit = calc(xfocs, yfocs, 'cy')
    dpsifit = calc(xfocs, yfocs, 'cpsi')

    plt.figure()
    plt.scatter(xfocs, yfocs)
    plt.xlabel('x focalplane (cm)')
    plt.ylabel('y focalplane (cm)')

    plt.figure()
    plt.scatter(xp, yp, label='LightTools')
    plt.scatter(xpfit, ypfit, label='from fitting')
    plt.xlabel('xp')
    plt.ylabel('yp')
    plt.legend()
    
    plt.figure()
    plt.plot((xpfit - xp)/np.std(xp), label='(xpfit - xp)/std(xp)')
    plt.plot((ypfit - yp)/np.std(yp), label='(ypfit - yp)/std(yp)')
    plt.plot((dpsifit - dpsi)/np.std(dpsi), label='(dpsifit - psi)/std(psi)')
    plt.xlabel('pixel#')
    plt.ylabel('relative error')
    plt.legend()


    v1 = xy2vec(xp, yp)
    v1fit = xy2vec(xpfit, ypfit)
    ang = angle_vecs(v1, v1fit)
    plt.figure()
    plt.plot(ang)
    plt.xlabel('pixel#')
    plt.ylabel('angle between projected and fit (deg)')

    print (f'max(xpfit-xp) = {max(xpfit-xp)}')
    print (f'max(ypfit-yp) = {max(ypfit-yp)}')
    print (f'max(dpsifit-dpsi) = {max(dpsifit-dpsi)}')
    print (f'max(ang) = {max(ang)}')
    
    plt.show()
    

def test_line():
    fname = 'gb_optics_sim.csv'
    LTconvention = True
    dat = read_data(fname)

    if LTconvention:
        xfocs = dat['Yfoc']
        yfocs = -dat['Xfoc']
    else:
        xfocs = dat['Xfoc']
        yfocs = dat['Yfoc']

    xline = np.linspace(-100, 100, 21) 
    yline = np.linspace(-100, 100, 21) 

    dpsi = dat['omtffr'] - dat['omtfoc']
    xp, yp = proj_thetaphi(dat['theta'], dat['phi'])

    xpfit = calc(xline, yline, 'cx')
    ypfit = calc(xline, yline, 'cy')
    dpsifit = calc(xline, yline, 'cpsi')

    plt.figure()
    plt.scatter(xfocs, yfocs)
    plt.scatter(xline, yline)
    plt.xlabel('x focalplane (cm)')
    plt.ylabel('y focalplane (cm)')

    plt.figure()
    plt.scatter(xp, yp, label='LightTools')
    plt.scatter(xpfit, ypfit, label='from fitting')
    plt.xlabel('xp')
    plt.ylabel('yp')
    plt.legend()
    
    """
    plt.figure()
    plt.plot((xpfit - xp)/np.std(xp), label='(xpfit - xp)/std(xp)')
    plt.plot((ypfit - yp)/np.std(yp), label='(ypfit - yp)/std(yp)')
    plt.plot((dpsifit - dpsi)/np.std(dpsi), label='(dpsifit - psi)/std(psi)')
    plt.xlabel('pixel#')
    plt.ylabel('relative error')
    plt.legend()
    """

    v1 = xy2vec(xp, yp)
    v1fit = xy2vec(xpfit, ypfit)
    ang = angle_vecs(v1, v1fit)
    plt.figure()
    plt.plot(ang)
    plt.xlabel('pixel#')
    plt.ylabel('angle between projected and fit (deg)')

    """

    print (f'max(xpfit-xp) = {max(xpfit-xp)}')
    print (f'max(ypfit-yp) = {max(ypfit-yp)}')
    print (f'max(dpsifit-dpsi) = {max(dpsifit-dpsi)}')
    print (f'max(ang) = {max(ang)}')
    """
    
    plt.show()


def angle_vecs(v1, v2, deg=True): 
    ang = np.arccos(np.diag(np.dot(v1, v2.T)))
    if deg:
        ang = np.degrees(ang)

    return ang 
    

def xy2vec(x, y):
    # convert projected x, y to corresponding unit vector
    # z is always positive
    z = np.sqrt(1 - (x**2 + y**2))
    return np.array([x, y, z]).T


def main():
    xfoc = float(sys.argv[1])
    yfoc = float(sys.argv[2])

    xp   = calc(xfoc, yfoc, 'cx')
    yp   = calc(xfoc, yfoc, 'cy')
    psip = calc(xfoc, yfoc, 'cpsi')
    theta, phi = xy2thetaphi(xp, yp)

    print('xp    =', xp)
    print('yp    =', yp)
    print('psi   =', psip)
    print('theta =', theta)
    print('phi   =', phi)


def main2():
    fname = sys.argv[1]
    try:
        LTconvention = sys.argv[2]
    except:
        LTconvention = False

    print (LTconvention)
    test_fullarray(fname, LTconvention)
    

def main3():
    test_line()
    

if __name__=='__main__':
    main2()


