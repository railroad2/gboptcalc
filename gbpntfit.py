import numpy as np
import pylab as plt

from pprint import pprint
from scipy.optimize import minimize
import lmfit


def read_data(fname):
    dat = np.genfromtxt(fname, delimiter=',', names=True)
    return dat


def proj_thetaphi(theta, phi, degrees=True):
    if degrees:
        theta = np.radians(theta)
        phi = np.radians(phi)

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)

    return x, y


def get_pnt_fit_old():
    dat = read_data('./gb_optics_sim.csv')
     
    xfoc = dat['Xfoc']
    yfoc = dat['Yfoc']

    theta = dat['theta']
    phi   = dat['phi']

    xp, yp = proj_thetaphi(theta, phi)
 
    xs = list(set(xfoc))
    xs.sort()
    idx_full = np.arange(len(dat))
    plt.figure()

    ps = []
    dyfit = []
    dy2fit = []
    for x0 in xs:
        idx = idx_full[xfoc==x0]
        xp0 = xp[idx]
        yp0 = yp[idx]

        p = np.polyfit(xp0, yp0, deg=2)
        ps.append(p)
        y = np.poly1d(p)
        x = np.linspace(-0.25, 0.25, 1001)
        plt.scatter(xp0, yp0, s=3, c='b')
        plt.plot(x, y(x), 'b:', linewidth='0.5')
        dyfit.append(sum((y(xp0) - yp0)))
        dy2fit.append(sum((y(xp0) - yp0)**2))

    ps = np.array(ps).T
    
    p0 = np.polyfit(xs, ps[0], deg=2)
    y0 = np.poly1d(p0)
    p1 = np.polyfit(xs, ps[1], deg=2)
    y1 = np.poly1d(p1)
    p2 = np.polyfit(xs, ps[2], deg=2)
    y2 = np.poly1d(p2)

    #plt.figure()
    dyfitfit = []
    dy2fitfit = []
    for x0 in xs:
        idx = idx_full[xfoc==x0]
        xp0 = xp[idx]
        yp0 = yp[idx]

        p = [y0(x0), y1(x0), y2(x0)] 
        y = np.poly1d(p)
        x = np.linspace(-0.25, 0.25, 1001)
        plt.scatter(xp0, yp0, s=3, c='r')
        plt.plot(x, y(x), 'r:', linewidth='0.5')
        dyfitfit.append(sum((y(xp0) - yp0)))
        dy2fitfit.append(sum((y(xp0) - yp0)**2))

    print ('fit')
    for d1, d2 in zip(dyfit, dyfitfit):
        print (d1, d2)

    print ('fitfit')
    for d1, d2 in zip(dy2fit, dy2fitfit):
        print (d1, d2)

    plt.figure()
    plt.plot(xs, ps[0], '*', label='c2')
    plt.plot(xs, ps[1], '*', label='c1')
    plt.plot(xs, ps[2], '*', label='c0')
    plt.plot(xs, y0(xs), label='c2_fit : {:0.5e}$x^2$ + {:0.5e}$x$ + {:0.5e}'.format(*p0))
    plt.plot(xs, y1(xs), label='c1_fit : {:0.5e}$x^2$ + {:0.5e}$x$ + {:0.5e}'.format(*p1))
    plt.plot(xs, y2(xs), label='c0_fit : {:0.5e}$x^2$ + {:0.5e}$x$ + {:0.5e}'.format(*p2))
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('fit parameters')

    #plt.plot(xs, ps[1], '*')
    #plt.plot(xs, ps[2], '*')
   
    print (p0)
    print (p1)
    print (p2)
     
    plt.show()  


def get_pnt_fit_y():
    dat = read_data('./gb_optics_sim.csv')
     
    xfoc = dat['Xfoc']
    yfoc = dat['Yfoc']

    theta = dat['theta']
    phi   = dat['phi']

    xp, yp = proj_thetaphi(theta, phi)
 
    xs = list(set(xfoc))
    xs.sort()
    idx_full = np.arange(len(dat))
    plt.figure()

    ps = []
    dyfit = []
    dy2fit = []
    for x0 in xs:
        idx = idx_full[xfoc==x0]
        xp0 = xp[idx]
        yp0 = yp[idx]

        p = np.polyfit(xp0, yp0, deg=2)
        ps.append(p)
        y = np.poly1d(p)
        x = np.linspace(-0.25, 0.25, 1001)
        plt.scatter(xp0, yp0, s=3, c='b')
        plt.plot(x, y(x), 'b:', linewidth='0.5')
        dyfit.append(sum((y(xp0) - yp0)))
        dy2fit.append(sum((y(xp0) - yp0)**2))

    ps = np.array(ps).T
    
    p0 = np.polyfit(xs, ps[0], deg=2)
    y0 = np.poly1d(p0)
    p1 = np.polyfit(xs, ps[1], deg=2)
    y1 = np.poly1d(p1)
    p2 = np.polyfit(xs, ps[2], deg=2)
    y2 = np.poly1d(p2)

    #plt.figure()
    dyfitfit = []
    dy2fitfit = []
    for x0 in xs:
        idx = idx_full[xfoc==x0]
        xp0 = xp[idx]
        yp0 = yp[idx]

        p = [y0(x0), y1(x0), y2(x0)] 
        y = np.poly1d(p)
        x = np.linspace(-0.25, 0.25, 1001)
        plt.scatter(xp0, yp0, s=3, c='r')
        plt.plot(x, y(x), 'r:', linewidth='0.5')
        dyfitfit.append(sum((y(xp0) - yp0)))
        dy2fitfit.append(sum((y(xp0) - yp0)**2))

    print ('fit')
    for d1, d2 in zip(dyfit, dyfitfit):
        print (d1, d2)

    print (p0)
    print (p1)
    print (p2)
     

def get_pnt_fit():
    dat = read_data('./gb_optics_sim.csv')
     
    ## take needed data
    xfoc = -dat['Yfoc']
    yfoc = dat['Xfoc']
    theta = dat['theta']
    phi   = dat['phi']
    xp, yp = proj_thetaphi(theta, phi)
 
    ## getting indices with the same y in focal plane.
    ys = list(set(yfoc))
    ys.sort()
    idx_full = np.arange(len(dat))

    pxs = []
    pys = []
    dyfit = []
    dy2fit = []
    dists = []

    ## plot sky projection
    plt.figure()
    plt.scatter(xp, yp, s=3, c='b')

    ## loop for y focal plane values
    ## fit xp and yp w.r.t. xfoc with the same yfoc value. 
    for y0 in ys:
        idx = idx_full[yfoc==y0]
        xp0 = xp[idx]
        yp0 = yp[idx]
        xfoc0 = xfoc[idx]

        parxp = np.polyfit(xfoc0, xp0, deg=3)
        paryp = np.polyfit(xfoc0, yp0, deg=2)

        xpfit = np.poly1d(parxp)
        ypfit = np.poly1d(paryp)
        print (y0)
        print (xpfit)
        print (ypfit)

        xarr = np.linspace(-100, 100, 1001)

        plt.scatter(xp0, yp0, s=3, c='b')
        plt.scatter(xpfit(xfoc0), ypfit(xfoc0), s=3, c='r')

        pxs.append(parxp)
        pys.append(paryp)

        dists.append(list(((xpfit(xfoc0)-xp0)**2 + (ypfit(xfoc0)-yp0)**2)**0.5))

    dists = np.array(sum(dists, []))

    dists_sum = np.sum(dists)
    dists2_sum = np.sum(dists**2)
    dists_std = np.std(dists)

    print ('dists_sum=  ', dists_sum)
    print ('dists2_sum= ', dists2_sum)
    print ('dists_std=  ', dists_std)
    print ('dist mean = ', np.mean(dists)) 
    print ('dist max = ', max(dists))
    print ('dist min = ', min(dists))

    pxs = np.array(pxs).T
    pys = np.array(pys).T

    ## fit the fit parameters
    ppxs = []
    pxfits = []
    plt.figure()

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    for i, px in enumerate(pxs):
        deg = 2
        ppxs.append(np.polyfit(ys, px, deg=deg))
        pxfits.append(np.poly1d(ppxs[-1]))

        plt.title('x parameters fit')
        plt.plot(ys, px, f'{colors[i]}*', label=f'c{len(pxs)-i-1}')
        plt.plot(ys, pxfits[-1](ys), f'{colors[i]}:', label=f'c{len(pxs)-i-1}')
        print (f'ppx{i} = ', ppxs[i])

    plt.legend()

    ppys = []
    pyfits = []
    plt.figure()

    for i, py in enumerate(pys):
        ppys.append(np.polyfit(ys, py, deg=deg))
        pyfits.append(np.poly1d(ppys[-1]))

        plt.title('y parameters fit')
        plt.plot(ys, py, f'{colors[i]}*', label=f'c{len(pys)-i-1}')
        plt.plot(ys, pyfits[-1](ys), f'{colors[i]}:', label=f'c{len(pys)-i-1}')
        print (f'ppy{i} = ', ppys[i])

    plt.legend()

    for px in ppxs: 
        print (poly_arr2tex(px))

    for py in ppys: 
        print (poly_arr2tex(py))

    return 
    

def num_sci2tex(sci, ndigits=4):
    # convert number in scientific expression to latex 
    b = f'{sci:+0.{ndigits}e}'.split('e')
    tex = fr'{b[0]} \times 10^{{{b[1]}}}'
    return tex


def poly_arr2tex(arr, ndigits=4, l2s=True):
    exponents = np.arange(len(arr))
    if l2s:
        exponents = exponents[::-1]

    tex = [] 
    for a, exp in zip(arr, exponents):
        tmp = num_sci2tex(a, ndigits=ndigits)
        if exp == 1:
            tmp +=  f' x '
        elif exp:   
            tmp +=  f' x^{exp} '

        tex.append(tmp)

    tex = ''.join(tex)

    if tex[0] == '+':
        tex = ' ' + tex[1:]

    tex = '$' + tex + '$'
    return tex


if __name__=="__main__":
    get_pnt_fit()
    plt.show()  


