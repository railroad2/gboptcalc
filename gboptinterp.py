import os

import numpy as np
import pylab as plt

from scipy.interpolate import interp2d


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


def plot_3d(theta, phi, psi):
    xp, yp = proj_thetaphi(theta, phi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xp, yp, psi, c=psi)


def plot_polar(theta, phi, psi, fig=None):
    xp, yp = proj_thetaphi(theta, phi)
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    sc = ax.scatter(xp, yp, c=psi, s=10)
    plt.colorbar(sc)


def interp_scipy(x, y, dat, npts=None):
    xref = dat['Xfoc']
    yref = dat['Yfoc']
    dtheta = dat['omtffr'] - dat['omtfoc']
    get_newtheta = interp2d(xref, yref, dat['theta'], kind='linear')
    get_newphi   = interp2d(xref, yref, dat['phi'], kind='linear')
    get_newpsi   = interp2d(xref, yref, dtheta, kind='linear')
     
    newtheta = get_newtheta(x, y)
    newphi   = get_newphi(x, y)
    newpsi   = get_newpsi(x, y)

    return newtheta[0], newphi[0], newpsi[0]


def find_neighborhoods(x, y, xref, yref, npts=2):
    xdist = x - xref
    ydist = y - yref
    dist = (xdist**2 + ydist**2)**0.5 
    indices = np.arange(len(dist))
    sort_indices = np.lexsort([indices, dist])
    neighbor_indices = sort_indices[:npts]

    return neighbor_indices


def dist_weighted_avg(arr, dists):
    warr = arr * dists 
    return np.sum(warr)/np.sum(dists)


def interp_thetaphi(x, y, dat, npts=2):
    xref = dat['Xfoc']
    yref = dat['Yfoc']
    dpsi = dat['omtffr'] - dat['omtfoc']

    theta_new = []
    phi_new = []
    psi_new = []

    for xi, yi in zip(x, y):
        nbi = find_neighborhoods(xi, yi, xref, yref, npts=npts)
        nbx = xref[nbi]
        nby = yref[nbi]

        nb_theta = dat['theta'][nbi]
        nb_phi = dat['phi'][nbi]
        nb_psi = dpsi[nbi]

        dists = ((xi - nbx)**2 + (yi - nby)**2)**0.5

        theta_new.append(dist_weighted_avg(nb_theta, dists))
        phi_new.append(dist_weighted_avg(nb_phi, dists))
        psi_new.append(dist_weighted_avg(nb_psi, dists))

    return theta_new, phi_new, psi_new
    

def lininterp(nbs, weights):
    return np.sum(nbs * weights[::-1]) / np.sum(weights)


def interp(x, y, dat, npts=2):
    xref = dat['Xfoc']
    yref = dat['Yfoc']
    dpsi = dat['omtffr'] - dat['omtfoc']
    theta = dat['theta']
    phi = dat['phi']
    xp, yp = proj_thetaphi(theta, phi)

    theta_new = []
    phi_new = []
    psi_new = []

    for xi, yi in zip(x, y):
        nbi = find_neighborhoods(xi, yi, xref, yref, npts=npts)
        nbx = xref[nbi]  
        nby = yref[nbi]

        nb_theta = theta[nbi]
        nb_phi = phi[nbi]
        nb_psi = dpsi[nbi]
        nb_xp = xp[nbi]
        nb_yp = yp[nbi]

        xdists = xi - nb_xp
        ydists = yi - nb_yp
        dists = (xdists**2 + ydists**2)**0.5

        #x_new = dist_weighted_avg(nb_xp, dists)
        #y_new = dist_weighted_avg(nb_yp, dists)
        x_new = lininterp(nb_xp, xdists)
        y_new = lininterp(nb_yp, ydists)
        print('dists= ', dists)

        psi_new.append(dist_weighted_avg(nb_psi, dists))
        phi_new.append(np.degrees(np.arctan2(x_new, y_new)))
        theta_new.append(np.degrees(np.arcsin((x_new**2+y_new**2)**0.5)))

    return theta_new, phi_new, psi_new


def main():
    dat = read_data('./gb_optics_sim.csv')
    dtheta = dat['omtffr'] - dat['omtfoc']
    print (dat)

    x = np.linspace(-100, 100, 101)
    y = np.linspace(-100, 100, 101)
    theta, phi, psi = interp_thetaphi(x, y, dat, npts=4)
    print (theta)
    print (phi)
    print (psi)

    plot_polar(dat['theta'], dat['phi'], dtheta)
    plot_polar(theta, phi, psi)

    plt.figure()
    plt.scatter(dat['Xfoc'], dat['Yfoc'], c='red', s=1)
    plt.scatter(x, y, c='blue', s=1)

    plt.show()


def main2():
    dat = read_data('./gb_optics_sim.csv')
    dtheta = dat['omtffr'] - dat['omtfoc']
    plot_3d(dat['theta'], dat['phi'], dtheta)
    xp, yp = proj_thetaphi(dat['theta'], dat['phi'])
    plt.figure()
    plt.scatter(xp, dtheta, c=yp)
    plt.figure()
    plt.scatter(yp, dtheta, c=xp)
    plt.show()
    return 
    print (dat)

    x = np.linspace(-100, 100, 101)
    y = np.linspace(-100, 100, 101)
    theta, phi, psi = interp(x, y, dat, npts=2)
    print (theta)
    print (phi)
    print (psi)

    xref, yref = proj_thetaphi(dat['theta'], dat['phi'])
    xnew, ynew = proj_thetaphi(theta, phi)

    plt.scatter(xref, yref, c='red', s=1)
    plt.scatter(xnew, ynew, c='blue', s=1)

    plt.figure()
    plt.scatter(dat['Yfoc'], dat['Xfoc'], c='red', s=1)
    plt.scatter(y, x, c='blue', s=1)

    plt.show()


def test_find_neighborhoods():
    dat = read_data('./gb_optics_sim.csv')
    xref = dat['Xfoc']
    yref = dat['Yfoc']
    
    x = np.linspace(-10, 10, 11)
    y = np.linspace(-10, 10, 11)

    for xi, yi in zip(x, y):
        min_indices = find_neighborhoods(xi, yi, xref, yref)
        print(xi, yi, xref[min_indices], yref[min_indices])

        plt.figure()
        plt.scatter(xref, yref, c='b', s=3)
        plt.scatter(xref[min_indices], yref[min_indices], c='g', s=3)
        plt.scatter(xi, yi, c='r', s=3)

    plt.show()
    return




if __name__=="__main__":
    main2()
    #test_find_neighborhoods()
    
