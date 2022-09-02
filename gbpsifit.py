import numpy as np
import pylab as plt

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


def get_psi_fit_old():
    dat = read_data('./gb_optics_sim.csv')
     
    xfoc = dat['Xfoc']
    yfoc = dat['Yfoc']

    theta = dat['theta']
    phi   = dat['phi']

    dpsi = dat['omtffr'] - dat['omtfoc']

    xp, yp = proj_thetaphi(theta, phi)
 
    xs = list(set(xfoc))
    xs.sort()
    idx_full = np.arange(len(dat))

    ps = []
    dyfit = []
    dy2fit = []
    for x0 in xs:
        idx = idx_full[xfoc==x0]
        psi0 = dpsi[idx]
        xp0 = xp[idx]
        yp0 = yp[idx]

        p = np.polyfit(xp0, psi0, deg=3)
        #print (p)
        ps.append(p)
        y = np.poly1d(p)
        x = np.linspace(-0.25, 0.25, 1001)
        plt.scatter(xp0, psi0, s=3)
        label='{:0.5e}$x^3$ {:+0.5e}$x^2$ {:+0.5e}$x$ {:+0.5e}'.format(*p)
        print (f'xfoc={x0:+02.2f}\t', 'psi=', label)
        plt.plot(x, y(x), linewidth='0.5', label=label)
        dyfit.append(sum((y(xp0) - psi0)))
        dy2fit.append(sum((y(xp0) - psi0)**2))

    plt.xlabel('x')
    plt.ylabel('psi')
    
    ps = np.array(ps).T
    print (ps[0])
    plt.figure()

    pf = []
    for p in ps:
        deg = 2
        p1 = np.polyfit(xs, p, deg=deg)
        pf.append(p1)
        y = np.poly1d(p1)
        x = np.linspace(-100, 100, 1001)
        fmt = ''
        for i in range(deg+1)[::-1]:
            fmt += '{:0.5e}$x^%d$ + ' % i

        fmt = fmt[:-8]
        label = fmt.format(*p1)
        plt.plot(xs, p, '*')
        plt.plot(xs, y(xs), label=label)

    pf = np.array(pf).T
    
    plt.xlabel('Xfoc')
    plt.ylabel('Fit parameters')
    plt.legend()
        
    dyfitfit = []
    dy2fitfit = []
    p0avg = np.average(ps[0])
    ps1 = []
    plt.figure()
    for x0 in xs:
        idx = idx_full[xfoc==x0]
        psi0 = dpsi[idx]
        xp0 = xp[idx]
        yp0 = yp[idx]

        p = np.polyfit(xp0, psi0, deg=3)
        p[0] = ps[0,0]
        ps1.append(p)
        y = np.poly1d(p)
        x = np.linspace(-0.55, 0.55, 1001)
        plt.scatter(xp0, psi0, s=3)
        label='{:+0.5e}$x^3$ {:+0.5e}$x^2$ {:+0.5e}$x$ {:+0.5e}'.format(*p)
        print (f'xfoc={x0:+02.2f}\t', 'psi=', label)
        plt.plot(x, y(x), linewidth='0.5', label=label)
        dyfitfit.append(sum((y(xp0) - psi0)))
        dy2fitfit.append(sum((y(xp0) - psi0)**2))

    plt.xlabel('x')
    plt.ylabel('psi')
    
    print ('before fixing p[0]')
    print (sum(dyfit))
    print (sum(dy2fit))

    print ('after fixing p[0]')
    print (sum(dyfitfit))
    print (sum(dy2fitfit))

    plt.show()  


def get_psi_fit():
    dat = read_data('./gb_optics_sim.csv')
     
    xfoc = -dat['Yfoc']
    yfoc = dat['Xfoc']
    theta = dat['theta']
    phi   = dat['phi']
    dpsi = dat['omtffr'] - dat['omtfoc']
    xp, yp = proj_thetaphi(theta, phi)
 
    ys = list(set(yfoc))
    ys.sort()
    idx_full = np.arange(len(dat))
    
    def totalresidual(p3, silence=True):
        ps = []
        dyfit = []
        dy2fit = []
        for y0 in ys:
            idx = idx_full[yfoc==y0]
            psi0 = dpsi[idx]
            xp0 = xp[idx]
            yp0 = yp[idx]
            xfoc0 = xfoc[idx]

            def fitfnc(c1, c3):
                y = lambda x: c3 * x**3 + c1 * x**1
                res = sum((y(xfoc0)- psi0)**2)
                return res 

            p30 = p3
            pres = minimize(fitfnc, (5), args=(p3))
            p = pres.x
            y = lambda x: p3 * x**3 + p[0] * x**1
            dyfit.append(sum((y(xfoc0) - psi0)))
            dy2fit.append(sum((y(xfoc0) - psi0)**2))
            ps.append(p)

            ## visualize
            if not silence:
                x = np.linspace(-100, 100, 1001)
                plt.scatter(xfoc0, psi0, s=3)
                label='{:0.5e}$x^3$ {:+0.5e}$x$'.format(p3, p[0])
                #print (f'xfoc={x0:+02.2f}\t', 'psi=', label)
                plt.plot(x, y(x), linewidth='0.5', label=label)


        if silence:
            return sum(dy2fit)
        else:
            print (sum(dy2fit), np.mean(dy2fit), np.std(dy2fit))
            return sum(dy2fit), ps

    #p1 = minimize(totalresidual, -6.726, tol=1e-9)
    #print (p1)

    ## using lmfit
    params = lmfit.Parameters()
    params.add('p3', 1.0)
    mini = lmfit.Minimizer(totalresidual, params)
    p1 = mini.minimize()
    lmfit.report_fit(p1)
    #ci = lmfit.conf_interval(mini, p1)
    #lmfit.printfuncs.report_ci(ci)

    print('residual', p1.residual)
    #res, ps = totalresidual(-6.727, silence=False)
    res, ps = totalresidual(-7.4290e-8, silence=False)
    
    plt.xlabel('x')
    plt.ylabel('psi')
    
    ps = np.array(ps).T
    plt.figure()

    pf = []
    for p in ps:
        deg = 2
        p1 = np.polyfit(ys, p, deg=deg)
        pf.append(p1)
        y = np.poly1d(p1)
        x = np.linspace(-100, 100, 1001)
        fmt = ''
        for i in range(deg+1)[::-1]:
            fmt += '{:0.5e}$x^%d$ + ' % i

        fmt = fmt[:-8]
        label = fmt.format(*p1)
        plt.plot(ys, p, '*')
        plt.plot(ys, y(ys), label=label)

    pf = np.array(pf).T
    
    plt.xlabel('Xfoc')
    plt.ylabel('Fit parameters')
    plt.legend()
    #print ('before fixing p[0]')

    #print (sum(dyfit))
    #print (sum(dy2fit))

    print (poly_arr2tex(pf.T[0]))
    print (pf.T[0])
     

    plt.show()  


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
    get_psi_fit()


