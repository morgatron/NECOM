import numpy as np
from statsmodels import api as sm
from scipy import optimize as opt


glb_fit_history = [list() for k in range(4) ]

"""
m[0], m[1], p[0], p[1] (Copied from notebook)
"""
def get_Ox_p(a, b=0, c=0):
    mat = [ [c,1],
           [a,b]
           ]
    return np.array(mat).flatten()[:,None]

def get_Oz_p(a, b=0, c=0):
    mat = [ [1,c],
           [b,a]
           ]
    return np.array(mat).flatten()[:,None]


# hardcode values here
M_x = None 
M_z = None 

def set_subtraction_params(p_x, p_z):
    global M_x, M_z
    M_x = get_Ox_p(*p_x)
    M_z = get_Oz_p(*p_z)

set_subtraction_params([0.3,0,0],
                       [-0.35, 0.02, -0.32]);
#set_subtraction_params([-0.126, -0.031, -0.179],
#                       [0.314, -0.034, -0.44]);

def calc_anom_sigs(fitted_m, fitted_p):
    print(fitted_m.shape)
    A = np.array([ fitted_m[:,:2].T, 
                  fitted_p[:,:2].T]).reshape(4,-1)
    anom_x =  (M_x*A).sum(axis=0) 
    anom_z =  (M_z*A).sum(axis=0) 
    #print(anom_x.shape, anom_z.shape)
    return anom_x, anom_z


#(PM is for plus/minus)
def full_dict_to_pms(full_d):
    #Npts = list(d.values())[0].size

    full_arr = np.array(full_d.values())
    Npts = full_arr.shape[-1]
    arr_p, arr_m = full_to_pm(full_arr)
    return dict(zip(full_d.keys(), arr_p)) , dict(zip(full_d.keys(), arr_m))
    #{key: full_to_pm(sig) for sig, key in d.items()}
    #y1, y2 = traces[...,:Npts//2], traces[...,-Npts//2:] 
    #return y1+y2, y1-y2

def full_to_pm(traces):
    Npts = traces.shape[-1]
    y1, y2 = traces[...,:Npts//2], traces[...,-Npts//2:] 
    return y1+y2, y1-y2


#fit plus/minus with plus sigs
def fit_plus_minus_with_plus(full_traces, full_signatures, bForceRecalc=True):
    """ Split signals into +/- sections, and fit each seperately.
    
    The first time it's run it'll calcualte the signatures, 
    but it shouldn't redo it unless signature change

    @remake_models_from_sigs is called only when necessary and uses side effects
    @fit_sigs takes in the new traces prior to splitting, as well as pre-made models to use for fitting. 
     (it's a pure function)

    """
    if len(full_traces)<1 or full_traces is None:
        return []
    self = fit_plus_minus_with_plus
    full_traces = np.array(full_traces)

    def remake_models_from_sigs(sigD):
        # Split up signatures into plus/minu

        # could consider grabbing pump_Phi here too
        print(sigD.keys())
        d = {key:sigD[key] for key in ['Bx', 'Bz']}
        full_arr = np.vstack(d.values())
        #Npts = full_arr.shape[-1]
        arr_p, arr_m = full_to_pm(full_arr)

        exog_p= sm.add_constant(arr_p.T, prepend=False) #THIS MAY NOT WANT THE .T???
        exog_m = arr_p.T.copy()
        #exog_m = exog_p.copy()

        self.exog_m = exog_m
        self.exog_p = exog_p

        model_m = sm.OLS(exog_m[:,0].copy(), exog_m)#, missing='drop')
        model_p = sm.OLS(exog_p[:,0].copy(), exog_p)#, missing='drop')

        self.model_m = model_m
        self.model_p = model_p
        
        # same for p...

    def fit_sigs(full_traces, model_p, model_m):
        traces_p, traces_m = full_to_pm(full_traces)

        res_pL = []
        for trace in traces_p:
            model_p.endog[:]=trace 
            res_pL.append(model_p.fit().params)
        res_mL = []
        for trace in traces_m:
            model_m.endog[:]=trace 
            res_mL.append(model_m.fit().params)
            print()

        res_pA = np.array(res_pL)
        res_mA = np.array(res_mL)
        #resD = {label: vals for label, vals in zip(['dc']+list(signatures.keys()), resA.T)}
        return res_pA, res_mA
        
    if bForceRecalc or not hasattr(self, "model_m"):
        remake_models_from_sigs(full_signatures)
    try:
        fitted_p, fitted_m = fit_sigs(full_traces, self.model_p, self.model_m)
    except:
        remake_models_from_sigs(full_signatures)
        fitted_p, fitted_m =fit_sigs(full_traces, self.model_p, self.model_m)

    anom_x, anom_z = calc_anom_sigs(fitted_m, fitted_p)
    #anom_x, anom_z = fitted_p[:,:2].T

    # NOW USE fitted_p AND fitted_m 
    # 
    resD= {
        'dc_p' : fitted_p[:,-1],
        #'dc_m' : fitted_m[:,-1],
        "x+": fitted_p[:,0],
        "z+": fitted_p[:,1],
        "x-": fitted_m[:,0],
        "z-": fitted_m[:,1],
        "anom_x": anom_x,
        "anom_z" : anom_z,
    }
    for k,val in enumerate([*(fitted_p[:,:2].T), *(fitted_m[:,:2].T)]):
        l = glb_fit_history[k]
        l.extend(val)
        N = 3900*16
        if len(l)> N:
            glb_fit_history[k] = l[-N:]
    #resD= {
    #    'dc' : fitted_p[:,-1],
    #    "anom_x": anom_x,
    #    "anom_z" : anom_z,
    #}
    return resD
    # p is electron response + nuclear response
    # m is nuclear response only (scaled by pulse size)
    # electron resp is p - scl*m
    # nuclear resp is probably best given by m alone (?)
    # mag resp is electron resp 

from box import Box
spu = Box()
spu.set_subtraction_params=set_subtraction_params
spu.calc_anom_sigs = calc_anom_sigs
def calc_subtraction_params(fitted_m, fitted_p, period_x, period_z, period_fast_x, period_fast_z):
    
    tAx = np.arange(fitted_m.shape[0]) 
    def power_with_period(sig, period):
        return np.mean(sig*np.sin(2*np.pi*tAx/period))**2 + np.mean(sig*np.cos(2*np.pi*tAx/period))**2

    def f(p):
        spu.set_subtraction_params([p[0],p[2], p[4]], [p[1],p[3], p[5]])
        anom_z, anom_x = spu.calc_anom_sigs(fitted_m, fitted_p)
        err_x = power_with_period(anom_x, period_x) + power_with_period(anom_x, period_z)
        err_z = power_with_period(anom_z, period_x) + power_with_period(anom_z, period_z)
        sig_x = power_with_period(anom_x, period_fast_x)
        sig_z = power_with_period(anom_z, period_fast_z)
        print(f"{p}, {sig_x/err_x}, {sig_z/err_z}, {err_x+err_z} | {err_x/sig_x+err_z/sig_z}")
        return err_x/sig_x+err_z/sig_z
        #return err_x+err_z
    res = opt.minimize(f, [0.05,-0.05,-0.05,0.05, 0.01, -0.01], bounds = [(-1,1), (-1,1), (-.8,.8),(-.8,.8), (-1, 1), (-1, 1)],method='Powell',
                       options={"gtol":1e-12, "eps":1e-6, 'ftol':1e-14} )
    p = res.x
    spu.set_subtraction_params([p[0],p[2], p[4]], [p[1],p[3], p[5]])
    anom_z, anom_x = spu.calc_anom_sigs(fitted_m, fitted_p)
    print(res)
    return res, anom_z, anom_x



def update_sub_params():
    res, ax, az=calc_subtraction_params(np.array(glb_fit_history[2:]).T, np.array(glb_fit_history[:2]).T, 2600, 3900,8,10)
    return ax, az






