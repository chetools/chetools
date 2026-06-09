from collections import namedtuple
import jax
import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
jax.config.update("jax_enable_x64", True)
from copy import deepcopy
from functools import partial

import re
import requests
import string
from collections import deque
from itertools import combinations

import sympy
from sympy.abc import s
from sympy import Symbol, fraction, simplify, together, expand, Wild, exp
import scipy as sp
import scipy.signal as sig

from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_dark"
eps =  np.finfo(np.float64).eps

two_pi=2*np.pi
one_third = 1/3
R=8.314



extract_single_props = {'Molecular Weight' : 'Mw',
                        'Critical Temperature' : 'Tc',
                        'Critical Pressure' : 'Pc',
                        'Critical Volume' : 'Vc',
                        'Acentric factor' : 'w',
                        'Normal boiling point' : 'Tbn',
                        'IG heat of formation' : 'HfIG',
                        'IG Gibbs of formation' : 'GfIG',
                        'Heat of vaporization' : 'HvapNB'}

extract_coeff_props={'Vapor Pressure' : 'Pvap',
                     'Ideal Gas Heat Capacity':'CpIG',
                     'Liquid Heat Capacity' : 'CpL',
                     'Solid Heat Capacity' : 'CpS',
                     'Heat of Vaporization' : 'Hvap',
                     'Liquid Density' : 'rhoL'}

extract_poly_coeff_props={'Polynomial Ideal Gas Heat Capacity (cal/mol-K)':'polyCpIG'}

base_url = 'https://raw.githubusercontent.com/profteachkids/chetools/master/data/'

BIP_file = 'https://raw.githubusercontent.com/profteachkids/chetools/master/data/BinaryNRTL.txt'

class Props():
    def __init__(self,comps, get_NRTL=True, base_url=base_url, extract_single_props=extract_single_props,
                 extract_coeff_props=extract_coeff_props, BIP_file=BIP_file, suffix='Props.txt'):

        comps = [comps] if isinstance(comps,str) else comps
        self.N_comps = len(comps)

        id_pat = re.compile(r'ID\s+(\d+)')
        formula_pat = re.compile(r'Formula:\s+([A-Z0-9]+)')
        single_props_pat = re.compile(r'^\s+([\w \/.]+?)\s+:\s+([-.0-9e+]+) +([-\w/.()*]*) *$', re.MULTILINE)
        coeffs_name_pat = re.compile(r"([\w ]+)\s[^\n]*?Equation.*?Min(.*?)Coeffs:([- e\d.+]+)+?", re.DOTALL)
        coeffs_pat = re.compile(r'([-\de.+]+)')
        limits_pat = re.compile(r':\s*?([-e\d.+]+)+?')
        poly_coeffs_pat = re.compile(r"([- \/'()A-Za-z]*)\n Coefficients: +([-+e\d.+]*)\n* *([-+e\d.+]*)\n* *([-+e\d.+]*)\n* *([-+e\d.+]*)\n* *([-+e\d.+]*)\n* *([-+e\d.+]*)\n* *([-+e\d.+]*)")

        props_deque=deque()
        for comp in comps:
            res = requests.get(base_url+comp + suffix)
            if res.status_code != 200:
                raise ValueError(f'{comp} - no available data')
            text=res.text
            props={'Name': comp}
            units={}
            props['ID']=id_pat.search(text).groups(1)[0]
            props['Formula']=formula_pat.search(text).groups(1)[0]
            single_props = dict((item[0], item[1:]) for item in single_props_pat.findall(text))
            for k,v in extract_single_props.items():
                props[v]=float(single_props[k][0])
                units[v]=single_props[k][1]
                props[v] = props[v]*2.20462*1055.6 if units[v]=='Btu/lbmol' else props[v]
                props[v] = props[v]*6894.76 if units[v]=='psia' else props[v]
                props[v] = (props[v]-32)*5/9 + 273.15 if units[v] =='F' else props[v]

            coeffs_name_strings = {name:(limits,coeffs) for name, limits, coeffs in coeffs_name_pat.findall(text)}

            for k,v in extract_coeff_props.items():

                limits_string,coeffs_string = coeffs_name_strings[k]
                coeffs=coeffs_pat.findall(coeffs_string)
                for letter, value in zip(string.ascii_uppercase,coeffs):
                    props[v+letter]=float(value)
                limits = limits_pat.findall(limits_string)
                props[v+'_minT']=limits[0]
                props[v+'_maxT']=limits[2]
            poly_props = dict([(item[0], item[1:]) for item in poly_coeffs_pat.findall(text)])
            for k,v in extract_poly_coeff_props.items():
                for letter, value in zip(string.ascii_uppercase,poly_props[k]):
                    if value == '':
                        break
                    props[v+letter]=float(value)


            props_deque.append(props)

        for prop in props_deque[0].keys():
            # if self.N_comps>1:
            values = np.array([comp[prop] for comp in props_deque])
            # else:
            #     values = props_deque[0][prop]
            setattr(self,prop,values)


        # kmol to mol
        self.Vc = self.Vc/1000.
        self.HfIG = self.HfIG/1000.
        self.HfL = self.HfIG - self.Hvap(298.15)
        self.GfIG = self.GfIG/1000.
        self.HvapNB=self.HvapNB/1000.

        if (self.N_comps > 1) and get_NRTL:
            text = requests.get(BIP_file).text

            comps_string = '|'.join(self.ID)
            id_name_pat = re.compile(r'^\s+(\d+)[ ]+(' + comps_string +')[ ]+[A-Za-z]',re.MULTILINE)
            id_str = id_name_pat.findall(text)

            #maintain order of components
            id_dict = {v:k for k,v in id_str}
            # list of comp IDs with BIP dataq
            id_str = [id_dict.get(id, None) for id in self.ID]
            id_str_indices=list(filter(lambda x: False if x[0] is None else True, zip(id_str,range(len(id_str)))))
            id_str,indices=list(zip(*id_str_indices))
            comb_strs = combinations(id_str,2)
            comb_indices = combinations(indices,2)
            self.NRTL_A, self.NRTL_B, self.NRTL_C, self.NRTL_D, self.NRTL_alpha = np.zeros((5, self.N_comps,self.N_comps))
            start=re.search(r'Dij\s+Dji',text).span()[0]

            for comb_id, comb_index in zip(comb_strs, comb_indices):
                comb_str = '|'.join(comb_id)
                comb_values_pat = re.compile(r'^[ ]+(' + comb_str +
                                             r')[ ]+(?:' + comb_str + r')(.*)$', re.MULTILINE)


                match = comb_values_pat.search(text[start:])
                if match is not None:
                    first_id, values = match.groups(1)
                    #if matched order is flipped, also flip indices
                    if first_id != comb_id[0]:
                        comb_index = (comb_index[1],comb_index[0])
                    bij, bji, alpha, aij, aji, cij, cji, dij, dji  = [float(val) for val in values.split()]
                    np.add.at(self.NRTL_B, comb_index, bij)
                    np.add.at(self.NRTL_B, (comb_index[1],comb_index[0]), bji)
                    np.add.at(self.NRTL_A, comb_index, aij)
                    np.add.at(self.NRTL_A, (comb_index[1],comb_index[0]), aji)
                    np.add.at(self.NRTL_C, comb_index, cij)
                    np.add.at(self.NRTL_C, (comb_index[1],comb_index[0]), cji)
                    np.add.at(self.NRTL_D, comb_index, dij)
                    np.add.at(self.NRTL_D, (comb_index[1],comb_index[0]), dji)
                    np.add.at(self.NRTL_alpha, comb_index, alpha)
                    np.add.at(self.NRTL_alpha, (comb_index[1],comb_index[0]), alpha)

        self.eos_dict=dict(VDW = (0.,0.,1/8, 27/64, lambda Tr: 1),
                RK = (1., 0., 0.08664, 0.42748, lambda Tr: Tr**(-0.5)),
                SRK = (1., 0., 0.08664, 0.42748, lambda Tr: (1 + (0.48 + 1.574*self.w - 0.176*(self.w)**2)*(1-Tr**0.5))**2),
                PR = (1+jnp.sqrt(2), 1-jnp.sqrt(2), 0.07780, 0.45724, lambda Tr: (1+ (0.37464 + 1.54226*self.w - 0.26992*(self.w)**2)*(1-Tr**0.5))**2))

    def Pvap(self,T):
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        return jnp.squeeze(jnp.exp(self.PvapA[None,:] + self.PvapB[None,:]/T[:,None] + self.PvapC[None,:]*jnp.log(T[:,None]) +
                       self.PvapD[None,:]*jnp.power(T[:,None],self.PvapE[None,:])))

    def Tb(self, P):
        P=jnp.atleast_1d(jnp.squeeze(jnp.asarray(P)))
        return jnp.squeeze(1 / (1 / self.Tbn[None,:] -  R / self.HvapNB[None,:] * jnp.log(P[:,None] / 101325)))

    def CpIG(self, T):
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        CpIGCT = self.CpIGC[None,:]/T[:,None]
        CpIGET = self.CpIGE[None,:]/T[:,None]
        return jnp.squeeze(self.CpIGA[None,:] + self.CpIGB[None,:]*(CpIGCT/jnp.sinh(CpIGCT))**2 +
                self.CpIGD[None,:]*(CpIGET/jnp.cosh(CpIGET))**2)/1000

    def deltaHsensIGpoly(self, T):
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        return jnp.squeeze(T[:,None] * (self.polyCpIGA[None,:] + T[:,None]* (self.polyCpIGB[None,:] / 2 + T[:,None] * (self.polyCpIGC[None,:] / 3 +
            T[:,None] * (self.polyCpIGD[None,:] / 4 + T[:,None]* (self.polyCpIGE[None,:] / 5 + T[:,None]*self.polyCpIGF[None,:]/6)))))*4.184)

    def HIGpoly(self, nV, T):
        nV=jnp.atleast_2d(nV).reshape(-1,self.N_comps)
        return jnp.squeeze(jnp.sum(nV * (self.HfIG + self.deltaHsensIGpoly(T) - self.deltaHsensIGpoly(298.15)),axis=-1))


    def deltaHsensIG(self, T):
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        return jnp.squeeze(self.CpIGA[None,:]*T[:,None] + self.CpIGB[None,:] * self.CpIGC[None,:]/jnp.tanh(self.CpIGC[None,:]/T[:,None]) -
            self.CpIGD[None,:] * self.CpIGE[None,:] * jnp.tanh(self.CpIGE[None,:]/T[:,None]))/1000

    def HIG(self, nV, T):
        nV=jnp.atleast_2d(nV).reshape(-1,self.N_comps)
        return jnp.squeeze(jnp.sum(nV*(self.HfIG + self.deltaHsensIG(T) - self.deltaHsensIG(298.15)),axis=-1))


    def Hvap(self, T):
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        Tr = T[:,None]/jnp.atleast_1d(self.Tc)[None,:]
        return jnp.squeeze(self.HvapA[None,:]*jnp.power(1-Tr[:,None] , self.HvapB[None,:] + (self.HvapC[None,:]
            +(self.HvapD[None,:]+self.HvapE[None,:]*Tr[:,None] )*Tr[:,None] )*Tr[:,None] ))/1000.


    def deltaHsensL(self, T):
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        return jnp.squeeze(T[:,None] * (self.CpLA[None,:] + T[:,None] * (self.CpLB[None,:]/ 2 + T[:,None] * (self.CpLC[None,:] / 3 + T[:,None] *
             (self.CpLD[None,:] / 4 + self.CpLE[None,:] / 5 * T[:,None])))))/1000.

    def Hv(self, nV, T):
        nV=jnp.atleast_2d(nV).reshape(-1,self.N_comps)
        return jnp.squeeze(self.Hl(nV, T) + jnp.sum(nV*self.Hvap(T),axis=-1))

    def Hl(self, nL, T):
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        nL=jnp.atleast_2d(nL).reshape(-1,self.N_comps)
        return jnp.squeeze(jnp.sum(nL*(self.HfL + self.deltaHsensL(T) - self.deltaHsensL(298.15)),axis=-1))

    def rhol(self, T):
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        return jnp.squeeze(self.rhoLA[None,:] / jnp.power(self.rhoLB[None,:], 1+ jnp.power((1.-T[:,None]/self.rhoLC[None,:]),self.rhoLD[None,:])) *self.Mw[None,:])*1000

    def NRTL_gamma(self, x, T):
        x=jnp.atleast_2d(x).reshape(-1,self.N_comps)
        T=jnp.atleast_1d(jnp.squeeze(jnp.asarray(T)))
        tau = (self.NRTL_A[None,:,:] + self.NRTL_B[None,:,:] / T[:,None,None] + self.NRTL_C[None,:,:] * jnp.log(T[:,None,None]) +
               self.NRTL_D[None,:,:] * T[:,None,None])

        G = jnp.exp(-self.NRTL_alpha[None,:,:] * tau)
        xG=x[:,None,:] @ G
        xtauGdivxG = (x[:,None,:]@ (tau*G)/ xG)

        lngamma = jnp.squeeze(xtauGdivxG) +  jnp.squeeze(((G*(tau - xtauGdivxG))/xG) @x[:,:,None])
        return jnp.exp(lngamma)


    def NRTL_gammaOld(self, x, T):
        x=jnp.asarray(x).reshape(-1)
        tau = (self.NRTL_A + self.NRTL_B / T + self.NRTL_C * jnp.log(T) +
               self.NRTL_D * T)
        G = jnp.exp(-self.NRTL_alpha * tau)

        xG=x.T @ G
        xtauGdivxG = (x.T@ (tau*G)/ xG)
        lngamma = xtauGdivxG.flatten() +  (((G*(tau - xtauGdivxG))/xG) @x).flatten()
        return jnp.exp(lngamma)

    def Gex_RT(self, x,T):
        x=jnp.asarray(x).reshape(-1)
        tau = (self.NRTL_A + self.NRTL_B / T + self.NRTL_C * jnp.log(T) +
               self.NRTL_D * T)
        G = jnp.exp(-self.NRTL_alpha * tau)
        xG= x.T @ G
        xtauGdivxG = x.T @ (tau*G) / xG
        return (xtauGdivxG @ x)

    def Hex(self, x, T):
        return -R*(T**2)*jax.grad(self.Gex_RT,1)(x, T)

    def NRTL_gamma2Old(self,x, T):
        return jnp.exp(jax.grad(self.Gex_RT)(x,T))

    def cubic_Z(self, P,T,eos='SRK'):
        Pr, Tr = P/self.Pc, T/self.Tc
        s, e, omega, phi, alpha = self.eos_dict[eos]
        beta = omega * Pr / Tr
        q = phi* alpha(Tr)/omega/Tr
        return cubic_roots((e+s)*beta - beta -1,
                    e*s*beta**2 -(e+s)*beta*(beta+1) + q*beta,
                    -beta**2 * (e*s*beta + e*s + q))

    def cubic_phi_pure(self, P,T,eos='SRK',vapor=True):
        Z = self.cubic_Z(P,T,eos)
        Z = jnp.nanmax(Z,axis=1) if vapor else jnp.nanmin(Z,axis=1)

        Pr, Tr = P/self.Pc, T/self.Tc
        s, e, omega, phi, alpha = self.eos_dict[eos]

        bi = omega * R * self.Tc/self.Pc
        ai = phi*alpha(Tr)*(R**2) * (self.Tc**2)/self.Pc
        qi = ai/bi/R/T
        beta = bi*P/R/T

        I = jax.lax.cond(e==s, lambda _: beta/(Z+e*beta), lambda _: jnp.log((Z+s*beta)/(Z+e*beta))/(s-e), (1))
        return jnp.exp(Z-1 - qi*I)/(Z-beta)



    def cubic_phi_mix(self, n,P,T,eos='SRK',vapor=True):
        n=jnp.asarray(n)
        x=n/jnp.sum(n)
        Z = self.cubic_Z(P,T,eos)
        Z = jnp.nanmax(Z,axis=1) if vapor else jnp.nanmin(Z,axis=1)

        Pr, Tr = P/self.Pc, T/self.Tc
        s, e, omega, phi, alpha = self.eos_dict[eos]

        bi = omega * R * self.Tc/self.Pc
        ai = phi*alpha(Tr)*(R**2) * (self.Tc**2)/self.Pc
        beta = bi*P/R/T

        I = jax.lax.cond(e==s, lambda _: beta/(Z+e*beta), lambda _: jnp.log((Z+s*beta)/(Z+e*beta))/(s-e), (1))
        bmix = bmix_calc(n,bi)
        amix = amix_calc(n,ai)
        qmix = amix/bmix/R/T
        abar = namix_grad(n,ai)
        qbar = qmix*(1 + abar/amix - bi/bmix)

        return jnp.exp(bi*(Z-1)/bmix - jnp.log(Z-beta) - qbar*I)

def amix_calc(n,a):
    x=n/jnp.sum(n)
    amat = jnp.sqrt(a[None,:]*a[:,None])
    return x.dot(amat@x)

def bmix_calc(n,b):
    x=n/jnp.sum(n)
    return x.dot(b)

def namix_calc(n,a):
    return jnp.sum(n)*amix_calc(n,a)

namix_grad = jax.grad(namix_calc,0)

def qtox(q):
    q=jnp.atleast_1d(q)
    xm1 = jnp.exp(q)/(1+jnp.sum(jnp.exp(q)))
    return jnp.concatenate((xm1, jnp.atleast_1d(1.-jnp.sum(xm1))))

def xtoq(x):
    x=jnp.atleast_1d(x)
    return jnp.log(x[:-1]) + jnp.log(1.+ (1. - x[-1])/x[-1])



@partial(jnp.vectorize, signature='(),(),()->(3)')
def cubic_roots(a, b, c):
    # Returns only the real roots of cubic equations with real coefficients
    # x**3 + a x**2 + b x + c = 0

    Q = (a * a - 3 * b) / 9
    R = (2 * a * a * a - 9 * a * b + 27 * c) / 54
    det = (R * R - Q ** 3)

    def roots3(v):
        theta = jnp.arccos(R / pow(Q, 1.5))
        x=jnp.array((jnp.cos(theta/3), jnp.cos((theta+two_pi)/3), jnp.cos((theta-two_pi)/3)))
        x = -2 * jnp.sqrt(Q)*x - a/3
        return x

    def roots1(v):
        A = -jnp.sign(R) * (abs(R) + jnp.sqrt(det)) ** one_third
        B = Q / A
        return jnp.array([(A + B) - a / 3, jnp.nan, jnp.nan])

    return jax.lax.cond(det < 0, roots3, roots1, (1))

def pid(K, taui=None, taud=None):
    if (taui is None) and (taud is None):
        return K
    if taud is None:
        return K*(1 + 1/(taui*Symbol('s')))
    return K*(1 + 1/(taui*s) + taud*s)

def TF1(K, tau):
    return K/(tau*s + 1)

def TF2(K, wn, z):
    return K*(wn**2)/(s**2 + 2*z*wn*s + wn**2)

def shift(f=None, theta=0):
    f=(lambda t: 1) if f is None else f
    return lambda t: (t>=theta)*f(t-theta)

def get_cpe(sys):
    n,d=fraction(together(simplify(sys)))

    a=Wild('a')
    b=Wild('b')
    c=Wild('c')
    exps = sympy.Symbol('exps')

    def cpe(expr):
        res=[]
        expr=expr.replace(exp(c*s), exps**c)
        expr= expr.expand()

        def process_term(term):
            d = term.match(a*(s**b)*(exps**c))
            res.append([d[a],d[b],d[c]])

        if expr.func!=sympy.core.mul.Add:
            process_term(expr)
        else:
            for term in expr.args:
                process_term(term)
        return np.array(res).astype(float)

    cpe_n = cpe(n)
    cpe_d = cpe(d)
    max_exponent = np.max(np.r_[cpe_n[:,-1], cpe_d[:,-1]])
    cpe_n[:,-1]-=max_exponent
    cpe_d[:,-1]-=max_exponent
    return cpe_n[np.lexsort([cpe_n[:,1],cpe_n[:,2]])[::-1]],cpe_d[np.lexsort([cpe_d[:,1],cpe_d[:,2]])[::-1]]

def get_ABC(n,d):

    def get_coeffs(nd, d_size):
        thetas = np.sort(np.unique(nd[:,-1]))[::-1]
        coeffs = np.zeros((thetas.size,d_size))
        for i,theta in enumerate(thetas):
            c = nd[nd[:,-1]==theta,0]
            p = nd[nd[:,-1]==theta,1]
            coeffs[i,p.astype(int)]=c
        return coeffs[:,::-1],thetas

    d_size = int(np.max(d[:,1]))+1
    d_coeffs, d_thetas = get_coeffs(d, d_size)
    n_coeffs, n_thetas = get_coeffs(n, d_size)
    norm = d_coeffs[0, np.argmax(d[d[:,-1]==0,1])]
    n_coeffs = n_coeffs/norm
    d_coeffs = d_coeffs/norm

    A = np.eye(d_coeffs.shape[1]-1,k=1)
    A[-1,:]=-d_coeffs[0,-1:0:-1]
    B=np.zeros(d_coeffs.shape[1]-1)
    B[-1]=1
    return A, B, n_coeffs, n_thetas, d_coeffs, d_thetas

def sim(sys, u, N=100, dt=0.1):
    n,d=get_cpe(sys)
    A,B, n_coeffs, n_thetas, d_coeffs, d_thetas = get_ABC(n,d)
    max_delay = np.min(np.r_[n_thetas, d_thetas])
    n_coeffs=n_coeffs[:,::-1]  #y history is stored in order from 0th to nth derivative.
    d_coeffs=d_coeffs[:,::-1]
    n_states=A.shape[1]
    n_inputs=np.atleast_1d(u(0)).size
    t_hist=np.full(N+2,np.inf)
    y_hist=np.zeros((N+2,n_states))
    t_hist[:2]=[max_delay,0]
    z=np.zeros(N)

    def hist_terms(t, coeffs, thetas):
        i=np.searchsorted(t_hist,t+thetas, side='right')-1
        yold= ((t+thetas)[:,None]-t_hist[i,None]) * (y_hist[i+1,:]-y_hist[i,:]) / (t_hist[i+1,None]-t_hist[i,None]) + y_hist[i,:]
        return np.sum(yold*coeffs)

    M1=np.c_[A*dt, B*dt, np.zeros((n_states, n_inputs))]
    M2=np.c_[np.zeros((n_inputs, n_states + n_inputs)),np.identity(n_inputs)]
    M3=np.zeros((n_inputs, n_states + 2 * n_inputs))

    M=np.r_[M1,M2,M3]
    expmat = sp.linalg.expm(M)
    Ad = expmat[:n_states, :n_states]
    B1d = expmat[:n_states, n_states:n_states+n_inputs]
    B2d = expmat[:n_states, n_states+n_inputs:]

    y=y_hist[1]
    v=hist_terms(0,d_coeffs[1:,:-1],d_thetas[1:])
    uvold = np.atleast_1d(u(0) - v)
    for i in range(1,N):
        t=i*dt
        v=hist_terms(t,d_coeffs[1:,:-1],d_thetas[1:])
        uvnew = np.atleast_1d(u(t)-v)
        y=y @ Ad.T + uvnew@B1d.T + (uvnew-uvold) @ B2d.T
        t_hist[i+1]=t
        y_hist[i+1]=y
        uvold=uvnew


    for i,t in enumerate(t_hist[1:-1]):
        z[i]=hist_terms(t, n_coeffs[:,:-1], n_thetas)
    return t_hist[1:-1], z[:i+1]

def rhs2TF(f, x0, u0, ios=None):
    A=np.array(jax.jacobian(f,argnums=1)(0,x0,u0))
    B=np.array(jax.jacobian(f,argnums=2)(0,x0,u0))
    tfs=[]
    den = np.poly(A)
    #(sI-A)X = B U
    #det[a1+b, a2, a3] = det[a1, a2, a3] + det[b, a2, a3]
    if ios is None:
        tfs = sympy.zeros(x0.size,u0.size)
        for xi in range(x0.size):
            for ui in range(u0.size):
                A2=A.copy()
                A2[:,xi]-=B[:,ui]
                num = np.poly(A2)-den

                tfs[xi,ui]=(sympy.Poly(np.round(num,12),s)/sympy.Poly(np.round(den,12),s)).as_expr().subs(1.0,1)
    else:
        ios=np.atleast_2d(ios)
        for xi, ui in ios:
            A2=A.copy()
            A2[:,xi]-=B[:,ui]
            num = np.poly(A2)-den
            tfs.append((sympy.Poly(np.round(num,12),s)/sympy.Poly(np.round(den,12),s)).as_expr().subs(1.0,1))
    return tfs

def bode(tf, w=None):
    f=sympy.utilities.lambdify(s,tf)
    w=np.logspace(-4,4,200) if w is None else w
    bode=f(w*1j)
    mag,angle = np.abs(bode), np.angle(bode)
    angle=np.unwrap(angle)*180/np.pi
    fig=make_subplots(rows=2,cols=1)
    fig.add_scatter(x=w, y=mag, row=1, col=1)
    fig.add_scatter(x=w,y=angle,row=2, col=1)
    fig.update_xaxes(type="log", dtick=1, row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(type="log", dtick=1, row=2, col=1)
    fig.update_yaxes(dtick=180, row=2, col=1)
    fig.update_layout(width=500, height=600, template='plotly_dark', showlegend=False)
    return fig, mag, angle