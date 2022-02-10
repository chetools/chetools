import jax.numpy as jnp
import jax
import jax.tree_util as tu
from dotmap import DotMap
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
from jax.config import config
from copy import deepcopy
from scipy.sparse import coo_matrix
config.update("jax_enable_x64", True)
EPS = jnp.finfo(jnp.float64).resolution


class VSC():
    def __init__(self, c, model):
        self.model = model
        self.c = c.toDict() if isinstance(c,DotMap) else c
        self.c_flat, self.idx, self.shapes, self.tree = flatten(self.c)
        self.r = DotMap()
        self.rdf = None
        self.v = None
        self.vdf = None
        self.sdf = None
        self.cdf = None
        self.nan_var = make_nan_variables(self.c)
        self.nan_var_flat, *_ = flatten(self.nan_var)
        self.update_idx = jnp.where(jnp.isnan(self.nan_var_flat))
        self.x = self.c_flat[self.update_idx]
        # self.test=self.model(self.xtoc(self.x))

    def xtoc(self,x):
        c = self.c_flat.at[self.update_idx].set(x)
        return DotMap(unflatten(c, self.idx, self.shapes, self.tree))

    def xtovs(self,x):
        v_c = self.c_flat.at[self.update_idx].set(x)
        v_tree= unflatten(v_c, self.idx, self.shapes, self.tree)
        v,s = splitvs(self.c,v_tree)
        return v,s

    def transform(self, model):
        def model_f(x):
            res = model(DotMap(self.xtoc(x)))
            if isinstance(res,tuple):
                res=res[0]
            return jnp.squeeze(res)
        return model_f

    def solve(self, jit=True, verbosity=1, sparse=False):
        def constraints(x):
            res = self.model(DotMap(self.xtoc(x)))
            eq = jnp.array([])
            if type(res) is tuple:
                if type(res[0]) is list:
                    for i in range(len(res[0])):
                        eq=jnp.append(eq,jnp.atleast_1d(res[0][i]))
                else:
                    eq=jnp.append(eq,jnp.atleast_1d(res[0]))
            else:
                if type(res) is list:
                    for i in range(len(res)):
                        eq=jnp.append(eq,jnp.atleast_1d(res[i]))
                else:
                    eq=jnp.append(eq,jnp.atleast_1d(res))
            return eq

        if jit:
            constraints = jax.jit(constraints)

        jac = jax.jacobian(constraints)
        if jit:
            jac= jax.jit(jac)

        def sparse_jac(x):
            jac_x = jac(x)
            idx = jnp.where(jac_x>1e-11)
            return coo_matrix((jac_x[idx], idx))


        if sparse:
            nlc = NonlinearConstraint(constraints, 0.,0., jac=sparse_jac)
        else:
            nlc = NonlinearConstraint(constraints, 0.,0., jac=jac)


        def cb(xk):
            if verbosity > 0:
                print (constraints(xk))

        bounds = [(-25.,25.)]*self.x.size

        res = minimize(lambda x: 0., self.x, method='SLSQP', bounds=bounds, constraints=nlc, callback=cb,
                       tol=1e-8, options=dict(maxiter=1000))

        if verbosity > 1:
            print(res)
            print(self.model(DotMap(self.xtoc(res.x))))
        self.x = res.x

        self.v, self.s = self.xtovs(self.x)
        self.vdf = todf(self.v)
        self.sdf = todf(self.s)

        c=self.xtoc(self.x)
        self.cdf=todf(c)
        res = self.model(c)
        if type(res) is tuple:
            self.r = res[1]
            self.rdf= todf(self.r)



def make_nan_variables(d):
    d = d.toDict() if isinstance(d,DotMap) else d
    dd = deepcopy(d)
    for (k,v), (dk, dv) in zip(dd.items(), d.items()):
        if isinstance(v,dict):
            make_nan_variables(v)
        elif isinstance(v,Comp):
            dd[k]=Comp(jnp.nan*jnp.ones_like(v.x))
        elif isinstance(v,Range):
            dd[k]=Range(jnp.nan, 0.,1.)
        elif isinstance(v,RangeArray):
            dd[k]=RangeArray(jnp.nan * jnp.ones_like(v.x), 0.,1.)
    return dd

def splitvs(d,d2):
    resv={}
    resp={}
    for (dk,dv), (d2k,d2v) in zip(d.items(), d2.items()):
        flat, tree= tu.tree_flatten(dv, lambda _: True)
        all_leaves=tu.all_leaves(flat)
        is_unk = isinstance(flat[0],Unk)
        if all_leaves or is_unk:
            if is_unk:
                resv[dk]=d2v
            else:
                resp[dk]=d2v
        else:
            resv[dk],resp[dk]=splitvs(dv,d2v)
    return resv, resp


def flatten_dict(d, pre='',flat={}, sep='.'):
    pre=pre+sep if not(pre=='') else pre
    if isinstance(d,dict):
        for k,v in d.items():
            if not(type(v) in (tuple,list,dict)):
                flat[pre+f'{k}']=v
            else:
                flatten(v,pre+f'{k}')
    elif type(d) in (tuple,list):
        for count,item in enumerate(d):
            if not(type(item) in (tuple,list,dict)):
                flat[pre+f'{count}']=item
            else:
                flatten(item,pre+f'{count}')
    return flat 

def todf(d):
    flat = flatten_dict(d)
    df=pd.DataFrame()
    for k,v in flat.items():
        try:
            if isinstance(v,Unk):
                v=v.x
            li = list(v)
            d = {(f'Vector{len(li)}',idx+1):[v] for idx,v in enumerate(li)}
            df2 = pd.DataFrame(d, index=[k])
            df=df.append(df2)
        except TypeError:
            d={('Scalar','1'):[v]}
            df2 = pd.DataFrame(d,index=[k])
            df=df.append(df2)
    df=df.fillna("")
    return df

def flatten(pytree):
    vals, tree = jax.tree_flatten(pytree)
    shapes = [jnp.atleast_1d(val).shape for val in vals]
    vals2 = [jnp.atleast_1d(val).reshape([-1,]) for val in vals] # convert scalars to array to allow concatenation
    v_flat = jnp.concatenate(vals2)
    idx = list(jnp.cumsum(jnp.array([val.size for val in vals2])))
    return v_flat, idx, shapes, tree

def unflatten(x, idx, shapes, tree):
    return jax.tree_unflatten(tree, [(lambda item, shape: jnp.squeeze(item) if shape==(1,) else item.reshape(shape))(item,shape)
                                     for item,shape in zip(jnp.split(x,idx[:-1]), shapes)])

class Unk():
    def __init__(self):
        pass

class Comp(Unk):
    def __init__(self,x):
        self.x=jnp.asarray(x).reshape(-1)
        if self.x.size<2:
            raise ValueError('At least 2 components required')

    def __repr__(self):
        return f'{self.x}'

    @staticmethod
    def flatten(c):
        return jnp.log(c.x[:-1]) + jnp.log(1.+ (1. - c.x[-1])/c.x[-1]), None


    @staticmethod
    def unflatten(aux, q):
        q=jnp.squeeze(jnp.asarray(q)) #q may be a tuple that can't be squeezed
        xm1 = jnp.exp(q)/(1+jnp.sum(jnp.exp(q)))
        return jnp.concatenate((jnp.atleast_1d(xm1), jnp.atleast_1d(1.-jnp.sum(xm1))))


jax.tree_util.register_pytree_node(Comp, Comp.flatten, Comp.unflatten)

class RangeArray(Unk):
    def __init__(self,x, lo, hi):
        self.x=x
        self.lo = lo
        self.diff = hi-lo
        if jnp.any(self.diff <= 0.) or jnp.any(self.x<lo) or jnp.any(self.x>hi):
            raise ValueError('Hi > x > Lo is required')

    def __repr__(self):
        return f'{self.x}, lo={self.lo}, diff={self.diff}'

    @staticmethod
    def flatten(v):
        p = (v.x-v.lo)/v.diff
        return (jnp.log(p)-jnp.log(1.-p),), (v.lo,v.diff)

    @staticmethod
    def unflatten(aux, f):
        f=jnp.squeeze(jnp.asarray(f))
        return jax.nn.sigmoid(f)*aux[1]+aux[0]

jax.tree_util.register_pytree_node(RangeArray, RangeArray.flatten, RangeArray.unflatten)


class Range(Unk):
    def __init__(self,x, lo, hi):
        self.x=x
        self.lo = lo
        self.diff = hi-lo
        if self.diff <= 0. or self.x<lo or self.x>hi:
            raise ValueError('Hi > x > Lo is required')

    def __repr__(self):
        return f'{self.x}, lo={self.lo}, diff={self.diff}'

    @staticmethod
    def flatten(v):
        p = (v.x-v.lo)/v.diff
        return (jnp.log(p)-jnp.log(1.-p),), (v.lo,v.diff)

    @staticmethod
    def unflatten(aux, f):
        return jax.nn.sigmoid(f[0])*aux[1]+aux[0]

jax.tree_util.register_pytree_node(Range, Range.flatten, Range.unflatten)
