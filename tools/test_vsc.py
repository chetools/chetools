from dotmap import DotMap
import pandas as pd
import jax
import jax.numpy as jnp
from jax.config import config
from jax.experimental.host_callback import id_print
config.update("jax_enable_x64", True) #JAX default is 32bit single precision

from tree_array_transform2 import VSC, Comp, Range
import che


def model(c):
    # c: combination of adjustable variables and static state parameters
    # r: DotMap - store intermediate results for reporting
    r=DotMap()
    r.V = c.Vy * c.Vtot # Moles of each component = mole fractions * total moles
    r.L = c.Lx * c.Ltot
    r.F = c.Fz * c.Ftot
    mass_balance = r.F - r.V - r.L # Mass balance for each component (vectors!)

    # Hmix calculates the enthalpy given the temperature and moles of each
    # component in the vapor and liquid phases
    r.FH = p.Hl(nL=r.F, T=c.FT)
    r.VH = p.Hv(nV=r.V, T=c.flashT)
    r.LH = p.Hl(nL=r.L, T=c.flashT)
    energy_balance = (r.FH - r.VH - r.LH)

    # Raoults with NRTL activity coefficient correction.  One-liner!
    r.fugL = c.Lx  * p.NRTL_gamma(c.Lx,c.flashT)* p.Pvap(c.flashT)
    r.fugV = c.Vy*c.flashP
    VLE = r.fugL - r.fugV
    id_print([mass_balance, energy_balance, VLE])
    return [mass_balance, energy_balance, VLE], r

p = che.Props(['Ethanol','Isopropanol', 'Water'])
c=DotMap()
c.Ftot=10 # Total Feed moles
c.Fz = jnp.array([1/3, 1/3, 1/3]) # Equimolar feed composition
c.FT = 450 # Feed temperature
c.flashP= 101325 # Flash drum pressure

c.Vy = Comp(c.Fz) # Guess vapor/liquid composition equal to feed
c.Lx = Comp(c.Fz) # Comp - constrains mole fractions to behave like mole fractions!
c.flashT = Range(360, 273.15, c.FT)  # Guess and bounds for flash temperature
c.Vtot = Range(c.Ftot/2, 0., c.Ftot)  # Guess half of feed in vapor
c.Ltot = Range(c.Ftot/2, 0., c.Ftot)

vsc=VSC(c, model)
vsc.solve(jit=True, verbosity=0)

print(vsc.vdf)
print(vsc.sdf)
print(vsc.cdf)

