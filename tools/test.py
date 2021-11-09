import che
import jax.numpy as jnp
import jax

p=che.Props(['Ethanol','Water'])
print(p.NRTL_B)
w = jnp.array([95.63, 4.37])
x = w/p.Mw / sum(w/p.Mw)
T=78.2+273.15
P=101325
g2 = p.NRTL_gamma2(x,T)
print(g2)
print(x*g2*p.Pvap(T), x*P)
g1 = p.NRTL_gamma(x,T)
print(g1)
