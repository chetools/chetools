import che
import jax.numpy as jnp
import jax

# p=che.Props(['Ethanol','Water'])
p=che.Props(['Water'])
# print(p.NRTL_B)
# w = jnp.array([95.63, 4.37])
# x = w/p.Mw / sum(w/p.Mw)
# T=78.2+273.15
# P=101325

# g2 = p.NRTL_gamma2(x,T)
# # g2 = p.Gex(x,T)
# print(g2)
# print(x*g2*p.Pvap(T), x*P)
# g1 = p.NRTL_gamma(x,T)
# print(g1)

print('---------')
T1=jnp.array([280, 350,400])
T2=400
T3=[350.]
T4=[350., 400.]
T5=[[300., 350.],[375., 400.]]
x=jnp.array([[0.3,0.7],[0.4,0.6],[0.5,0.5]])

n1 = jnp.array([[1.,1.],[2.,3.],[1.,1.]])
n2=[1.,1.]
print(p.rhol(T1))
print(p.rhol(T2))
print(p.rhol(T3))
print(p.rhol(T4))
print()
print(p.rhol2(T5))
print(p.Mw)