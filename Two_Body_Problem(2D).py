import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


import scipy as sci
import scipy.integrate as sp_int



#Define universal gravitation constant
G=6.67408e-11 #N-m2/kg2
#Reference quantities
m_nd=1.989e+30 #kg #mass of the sun
r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
v_nd=30000 #m/s #relative velocity of earth around the sun
t_nd=79.91*365*24*3600*0.51 #s #orbital period of Alpha Centauri
#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd
m1=1.1 #Alpha Centauri A
m2=0.9 #Alpha Centauri B


r1 = [-0.5,0,0]  
r2 = [0.5,0,0]
v1 = [0.05,0,0.1]
v2= [-0.05,0,-0.1]

r1 = np.array(r1)
r2 = np.array(r2)
v1 = np.array(v1)
v2 = np.array(v2)



def TwoBodyEquations(w,t,G,m1,m2):

    r1 = w[:3]
    r2 = w[3:6]
    v1 = w[6:9]
    v2 = w[9:12]


    r = sci.linalg.norm(r2-r1)


    a_1 = K1*m2*(r2-r1)/r**3 #dv1dt
    a_2 = K1*m1*(r1-r2)/r**3 # dv2dt
    v_1 = K2*v1 #dr1dt
    v_2 = K2*v2 # dr2dt

    r_derivs = np.concatenate((v_1,v_2))
    derivs = np.concatenate((r_derivs,a_1,a_2))
    return derivs


def norm(v):
    x = []
    for i in range(0,len(v)):
        c = np.linalg.norm(v[i,:3])
        x.append(c)
    return x

init_params = sci.array([r1,r2,v1,v2]).flatten()
time_span = np.linspace(0,16,501)
t = np.arange(0,501)

two_body_sol = sp_int.odeint(TwoBodyEquations, init_params, time_span, args = (G,m1,m2))


r1_sol = two_body_sol[:,:3]
r2_sol = two_body_sol[:,3:6]
v1_sol = two_body_sol[:,6:9]
v2_sol = two_body_sol[:,9:12]

fig = plt.figure(figsize=plt.figaspect(2.))

ax = fig.add_subplot(2,2,1)
ax.set_xlabel("x",fontsize=14)
ax.set_ylabel("z",fontsize=14)
ax.set_title("Visualization of orbits of stars in a 2-body system\n",fontsize=16)
ax.legend(loc="upper left",fontsize=14)


bx = fig.add_subplot(2,2,3)
bx.set_xlabel("time")
bx.set_ylabel("Distance")
bx.set_xlim(0, 500)

cx=fig.add_subplot(2,2,4)
cx.set_xlabel("time")
cx.set_ylabel("Distance")
cx.set_xlim(0, 500)

h1 = [ax.scatter(r1_sol[0,0],r1_sol[0,2],color="darkblue",marker="o",s=80,label="Alpha Centauri A")]
h2 = [ax.scatter(r2_sol[0,0],r2_sol[0,2],color="tab:red",marker="o",s=80,label="Alpha Centauri B")]

V_1 = norm(v1_sol)
V_2 = norm(v2_sol)

def Animate_2b(i,head1,head2):
        #Remove old markers
    h1[0].remove()
    h2[0].remove()
       
        # Plotting the orbits (for every i, we plot from init pos to final pos)
    t1 = ax.plot(r1_sol[:i,0],r1_sol[:i,2],color='darkblue')
    t2 = ax.plot(r2_sol[:i,0],r2_sol[:i,2],color='r')
       
        # Plotting the current markers
    h1[0]=ax.scatter(r1_sol[i,0],r1_sol[i,2],color="darkblue",marker="o",s=80)
    h2[0]=ax.scatter(r2_sol[i,0],r2_sol[i,2],color="r",marker="o",s=80)


    bx.plot(t[:i],V_1[:i],color = 'b')
    cx.plot(t[:i],V_2[:i],color = 'r')

    return t1,t2,h1,h2

plt.show()  

anim_2b = animation.FuncAnimation(fig,Animate_2b,frames=750,interval=300,repeat=False,blit=False,fargs=(h1,h2))
FFwriter = animation.FFMpegWriter(fps=30, bitrate=4000, metadata=dict(artist = "me"))
anim_2b.save("twobodyproblem2.mp4",writer = FFwriter)

