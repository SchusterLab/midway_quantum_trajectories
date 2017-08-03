from ODE87s_CPU import ODE87s, qutip_verification
from tqdm import tqdm_notebook
import numpy as np
#%matplotlib inline
#import matplotlib.pyplot as plt

complexX = np.complex128


def simulate():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", help="assign task id for this run",default='0')
    args = parser.parse_args()
    task_id = args.task_id

    
    dt=0.1
    T=100

    Hz=np.array([[1,0],[0,-1]],dtype=complexX)
    Hx=np.array([[0,1],[1,0]],dtype=complexX)
    Hy=np.array([[0,1j],[-1j,0]],dtype=complexX)
    psi0=np.array([[1],[0]],dtype=complexX)
    #psi0=np.array([[1],[1]],dtype=complexX)/np.sqrt(2)

    C_t1 = 0.5*np.array([[0,1],[0,0]],dtype=complexX)

    Cms = [C_t1]


    w0=2*np.pi*0.5
    H0=Hz*w0/2.
    Hks=[Hx,Hy]

    cqtpts=np.arange(0,T,dt)
    citpts=np.arange(0,T,dt)
    amp=0.1
    t0=50.
    sigma=10.
    cipts=amp*np.exp(-(citpts-t0)**2/(2*sigma**2))*np.cos(w0*citpts)
    cqpts=0.1*amp*(-(cqtpts-t0)*np.exp(-(cqtpts-t0)**2/(2*sigma**2)))*np.sin(w0*cqtpts)
    ukts=[citpts,cqtpts]
    ukjs=[cipts,cqpts]

    ode=ODE87s(H0, Cms, Hks, ukts, ukjs, psi0, dt, T, sparse = False, display_tqdm = False)
    tpts,psis=ode.integrate()
    
    np.save("../data/qubit/tpts",tpts)
    np.save("../data/qubit/psis_%s" %task_id, psis)
    
    
    return tpts, psis


if __name__ == "__main__":
    simulate()
