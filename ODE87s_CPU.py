import numpy as np
from scipy import sparse
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

complexX = np.complex128



def _force_forder(x):
    """
    Converts arrays x to fortran order. Returns
    a tuple in the form (x, is_transposed).
    """
    if x.flags.c_contiguous:
        return (x.T, True)
    else:
        return (x, False)


def fdot(A, B):
    """
    Uses cpu blas libraries directly to perform dot product
    """
    from scipy import linalg

    A, trans_a = _force_forder(A)
    B, trans_b = _force_forder(B)
    gemm_dot = linalg.get_blas_funcs("gemm", arrays=(A, B))

    # gemm is implemented to compute: C = alpha*AB  + beta*C
    return gemm_dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)

def nphermitian(a):
    return a.T.conj()


class ODE87s:
    #### Coefficients for RK87 method
    c = np.array([0,1./18, 1./12, 1./8, 5./16, 3./8, 59./400, 93./200, 5490023248./9719169821, 13./20, 1201146811./1299019798, 1., 1.])
    a = np.array([[0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1./18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1./48, 1./16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1./32, 0, 3./32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [5./16, 0, -75./64, 75./64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [3./80, 0, 0, 3./16, 3./20, 0, 0, 0, 0, 0, 0, 0, 0],
              [29443841./614563906, 0, 0, 77736538./692538347, -28693883./1125000000, 23124283./1800000000, 0, 0, 0, 0, 0, 0, 0],
              [16016141./946692911, 0, 0, 61564180./158732637, 22789713./633445777, 545815736./2771057229, -180193667./1043307555, 0, 0, 0, 0, 0, 0],
              [39632708./573591083, 0, 0, -433636366./683701615, -421739975./2616292301, 100302831./723423059, 790204164./839813087, 800635310./3783071287, 0, 0, 0, 0, 0],
              [246121993./1340847787, 0, 0, -37695042795./15268766246, -309121744./1061227803, -12992083./490766935, 6005943493./2108947869, 393006217./1396673457, 123872331./1001029789, 0, 0, 0, 0],
             [-1028468189./846180014, 0, 0, 8478235783./508512852, 1311729495./1432422823, -10304129995./1701304382, -48777925059./3047939560, 15336726248./1032824649, -45442868181./3398467696, 3065993473./597172653, 0, 0, 0],
              [185892177./718116043, 0, 0, -3185094517./667107341, -477755414./1098053517, -703635378./230739211, 5731566787./1027545527, 5232866602./850066563, -4093664535./808688257, 3962137247./1805957418, 65686358./487910083, 0, 0],
              [403863854./491063109, 0, 0, -5068492393./434740067, -411421997./543043805, 652783627./914296604, 11173962825./925320556, -13158990841./6184727034, 3936647629./1978049680, -160528059./685178525, 248638103./1413531060, 0, 0]],
             dtype=complexX)
    b = np.array([14005451./335480064, 0, 0, 0, 0, -59238493./1068277825, 181606767./758867731,   561292985./797845732,   -1041891430./1371343529,  760417239./1151165299, 118820643./751138087, -528747749./2220607170,  1./4],
                dtype=complexX)

#     #### Coefficients for RK4 method
#     c = np.array([0,1./2, 1./2, 1.])
#     a = np.array([[0,0,0,0],
#             [1./2, 0, 0, 0],
#            [0, 1./2, 0, 0],
#            [0, 0, 1, 0]],
#              dtype=complexX)
#     b = np.array([1./6, 1./3, 1./3, 1./6],
#                 dtype=complexX)
    

    
    def __init__(self, H0, Cms, Hks, ukts, ukjs, psi0, dt, T, sparse = False, display_tqdm = True):
        self.sparse = sparse
        self.display_tqdm = display_tqdm
        
        self.setup()
        self.a=ODE87s.a
        self.b=ODE87s.b
        
        self.dt=dt
        self.T=T
        self.tpts=np.arange(0,self.T,self.dt)
        
        
        
        if Cms == [] :
            Cms = [np.zeros_like(H0)]
        
        Hr = [np.dot(nphermitian(op),op) for op in Cms]
        self.H0 = H0 - 0.5j* reduce(np.add, Hr)# H_eff for quantum trajectories
        self.Hks = Hks
        self.ukts = ukts
        self.ukjs = ukjs
        
        self.psi0 = psi0.astype(complexX).reshape((-1,))
        
        self.D0 = self.matrix(-1j*self.dt*self.H0).astype(complexX)
        
        self.Dks= [self.matrix(-1j*self.dt*op).astype(complexX) for op in self.Hks]
        #self.Dks = np.moveaxis(np.array(self.Dks[:]), [0,1,2], [1,0,2])
        
        self.num_ops = len(self.Hks)
        
        self.Cms = [self.matrix(op).astype(complexX) for op in Cms]
        
        self.ukjs_int = self.interp_controls(self.ukts,self.ukjs)
        
        #self.ukjs_int = np.moveaxis(self.ukjs_int, [0,1,2], [-1,0,1])
        
        #print self.ukjs_int.shape
        

    def interp_controls(self, xps, fps):
        return [interp1d(xp, fp, kind='cubic', bounds_error=False, fill_value = 0) for xp,fp in zip(xps,fps)]
    

    def setup(self):
        la_funcs = ["dot", "conj", "transpose", "diag", "trace"]

        for f in la_funcs: setattr(self, f, getattr(np, f))
        self.zeros = np.zeros
        self.hermitian = nphermitian
        self.transpose = np.transpose
        
        if self.sparse:
            self.zeros_matrix = sparse.csr_matrix
            self.matrix = sparse.csr_matrix
        else:
            self.zeros_matrix = np.zeros
            self.matrix = np.array
      
    def integrate(self):
        psi = self.psi0.copy()
        psis = self.zeros((len(self.tpts),len(psi)),dtype=complexX)
        
        self.rand = np.random.uniform(0, 1, size = (len(self.tpts))) # random number deciding jump happen
        self.rand_2 = np.random.uniform(0, 1, size = (len(self.tpts))) # random number deciding which jump
        
        for jj,t in tqdm_notebook(enumerate(self.tpts),disable=not self.display_tqdm):
            
            #psi=psi/np.sqrt(self.dot(self.hermitian(psi),psi))
            psi = self.evolve_w_jump(psi, t, jj)
            psis[jj] = psi.copy()
        return self.tpts, psis
    
    def evolve_w_jump(self, psi, t, jj):
        dps = [self.dt*np.sum(np.square(np.abs(op.dot(psi)))) for op in self.Cms]
        #print dps
        dps_tot = sum(dps)
        #print dps_tot
        
        #self.rand = random.uniform(0, 1)
                    
        if self.rand[jj] >= dps_tot : # no jump happen
            # evolve under H_eff
            k = self.zeros((len(ODE87s.c), len(psi)),dtype=complexX)
            #k = [self.zeros((len(psi),1),dtype=complexX) for ii in range(len(ODE87s.c))]
            
            ukjs_int_at_t = [ukj_int(t + ODE87s.c * self.dt) for ukj_int in self.ukjs_int]
            
            for ii in xrange(len(ODE87s.c)):
                psip=psi.copy()
                psip += np.dot(self.a[ii,:ii], k[:ii,:])
                #for nn in range(ii):
                #    psip+=self.a[ii,nn]*k[nn]
                #k[ii] = self.D0.dot(psip)
                
                #k[ii] += self.dot(np.dot(self.ukjs_int[:,ii,jj], self.Dks ),psip )
                #for mm in range(self.num_ops):
                #    k[ii] += self.dot(self.ukjs_int[mm,ii,jj]*self.Dks[mm],psip )
                
                #Dk_tot = self.zeros_matrix((len(psi),len(psi)),dtype=complexX)
                Dk_tot = self.D0.copy()
                for mm in range(self.num_ops):
                    Dk_tot += ukjs_int_at_t[mm][ii]*self.Dks[mm]
                
                k[ii] = Dk_tot.dot(psip)
                
            psi += np.dot(self.b, k)
            #for ii in xrange(len(ODE87s.c)):
            #    psi += self.b[ii]*k[ii]
                
            # normalize
            psi = psi/np.sqrt(1-dps_tot)
            
        else: # a jump shall happen
            rand_picker = dps_tot * self.rand_2[jj]
            dp_acc = 0
            for dp, op in zip(dps, self.Cms):
                dp_acc += dp
                if dp_acc >= rand_picker: # sample a particular Cm to jump
                    psi = op.dot(psi)/np.sqrt(dp/self.dt) # jump happens

                    break
        
        return psi
    
    
import numpy as np
import h5py
import qutip as qt

def qutip_verification(H0, Cms, Hks, ukts, ukjs, psi0, dt, T , measurements_avg, atol = 1e-2):
    
    
    # load data
    gate_time = T
    dt = dt
    H0 = H0
    Hops = Hks
    Cms = Cms
    initial_vectors_c = psi0
    ukts = ukts
    uks = ukjs
    
    ukdts = [ukt[1]-ukt[0] for ukt in ukts]
    
    tpts=np.arange(0,T,dt)

    
    
    average_abs_diff_list = []
    all_close_list = []
    
    # H0 and Hops
    H0_qobj = qt.Qobj(H0)
    Hops_qobj = []

    for Hop in Hops:
        Hops_qobj.append(qt.Qobj(Hop))
        
    # lindblad dissipator
    Cms_ops = [qt.lindblad_dissipator(qt.Qobj(Cm)) for Cm in Cms]
            
    # define time    
    tlist = np.arange(0,T,dt)
        
    
        
        
    # initial vector
    psi0 = qt.Qobj(initial_vectors_c)


    # make functions to return uks field
    def make_get_uks_func(uid):
        def _function(t,args=None):
            time_id = int(t/ukdts[uid])
            if time_id >= len(ukts[uid]): return 0
            return uks[uid][time_id]
        return _function

    # create the time-dependent Hamiltonian list
    Ht_list = []
    Ht_list.append(H0_qobj)
    for ii in range(len(Hops)):
        Ht_list.append([Hops_qobj[ii],make_get_uks_func(ii)])

    # solving the Schrodinger evolution in QuTiP's sesolve
    output = qt.mesolve(Ht_list, psi0*psi0.dag(), tlist, Cms_ops, [], progress_bar = True)

    # obtaining the simulation result
    state_tlist = []
    for state in output.states:
        state_tlist.append(state.full())
    state_tlist = np.array(state_tlist)[:,0,0]
    
    plt.figure(figsize=(10,10))
    plt.subplot(211, xlabel='time', ylabel='control amp')
    for ukt,ukj in zip(ukts,uks):
        plt.plot(ukt,ukj)
    
    plt.subplot(212, xlabel='time', ylabel='probability')
    plt.plot(tpts,measurements_avg, label='trajectory')
    plt.plot(tpts,state_tlist, label='master eq.')
    plt.legend()
    
#     plt.ylim(-0.01,1.01)


    # average absolute difference of simulation result from Tensorflow and QuTiP
    abs_diff = np.abs(state_tlist) - np.abs(measurements_avg)        
    average_abs_diff_list.append(np.average(np.abs(abs_diff)))

    # if all close between simulation result from Tensorflow and QuTiP
    all_close = np.allclose(state_tlist,measurements_avg,atol=atol)        
    all_close_list.append(all_close)
    
    print "QuTiP simulation verification result for each initial state"
    print "================================================"
    print "average abs diff: " + str(average_abs_diff_list)
    print "all close: " + str(all_close_list)
    print "================================================"
