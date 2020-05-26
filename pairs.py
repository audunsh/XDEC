import sys
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
import copy as cp
import os
import numpy.lib.recfunctions as nlr

sys.path.append(os.path.realpath('..'))



def setup_pairs(    p,
                    fragms,
                    coords,
                    centers,
                    spreads,
                    fot,
                    sort_crit='standard',
                    layer_groups=None,
                    min_incr=7.0,
                    cutoff_factor=1.0,
                    cutoff_algorithm=2):

    min_dist = 0.0
    max_dist = 30.0
    cutoff_algorithm = 4

    print ()
    print ('Setting up pairs ...')
    print ('Parameters for pair setup: ')
    print ('Min distance: ',min_dist)
    print ('Max distance: ',max_dist)
    print ('Cutoff algorithm: ',cutoff_algorithm)
    print ()

    if (cutoff_algorithm == 4) and (len(fragms) == 1):
        cutoff_algorithm = 0
        print ('Only one fragment in the refenerence cell, the cutoff algorithm is therefore changed from 4 to 0')

    SP = SortPairs()
    SP.set_crit(    fragms,
                    spreads,
                    crit_list=sort_crit,
                    layer_groups=layer_groups   )

    P = Pairs()

    PD = PairDealer(P,SP)
    PD.set_algorithm(cutoff_algorithm)
    PD.set_conv(conv_eps=fot,fot_factor=cutoff_factor)
    PD.create_complete_pairlist(    p,
                                    coords,
                                    fragms,
                                    spreads,
                                    centers,
                                    layer_groups=layer_groups,
                                    min_dist = min_dist,
                                    max_dist = max_dist,
                                    min_incr = min_incr     )

    return PD







class Pairs:
    def __init__(self):
        self.a              = None  #structured array:ind_a,ind_b,cell_b,group_id,dist,E,estimE,calced
        self.n_pairs        = None  #total number of pairs
        self.n_groups       = None  #total number of groups
        self.n_notcalced    = None  #number of notcalced pairs
        self.n_cells        = None  #number of unit cells
        ###IMPORTANT: self.a is assumed to be sorted by dist at all time###

    def init_arrays(self,pair_indx,pair_coords,group_id,dist,n_cpairs,n_cpairs_ref,min_dist,max_dist):
        self.n_pairs = len(dist)

        a = np.zeros([self.n_pairs,5],dtype=float)
        a[:,0:2] = pair_indx
        a[:,2] = pair_coords
        a[:,3] = group_id
        a[:,4] = dist
        dt = np.dtype([ ('ind_a',int),('ind_b',int),('cell_b',int),
                        ('group_id',int),('dist',float)   ])

        ind_min = a[:,4] >= min_dist
        ind_max = a[:,4] <= max_dist
        ind = ind_min*ind_max
        a = a[ind]

        self.n_pairs = len(a)

        #self.a = nlr.unstructured_to_structured(a,dtype=dt)
        self.a = self.unstruct_to_struct(a,dt)
        self.n_notcalced = len(a)
        self.n_groups = int(np.max(self.a['group_id'])) + 1
        self.n_cells = int(np.max(self.a['cell_b'])) + 1
        self.n_cpairs = n_cpairs
        self.n_cpairs_ref = n_cpairs_ref

    def add_param(self,names,dtype,values):
        self.a = nlr.append_fields(self.a,names,np.zeros(self.n_pairs),dtype,usemask=False)
        self.a[names] = values

    def set_estimE(self,estimE,group_id):
        ind = np.equal(self.a['group_id'],group_id)
        ind *= np.invert(self.a['calced'])
        self.a['estimE'][ind] = estimE

    def get_calced(self,names,group_id):
        ind = np.equal(self.a['group_id'],group_id)
        ind *= self.a['calced']
        return self.struct_to_unstruct(self.a[names][ind],names)

    def get_notcalced(self,names,group_id):
        ind = np.equal(self.a['group_id'],group_id)
        ind *= np.invert(self.a['calced'])
        return self.struct_to_unstruct(self.a[names][ind],names)

    def get_notcalced_by_estimE(self,names=['ind_a','ind_b','cell_b','dist']):
        ind = np.invert(self.a['calced'])
        where = np.argmax(self.a['estimE'][ind])
        return self.a[names][ind][where]############################################ np.array()

    def get_notcalced_by_dist(self,names=['ind_a','ind_b','cell_b','dist']):
        ind = np.invert(self.a['calced'])
        where = np.argmin(self.a['dist'][ind])

        return self.a[names][ind][where]

    def get_notcalced_by_dist4(self,names=['ind_a','ind_b','cell_b','dist']):
        ind = np.invert(self.a['calced'])
        ind *= self.a['ind_a'] != self.a['ind_b']
        where = np.argmin(self.a['dist'][ind])
        return self.a[names][ind][where]

    def compute_cell(self,name):
        """
        returns the values summed over each unit cell, along with a boolean
        array of which cells are calculated
        """
        ret_vals = np.zeros(self.n_cells,dtype=float)
        ret_bool = np.zeros(self.n_cells,dtype=bool)

        ret_vals[0] = np.sum(self.a[name][:self.n_cpairs_ref])
        ret_vals[1:] = np.sum(self.a[name][self.n_cpairs_ref:].reshape(self.n_cells-1,self.n_cpairs),axis=1)

        ret_bool[0] = self.a['calced'][0]
        ret_bool[1:] = np.sum(self.a['calced'][self.n_cpairs_ref:].reshape(self.n_cells-1,self.n_cpairs),axis=1)

        return ret_vals,ret_bool


    def struct_to_unstruct(self,a_struct,names):
        n_el = len(a_struct)
        n_fields = len(a_struct[0])
        array = np.zeros([n_el,n_fields])
        for i in range(n_fields):
            array[:,i] = a_struct[names[i]]
        return array

    def unstruct_to_struct(self,array,dt):
        n_el = len(array)
        n_fields = len(dt.names)
        a_struct = np.empty(n_el,dtype=dt)
        for i in range(n_fields):
            a_struct[dt.names[i]] = array[:,i]
        return a_struct

    def sort_by(self,names):
        self.a = np.sort(self.a,order=names)



    def add_cellE(cellE,cell_b):
        ind = np.equal(self.a['cell_b'],cell_b)
        self.a['E'][ind] = cellE

    def add_E(self,pair):
        ind = np.where((self.a['ind_a']==pair[0]) & (self.a['ind_b']==pair[1]) & (self.a['cell_b']==pair[2]))[0][0]
        self.a['E'][ind] = pair[3]
        self.a['calced'][ind] = True
        self.n_notcalced -= 1
        return self.a['group_id'][ind]

    def add_E4(self,pair):
        ind = np.where((self.a['ind_a']==pair[0]) & (self.a['ind_b']==pair[1]) & (self.a['cell_b']==pair[2]))[0]
        if len(ind) == 0:
            return self.a['group_id'][ind]
        else:
            if self.a['calced'][ind]:
                if pair[3] < self.a['E'][ind]:
                    self.a['E'][ind] = pair[3]
            else:
                self.a['E'][ind] = pair[3]
                self.a['calced'][ind] = True
                self.n_notcalced -= 1
        return self.a['group_id'][ind]

    def save_de(self,outfile='de.txt'):
        ind = self.a['calced']
        de = self.struct_to_unstruct(self.a[ind][['dist','E']],['dist','E'])
        np.savetxt(outfile,de)

    def print_de(self):
        ind = self.a['calced']
        de = self.struct_to_unstruct(self.a[ind][['dist','E','ind_a','ind_b','cell_b']],['dist','E','ind_a','ind_b','cell_b'])
        print ('    Distance          Energy                     index_A  index_B   cell_B')
        for i in np.arange(len(de)):
            print ('{0:16.10f}     {1:20.16f} {2:8d} {3:8d}  {4:8d}'.format(de[i,0],de[i,1],int(de[i,2]),int(de[i,3]),int(de[i,4])))



    """

    def add_E(self,pair):
        ind = np.where((self.a['ind_a']==pair[0]) & (self.a['ind_b']==pair[1]) & (self.a['cell_b']==pair[2]))[0][0]
        self.a['E'][ind] = pair[4]
        self.a['calced'][ind] = True
        self.n_notcalced -= 1
        return self.a['group_id'][ind]

    def get_calced_dist_E(self,group_id):
        ind = np.where( self.a['group_id'] == group_id )
        group_a = self.a[ind][['dist','E']]
        group_a = group_a[ self.a[ind]['calced'] ]
        return self.struct_to_unstruct(group_a,['dist','E'])
        #return nlr.structured_to_unstructured(group_a,['dist','E'])

    def get_notcalced_dist_E(self,group_id):
        ind = np.where( self.a['group_id'] == group_id )
        group_a = self.a[ind][['dist','E']]
        group_a = group_a[ np.invert(self.a[ind]['calced']) ]
        return self.struct_to_unstruct(group_a,['dist','E'])
    """


    def get_pairs_group(self,group_id):
        """returns structured array (ind_a,ind_b,cell_b,dist,E)"""
        ind = np.where( self.a['group_id'] == group_id )
        return self.a[ind][['ind_a','ind_b','cell_b','dist','E']]


    def get_pair_by_ind(self,group_id,ind,names=['ind_a','ind_b','cell_b','dist']):
        inds = np.where( self.a['group_id'] == group_id )
        return list(self.a[names][inds][ind])


    def get_distant_pair(self,group_id,names=['ind_a','ind_b','cell_b','dist']):
        inds = np.where(self.a['group_id'] == group_id)
        return list(self.a[names][inds][-1])


    def get_closest_pair(self,group_id,names=['ind_a','ind_b','cell_b','dist']):
        inds = np.where(self.a['group_id'] == group_id)
        return list(self.a[names][inds][0])









class SortPairs:
    def __init__(self):
        self.crit  = None
        self.n_groups   = None
        self.n_groups_tot = None
        self.pair_ind_array = None
        self.group_IDs  = None
        self.layer_groups = None
        self.layer_newgroups = None

    def set_crit(self,fragms,spreads,crit_list='all_unique',spread_eps = 0.1,layer_groups=None):
        # * system is a DECSystem object
        # * 'all_equiv': no sorting
        # * 'all_unique': each atom in unit cell is treated as unique
        # * 'atom_type':  sort by atomic element type
        # * 'standard': atom type, but fragments of same atom type is treated
        #   as non-equivalent if they do not have a matching set of MO spreads
        # * crit_list format example: [[0,1],[3,4,5],[2]] : 0 and 1 are equivalent, along with 3, 4 and 5. 2 is unique
        n_fragms = len(fragms)

        if crit_list == 'all_unique':
            n_nonequiv = n_fragms
            self.crit = [[i] for i in range(n_nonequiv)]

        if crit_list == 'all_equiv' or 'standard':
            n_nonequiv = 1
            self.crit = [[i for i in range(n_fragms)]]

        if crit_list == 'standard':
            self.crit = self.standard_sort(self.crit,spreads,fragms,spread_eps)
            n_nonequiv = len(self.crit)


        if type(crit_list) == list:
            self.crit = crit_list
            n_nonequiv = len(crit_list)


        self.n_groups = int(0.5*n_nonequiv*(n_nonequiv+1))

        #setting up pair_ind_array
        self.group_IDs = -1*np.ones([n_fragms,n_fragms],dtype=int)
        #setting up group_IDs
        #add pairs of equivalent atoms
        #self.group_IDs = -1*np.ones(len(xy[:,0]))
        group_id = 0
        for i in range(len(self.crit)):
            for j in range(n_fragms):
                for k in range(n_fragms):
                    if (j in self.crit[i]) and (k in self.crit[i]):
                        self.group_IDs[j,k] = group_id
            group_id += 1

        #add pairs of non-equivalent atoms
        for i in range(len(self.crit)-1):
            for j in range(i+1,len(self.crit)):
                for k in range(n_fragms):
                    for l in range(n_fragms):
                        if ((k in self.crit[i]) and (l in self.crit[j])) or ((k in self.crit[j]) and (l in self.crit[i])):
                            self.group_IDs[k,l] = group_id
                group_id +=1



        self.n_groups_tot = self.n_groups
        if layer_groups is not None:
            self.layer_groups = layer_groups
            self.layer_newgroups = [i for i in range(int(np.max(self.group_IDs))+1,int(np.max(self.group_IDs))+int(len(layer_groups))+1)]
            self.layer_groups = np.array(self.layer_groups)
            self.layer_newgroups = np.array(self.layer_newgroups)
            self.n_groups_tot = int(np.max(self.layer_newgroups)) + 1



    def standard_sort(self,crit,spreads,fragms,spread_eps):
        """
        fragments with similar set of orbital spreads are grouped together
        """
        index = -1
        temp_crit = []
        for i in range(len(crit)):
            distrib_list = cp.deepcopy(crit[i])
            n_distrib = len(distrib_list)
            while n_distrib > 0:
                if len(distrib_list) == 1:
                    temp_crit.append([cp.deepcopy(distrib_list[0])]);index += 1
                    break
                temp_crit.append([cp.deepcopy(distrib_list[0])]);index += 1
                del distrib_list[0]
                for_deletion = []
                for j in range(len(distrib_list)):
                    ind_a = temp_crit[index][0]
                    ind_b = distrib_list[j]
                    occ_mos_a = np.array(fragms[ind_a])
                    occ_mos_b = np.array(fragms[ind_b])
                    mo_list_a = spreads[occ_mos_a]
                    mo_list_b = spreads[occ_mos_b]
                    if len(mo_list_a) != len(mo_list_b):
                        equal_mos = False
                    else:
                        equal_mos = np.allclose(mo_list_a,mo_list_b,rtol=0,atol=spread_eps)
                    if equal_mos == True:
                        temp_crit[index].append(distrib_list[j])
                        for_deletion.append(distrib_list[j])
                for j in for_deletion:
                    distrib_list.remove(j)
                n_distrib = len(distrib_list)
        crit = temp_crit
        return crit



    def layer_regroup(self,g_id,latvec,direction=1,eps=1e-10):
        j = 0
        for i in self.layer_groups:
            ind = np.where((g_id == i)&(np.abs(latvec[:,direction]) < eps))
            g_id[ind] = self.layer_newgroups[j]
            j += 1
        return g_id



    def get_group_IDs(self,pindx):
        return self.group_IDs[pindx[:,0],pindx[:,1]]










class PairDealer:
    def __init__(self,pairs,sort_pairs):
        self.P                      = pairs
        self.SP                     = sort_pairs
        self.conv_eps               = None
        self.conv_list              = None
        self.s_list_bool            = None
        self.remainE_list           = None
        self.remainE                = None
        self.early_pairs            = None
        self.get_pair               = None      #method
        self.add                    = None      #method
        self.estim_remainE          = None      #method
        self.pair_alg               = None
        self.conv                   = False
        self.contr_conv             = False
        self.temp_group             = None      #group_id of prev calced pair



    def create_complete_pairlist(self,p,coords,fragms,spreads,centers,layer_groups=None,min_dist=0,max_dist=np.inf,min_incr=7.0):
        """
        takes a prism object p, cell coordinates. Creates a Pairs object
        with pair distances
        """
        n_fragms = len(fragms)

        n_cpairs = n_fragms**2
        n_cpairs_ref = int(0.5*n_fragms*(n_fragms-1))
        #setting up coords for all fragment pairs, corresponding pair
        #indices and distances.
        coords_ext = self.expand_coords(coords,n_cpairs,n_cpairs_ref)
        cell_indx = self.create_cell_indx(coords,n_cpairs,n_cpairs_ref)
        pair_indx = self.create_pair_indx(coords_ext,n_fragms,n_cpairs,n_cpairs_ref)
        fragm_pos = self.create_fragm_pos(spreads,centers,fragms)
        dists = self.create_dists(p,coords_ext,pair_indx,fragm_pos)

        group_id = self.SP.get_group_IDs(pair_indx)

        if layer_groups is not None:
            group_id = self.SP.layer_regroup(group_id,coords_ext)


        self.P.init_arrays(pair_indx,cell_indx,group_id,dists,n_cpairs,n_cpairs_ref,min_dist,max_dist)
        self.P.add_param('E',float,0)
        self.P.add_param('estimE',float,np.inf)
        self.P.add_param('calced',bool,False)
        self.P.sort_by('dist')

        if self.pair_alg != 0:
            #self.early_cells = self.get_early_cells(coords_ext,cell_indx,dists)
            self.early_pairs = self.get_early_pairs(incr=min_incr)


    def expand_coords(self,coords,n_cpairs,n_cpairs_ref):
        """
        takes in cell coords array and expands it to cell coords for all fragment pairs
        """
        old_len = len(coords)
        new_len = n_cpairs_ref + n_cpairs*(old_len-1)
        coords_ret = np.zeros([new_len,3],dtype=int)
        coords_ret[n_cpairs_ref:] = np.repeat(coords[1:],n_cpairs,axis=0)
        return coords_ret

    def create_cell_indx(self,coords,n_cpairs,n_cpairs_ref):
        old_len = len(coords)
        new_len = n_cpairs_ref + n_cpairs*(old_len-1)
        indx_ret = np.zeros(new_len,dtype=int)
        indx_ret[n_cpairs_ref:] = np.repeat(np.arange(1,old_len),n_cpairs)
        return indx_ret


    def create_pair_indx(self,coords,n_fragms,n_cpairs,n_cpairs_ref):
        """
        returns array of cell pairs corresponding to coords_exp
        """
        X,Y = np.mgrid[0:n_fragms:1, 0:n_fragms:1]
        X = X.ravel()
        Y = Y.ravel()
        indx_cell = np.zeros([len(X),2],dtype=int)
        indx_cell[:,0] = X
        indx_cell[:,1] = Y
        indx_cell_ref = indx_cell[np.invert(np.equal(X,Y))]
        if n_fragms > 1:
            indx_cell_ref = np.sort(indx_cell_ref,axis=1)
            indx_cell_ref = np.unique(indx_cell_ref,axis=0)

        ind_len = len(coords)
        indx = np.zeros([ind_len,2],dtype=int)
        indx[:n_cpairs_ref] = indx_cell_ref
        indx[n_cpairs_ref:] = np.tile(indx_cell.T,int(len(indx[n_cpairs_ref:])/n_cpairs)).T

        return indx


    def create_dists(self,p,coords,pair_indx,fragm_pos):
        """
        return array with all pair distances
        """
        lattice = np.copy(p.lattice)
        if p.cperiodicity == 'POLYMER':
            lattice[2,:] = lattice[1,:] = 0
        elif p.cperiodicity=="SLAB":
            lattice[2,:] = 0

        vecs = lattice[0,:]*coords[:,0][:,np.newaxis] \
             + lattice[1,:]*coords[:,1][:,np.newaxis] \
             + lattice[2,:]*coords[:,2][:,np.newaxis]

        vecs -= fragm_pos[pair_indx[:,0],:]
        vecs += fragm_pos[pair_indx[:,1],:]

        dists = np.linalg.norm(vecs,axis=1)
        return dists


    def create_fragm_pos(self,spreads,centers,fragms,spread_tol=0.1):
        """
        Takes mo_spreads, mo_centers and center_fragments list.
        Generates fragment positions in the reference cell.
        If there is one spreadwise dominating occupied MO, the fragment
        position is the same as that MO center. If there are MOs with spread
        less than spread_tol smaller than the max spread, the centroid of
        those orbital centers is used as fragment position.
        """
        n_fragms = len(fragms)
        fragm_pos = np.zeros([n_fragms,3],dtype=float)
        i = 0
        for fragm in fragms:
            spreads_i = spreads[fragm]
            centers_i = centers[fragm]
            #ind = np.argmax(spreads_i)
            ind_max = np.argmax(spreads_i)
            where = np.argwhere( (spreads_i[ind_max]-spread_tol < spreads_i) & (spreads_i[ind_max]+spread_tol > spreads_i) )
            positions = np.atleast_2d( centers_i[where] )
            n = len(positions)
            pos = np.sum(positions,axis=0)/n
            fragm_pos[i,:] = pos
            #print ('Position fragment : ',i)
            i += 1
        return fragm_pos


    def set_algorithm(self,alg_num=0):
        m = "pair_alg must be a non-negative integer smaller than 3"
        assert ( type(alg_num) is int ) and ( alg_num < 5 ), m

        self.pair_alg = alg_num

        if alg_num == 0:
            """
            no autocutoff
            """
            self.get_pair = self.get_pair0
            self.add = self.add0
            self.estim_remainE = self.estim_remainE0
        elif alg_num == 1:
            """
            autocutoff, pair is chosen by distance, groups are converged separately
            """
            self.get_pair = self.get_pair1
            self.add = self.add0
            self.estim_remainE = self.estim_remainE1
        elif alg_num == 2:
            """
            autocutoff, pair is chosen by spline estimated energy, converges
            the totality of all group energies
            """
            self.get_pair = self.get_pair2
            self.add = self.add0
            self.estim_remainE = self.estim_remainE2
        elif alg_num == 3:
            """
            same as algorithm 2, but at a unit cell level
            """
        elif alg_num == 4:
            """
            same as algorithm 4, but only different orbital indices
            """
            self.get_pair = self.get_pair4
            self.add = self.add4
            self.estim_remainE = self.estim_remainE0

    def get_early_cells(self,coords_ext,cell_indx,dist,med_dist=20):
        early_cells = []
        early_cells.append(0)
        early_cells.append(1)
        early_cells.append(cell_indx[np.where((dist>med_dist)&(cell_indx!=1))[0][0]])
        early_cells.append(cell_indx[-1])

        if self.SP.layer_groups is not None:
            if coords[1,1] == 0:
                ind = np.where(coords_ext[:,1]!=0)[0][0]
            else:
                ind = np.where(coords_ext[:,1]==0)[0][0]
            early_cells.append(cell_indx[ind])

            if coords[early_cells[2],1] == 0:
                ind = np.where((dist>med_dist)&(cell_indx!=1)&(coords_ext[:,1]!=0))[0][0]
            else:
                ind = np.where((dist>med_dist)&(cell_indx!=1)&(coords_ext[:,1]==0))[0][0]
            early_cells.append(cell_indx[ind])

            if coords[cell_indx[-1],1] == 0:
                ind = np.where(coords_ext[:,1]!=0)[0][-1]
            else:
                ind = np.where(coords_ext[:,1]==0)[0][-1]
            early_cells.append(cell_indx[ind])

        return early_cells



    def get_closest_pairs(self):
        P = self.P
        closest_pairs = []
        for i in range(P.n_groups):
            closest_pairs.append(P.get_closest_pair(i))
        return closest_pairs


    def get_distant_pairs(self):
        P = self.P
        distant_pairs = []
        for i in range(P.n_groups):
            distant_pairs.append(P.get_distant_pair(i))
        return distant_pairs


    def get_early_pairs(self,incr=7.0):
        """
        Setting up the initial pairs
        """
        #print ('Least distance between early pairs: ',incr)
        P = self.P
        distant_pairs = self.get_distant_pairs()
        closest_pairs = self.get_closest_pairs()
        early_pairs = list(closest_pairs)
        for i in range(P.n_groups):
            c_added = 0
            prev_dist = closest_pairs[i][3]
            pair_i = P.get_pairs_group(i)
            for j in range(len(pair_i)):
                #print ()
                #print ('dist[j] --- prev_dist')
                #print ()
                #print (pair_i[j],flush=True)
                #print ()
                #print (prev_dist,flush=True)
                #while 0 < 1:
                #    pass
                if pair_i['dist'][j] > prev_dist + incr:
                    early_pairs.append(P.get_pair_by_ind(i,j))
                    prev_dist = pair_i['dist'][j]
                    c_added += 1
                if c_added == 2:
                    break
        early_pairs = early_pairs + distant_pairs
        #print ('IN earlypairs')
        #print (early_pairs)
        return early_pairs

    """
    def get_pair(self):
        pairs = self.pair_array
        E_max_pairs = []
        for i in self.active_groups:
            pairs_i = self.get_notcalced(i)
            ind = np.argmax( pairs_i[:,5] )
            E_max_pairs.append(pairs[i][ind,:])
        E_max_ind = np.argmax(np.array(E_max_pairs)[:,5])
        E_max = np.array(E_max_pairs)[E_max_ind,:]
        return int(E_max[0]), int(E_max[1]), E_max[2], int(E_max[4])
        #return np.array(E_max_pairs)[E_max_ind,:]
    """

    def get_pair0(self):
        p = self.P.get_notcalced_by_dist()
        return p[0],p[1],p[2],p[3]

    def get_pair1(self):
        if self.early_pairs is not None:
            ep = self.early_pairs[0]
            del self.early_pairs[0]
            if len(self.early_pairs) == 0:
                self.early_pairs = None
            return ep[0],ep[1],ep[2],ep[3],ep[4]
        else:
            group_id = self.conv_list.index(False)
            p = self.P.get_notcalced_by_groupdist(group_id)
            return p[0],p[1],p[2],p[3],p[4]

    def get_pair2(self):
        if self.early_pairs is not None:
            ep = self.early_pairs[0]
            del self.early_pairs[0]
            if len(self.early_pairs) == 0:
                self.early_pairs = None
            #print ('-----------')
            #print ('EARLY PAIRS')
            #print (ep[0],ep[1],ep[2],ep[3])
            #print ('-----------')
            return ep[0],ep[1],ep[2],ep[3]
        else:
            p = self.P.get_notcalced_by_estimE()
            #print ('-----------')
            #print ('GENERAL PAIRS')
            #print (p)
            #print (type(p))
            #print (len(p))
            #print (p[0],p[1],p[2])
            #print ('-----------')
            return p[0],p[1],p[2],p[3]

    def get_pair3(self):
        if self.early_cells is not None:
            ec = self.early_cells[0]
            del self.early_cells[0]
            if len(self.early_cells) == 0:
                self.early_cells = None
            return ec
        else:
            cE,cC = self.P.compute_cell('estimE')
            notcalced_ind = np.where(cC == False)[0]
            ind = np.argmax(cE[np.invert(cC)])
            return notcalced_ind[ind]

    def get_pair4(self):
        p = self.P.get_notcalced_by_dist4()
        return p[0],p[1],p[2],p[3]



    def add0(self,pair,pair_list):
        self.temp_group = self.P.add_E(pair)

    def add3(self,cellE,pair_list):
        self.SP.add_cellE(cellE,self.temp_cell)

    def add4(self,pair,pair_list):
        j = [[0,0],[1,1],[0,1],[1,0]]
        if pair[2] == 0:
            self.temp_group = self.P.add_E4(pair)
        else:
            for i in np.arange(0,4):
                self.temp_group = self.P.add_E4([pair[j[i][0]],pair[j[i][1]],pair[2],pair_list[i]])

    def set_conv(self,conv_eps=0.00001,fot_factor=1.0):
        conv_eps *= fot_factor
        self.conv_tot = conv_eps
        self.conv_eps = [conv_eps/self.SP.n_groups_tot]*self.SP.n_groups_tot
        self.conv_lim = conv_eps
        #self.remain_energy_list = [np.infty]*self.P.n_groups
        self.remainE_list = np.inf*np.ones(self.SP.n_groups_tot,dtype=float)
        self.conv_list = [False]*self.SP.n_groups_tot


    def check_conv(self,ind_ab):
        id = self.get_group_id(ind_ab)
        return self.conv_list[id]

    def estim_remainE0(self,extrap_pairs=False):
        if self.P.n_notcalced == 0:
            self.conv = True

    def estim_remainE1(self,extrap_pairs=False):
        g = self.temp_group
        calced_pairs = self.P.get_calced_dist_E(g)
        notcalced_pairs = self.P.get_notcalced_dist_E(g)
        spline_pairs = self.create_spline_datapoints(calced_pairs)
        if spline_pairs is False:
            return 0
        #print ('STALLED',flush=True)
        #while 0 < 1:
        #    pass
        spline_obj = self.run_spline(spline_pairs,extrap_pairs = extrap_pairs)
        notcalced_E = self.get_yspline(notcalced_pairs[:,0],spline_obj)
        self.remainE_list[g] = np.sum(notcalced_E)
        self.P.set_estimE(notcalced_E,g)

        if self.remainE_list[g] < self.conv_eps[g]:
            self.conv_list[g] = True
        if self.conv_list == [True]*self.P.n_groups:
            self.conv = True

    def estim_remainE2(self,extrap_pairs=True):
        g = self.temp_group
        calced_pairs = self.P.get_calced(['dist','E'],g)
        notcalced_pairs = self.P.get_notcalced(['dist','E'],g)
        spline_pairs = self.create_spline_datapoints(calced_pairs)
        #print ('///   remainE_list')
        #print (self.remainE_list)
        #print ()
        if spline_pairs is False:
            return 0
        spline_obj = self.run_spline(spline_pairs,extrap_pairs = extrap_pairs)
        notcalced_E = self.get_yspline(notcalced_pairs[:,0],spline_obj)
        self.remainE_list[g] = np.sum(notcalced_E)
        self.remainE = np.sum(self.remainE_list)
        #print ('///   remainE')
        #print (self.remainE)
        self.P.set_estimE(notcalced_E,g)

        if self.remainE < self.conv_tot:
            self.conv = True


    def estim_remainE3(self,extrap_pairs=False):
        for g in np.arange(self.SP.n_groups_tot):
            calced_pairs = self.P.get_calced_dist_E(g)
            notcalced_pairs = self.P.get_notcalced_dist_E(g)
            spline_pairs = self.create_spline_datapoints(calced_pairs)

            if spline_pairs is False:
                return 0

            spline_obj = self.run_spline(spline_pairs,extrap_pairs = extrap_pairs)
            notcalced_E = self.get_yspline(notcalced_pairs[:,0],spline_obj)
            self.remainE_list[g] = np.sum(notcalced_E)

        remainE = np.sum(self.remainE_list)

        self.P.set_estimE(notcalced_E,g)

        if remainE < self.conv_tot:
            self.conv = True

        #print ('///   remainE_list')
        #print (self.remainE_list)
        #print ()
        #print ('///   remainE')
        #print (remainE)





    def num_control(self):
        return 0

    def error_control(self):
        return 0



    """
    def estim_remain_energy(self,extrap_pairs=False):
        from time import time
        #print ('#########remain_energy_list########')
        #print (self.remain_energy_list)
        #print ()
        for i in self.active_groups:
            calced_pair_array = self.get_calced(i)
            pair_spline_array = self.create_spline_datapoints( calced_pair_array )
            #print ('######pair_spline_array#######')
            #print (pair_spline_array,flush=True)
            #print ()
            if pair_spline_array is False:
                continue
            #print ('condition was TRUE')
            #print ()
            spline_obj = self.run_spline(pair_spline_array,extrap_pairs = extrap_pairs)
            not_calced_pair_array = self.get_notcalced(i)
            not_calced_energies = self.get_yspline(not_calced_pair_array[:,2],spline_obj)
            self.pair_array[i][:,5][np.invert(self.calced_pairs[i])] = not_calced_energies
            self.remain_energy_list[i] = np.sum(not_calced_energies)
            if self.remain_energy_list[i] < self.conv_eps[i]:
                self.conv_list[i] = False
        conv_tot = np.sum(self.remain_energy_list)
        if conv_tot < self.conv_lim:
            self.conv_tot = True
    """



    def create_spline_datapoints (self,datapoints,decimal_tol=5):
        if datapoints is None:
            return False

        temp_pair_array = np.copy(datapoints)
        spline_datapoints = []


        spline_points_k = []
        temp_pair_array[:,0] = np.round(temp_pair_array[:,0],decimals=decimal_tol)
        distance, num_of_duplicates = np.unique(temp_pair_array[:,0], return_counts=True)
        i = 0
        for el in num_of_duplicates:
            energy_mean = 0
            for j in range(el):
                energy_mean += temp_pair_array[i+j,1]
            energy_mean = energy_mean/float(el)
            spline_points_k.append([temp_pair_array[i,0],energy_mean])
            i += el

        spline_datapoints = np.array(spline_points_k)

        #If there are not enough spline datapoints:return False
        enough_points = False

        try:
            test_len = len(spline_datapoints[:,0])
        except:
            return False

        if test_len >= 4:
            enough_points = True
            self.s_list_bool = True

        if not enough_points:
            return False

        return spline_datapoints


    def run_spline (self,spline_datapoints,first_el=0,extrap_pairs=False,eff_opt=False):
        de_spline = np.copy(spline_datapoints)

        de_spline[:,1] = -de_spline[:,1]
        de_spline = np.log10(de_spline)

        if extrap_pairs == True:
            extrap_pair_i = self.get_extrap_pair(de_spline)
            de_spline = np.concatenate((de_spline,extrap_pair_i),axis=0)


        ##### performing spline #####
        s = UnivariateSpline(de_spline[first_el:,0], de_spline[first_el:,1], s=1000)

        return s


    def get_extrap_pair(self,de_array,x_value=200):
        max_x = 10**(np.amax(de_array[:,0]))
        x_val = np.log10( max_x + x_value )
        slope, intercept, r_value, p_value, std_err = linregress(de_array[:,0],de_array[:,1])
        y_val = intercept + slope*x_val
        artif_pair = np.array([[x_val,y_val]])
        return artif_pair


    def get_yspline(self,xspline,spline_obj):
        xspline_l = np.log10(xspline)
        yspline_l = cp.deepcopy(spline_obj(xspline_l))
        yspline = 10**yspline_l
        return yspline


    def run_spline_simple(self,datapoints):
        spline_datapoints = self.create_spline_datapoints(spline_datapoints)
        s = self.run_spline(spline_datapoints)
        return s
