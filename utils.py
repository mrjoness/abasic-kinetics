# objects and functions to assist abasic analyis
import numpy as np
import glob
import pyemma as py
from hde import HDE, analysis
import mdtraj as md

# dictionary linking sequence and site with a near Tm simulation temp
abasic_configs = {
    'TTTTTTTTTTT':{'control':336, 'b2':335, 'b4':326, 'b6':320}, 
    'TATAGCGATAT':{'control':336, 'b2':328, 'b4':321, 'b6':303},
    'CCTATATATCC':{'control':328, 'b2':315, 'b4':309, 'b6':307},
    'CGCATATATAT':{'control':330, 'b2':312, 'b4':309, 'b6':312},
}

def translate_pairs(num_bp, base, strand=1, exclude_list=[1, 3, 5]):
    ''' exclude unique distances between base types, 
        translate share coordinates,
        return list of indexs needed to establish a shared basis
       
       Inputs:
       num_bp -- number of base pairs in sequence
       base -- abasic site'''
     
    # account for control
    s1_control = [i*3 + 1 for i in range(num_bp)]
    s2_control = [i*3 + num_bp*3 for i in range(num_bp)]
    
    # make list of excluded bases
    b1_excluded = np.array(s1_control)[exclude_list]
    b1_excluded = list(b1_excluded) + list(b1_excluded - 1)
    
    if base == 'control':
        s1 = s1_control
        s2 = s2_control
    else:
        # shift index down for missing bp
        b_idx = int(base[-1])
        s1 = [i*3 + 1 for i in range(b_idx-1)] + [i*3 for i in range(b_idx, num_bp)]
        s2 = s2_control
        
    idx = 0
    idx_new = 0
    idx_list = []
    for b1 in s1:
        for b2 in s2:
            # only exclude first strand
            if b1 not in b1_excluded:
                idx_list.append(idx)
                
                #check for specific idxs
                if base == 'control':
                    if s1.index(b1) - (10-s2.index(b2)) ==0:
                        #print(idx_new, idx, (b1, b2))
                        pass
                    idx_new += 1
            idx += 1
    
    return idx_list


def load_trajs(seq, base, temp, max_frames, max_trajs, common_idx=False):
    '''load feature distance for given seq, base, temp
    
    keywords:
    max_frames -- loads last n frames of each traj
    max_trajs -- loads first n trajectories
    common_idxs -- removes 2, 4, 6 base site to ensure common bases
    
    return: max_trajs x max_frames x n_features list of features'''

    # load in npy distance data
    npy_name = glob.glob(f'../abasic_dists/{seq}_msm/{base}*T-{temp}*')[0]
    
    if common_idx:
        fix_idxs = translate_pairs(len(seq), base)
        npy = np.load(npy_name)[:max_trajs, -max_frames:, fix_idxs]
    else:
        npy = np.load(npy_name)[:max_trajs, -max_frames:, :]
    
    return [d for d in npy]


def fit_SRV(traj_list, dim, max_epochs, lag, 
            batch_size=50000, lrate=0.01, val_split=0.0001):
    '''instantiate and fit SRV object, return the object'''
    
    SRV = HDE(
        np.shape(traj_list)[-1], 
        n_components=dim, 
        validation_split=val_split, 
        n_epochs=max_epochs, 
        lag_time=lag, 
        batch_size=batch_size, #500000
        #callbacks=calls,  # comment out for consistet training time
        learning_rate=lrate, 
        batch_normalization=True,
        latent_space_noise=0.0,
        verbose=False
    )
    
    SRV = SRV.fit(traj_list)
    return SRV

def unwrap_coords(xyz, box_L=3.887, ca_idx=13, cb_idx=42):
    '''wrap into specified box size'''
    
    frame_list = []
    for i in range(len(xyz)):
        
        #print('new frame')

        ## check for wrapping in trajs
        strand_a = xyz[i, :29]
        strand_b = xyz[i, 29:]
        #print(strand_a[1::3])

        ## center on two arbitrary (central base pairs)
        ca_mean = xyz[i, ca_idx]
        cb_mean = xyz[i, cb_idx]
        
        # save initial strand dist
        diff_mean = ca_mean - cb_mean
        diff_total = diff_mean + box_L*2*(diff_mean < -box_L) - box_L*2*(diff_mean > box_L) 

        # subtract means of central base xyz
        strand_a_norm = strand_a - ca_mean + box_L
        strand_b_norm = strand_b - cb_mean + box_L
        #print(strand_a_norm[1::3])
        
        # calculate box shifts seperatley to prevent editing arraray
        strand_a_shift = box_L*2 * (strand_a_norm // (box_L*2))
        strand_b_shift = box_L*2 * (strand_b_norm // (box_L*2))
        #print(strand_a_shift[1::3])

        # connect all strands in same box
        strand_a_norm -= strand_a_shift
        strand_b_norm -= strand_b_shift
        #print(strand_a_norm[1::3])

        # once connected, add back in mean between the shifts
        strand_b_norm -= diff_total # this should be correct
    
        # now center all coordinates togther
        full_mean = (np.mean(strand_a_norm, axis=0) + np.mean(strand_b_norm, axis=0))/2
        
        strand_a_norm -= full_mean
        strand_b_norm -= full_mean
    
        xyz_fixed = np.append(strand_a_norm, strand_b_norm, axis=0)
        frame_list.append(xyz_fixed)
    
    print(np.shape(frame_list))
    return np.array(frame_list)

def save_synth_xyz(name, xyz):
    '''save synth xyzs to load as mdtraj objs'''

    n_atoms = np.shape(xyz)[1]
    with open(f'./synth_xyzs/{name}.xyz', 'w') as f:
        for i in range(xyz.shape[0]):
            f.write('%d\n' % n_atoms)
            f.write('\n')
            for k in range(n_atoms):
                # nm to Angstroms for xyz write
                f.write('%3s%17.9f%17.9f%17.9f\n' % 
                        ('C', xyz[i][k][0]*10, xyz[i][k][1]*10, xyz[i][k][2]*10) ) 
                
def process_xyz(traj, ref_pdb='./dna.pdb', save_xyz=False):
    '''input xyz and return unwrapped, connected, and superposed xyz'''
    
    # wrap and save for visuaulization
    xyz_unwrap = unwrap_coords(traj.xyz) 
    ref = md.load(ref_pdb)
    print('unwrapped coords')

    # need to save an intermediate xyz
    save_synth_xyz('temp', xyz_unwrap)
    print('saved xyz')
    
    # load into new traj and superpose 
    traj_unwrap = md.load('./synth_xyzs/temp.xyz', top=ref_pdb)
    traj_unwrap.superpose(ref[0])
    xyz = traj_unwrap.xyz.reshape(-1, traj_unwrap.n_atoms*3)
    print('superposed')
    
    # save an output 
    if save_xyz:
        save_synth_xyz('test_connect_sup', traj_unwrap[::100].xyz)
    
    return xyz