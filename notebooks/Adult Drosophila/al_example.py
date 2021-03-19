import numpy as np
import pandas as pd
import os

class HemibrainALSim:
    def __init__(self, file_folder = 'data/', dfpath = 'adult_hallem06.csv', threshold=5., return_flycircuit=True):
        """Creates the AL with a simple model for Hemibrain and FlyCircuit.
        
        # Arguments:
            file_folder (str): The folder with the matrices to load for presynaptically acting and postsynaptically acting LNs.
            dfpath (str): The path to the csv file to be used for loading the affinity values.
        """
        df = pd.read_csv(os.path.join(file_folder, dfpath) )
        all_odorant_names = list(df.iloc[:,0])

        self.preLN_field = np.load(os.path.join(file_folder, 'preLN_field.npy'))
        self.postLN_field = np.load(os.path.join(file_folder, 'postLN_field.npy'))
        self.preLN_field_a = np.load(os.path.join(file_folder, 'preLN_field_a.npy')) # OSN-to-LN
        self.preLN_field_b = np.load(os.path.join(file_folder, 'preLN_field_b.npy')) # LN-to-PN

        glom_names = ['VP3',
                      'VC3l',
                      'VP5',
                      'DL2v',
                      'V',
                      'VL2a',
                      'VC5',
                      'DM4',
                      'DM3',
                      'DA4m',
                      'VP2',
                      'VP1l',
                      'DL2d',
                      'VP1m',
                      'DM5',
                      'DC4',
                      'DA1',
                      'VA3',
                      'VM2',
                      'D',
                      'VL2p',
                      'VM5d',
                      'VA1v',
                      'DL3',
                      'VA7m',
                      'DA2',
                      'VM7d',
                      'VC3m',
                      'VM1',
                      'VM4',
                      'VA4',
                      'DL1',
                      'DC1',
                      'DA4l',
                      'DP1m',
                      'VA2',
                      'VA1d',
                      'DM2',
                      'DP1l',
                      'VC4',
                      'VM7v',
                      'VA5',
                      'VA6',
                      'DC2',
                      'DM1',
                      'DL4',
                      'VA7l',
                      'DM6',
                      'VM3',
                      'VM5v',
                      'VC2',
                      'DA3',
                      'DC3',
                      'DL5',
                      'VC1',
                      'VL1']

        GL_to_OR = {
                'D': {'receptors': ['OR69a','OR69b'], 'name': 'ab9', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DA1': {'receptors': ['OR67d'], 'name': 'at1A', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'DA2': {'receptors': ['OR33a','OR56a'], 'name': 'ab4B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DA3': {'receptors': ['OR23a'], 'name': 'at2B', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'DA4l': {'receptors': ['OR43a'], 'name': 'at3', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'DA4m': {'receptors': ['OR2a'], 'name': 'at3', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'DC1': {'receptors': ['OR19a','OR19b'], 'name': 'at3A', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'DC2': {'receptors': ['OR13a'], 'name': 'ab6A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DC3': {'receptors': ['OR83c'], 'name': 'at2A', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'DC4': {'receptors': ['IR64a'], 'name': 'Sac III', 'co-receptors': ['IR8a'], 'sensillum': 'sacculus', 'sensillum location': 'antenna'},
                'DL1': {'receptors': ['OR10a','GR10a'], 'name': 'ab1D', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DL2d': {'receptors': ['IR75b'], 'name': 'ac3A', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'DL2v': {'receptors': ['IR75c'], 'name': 'ac3A', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'DL3': {'receptors': ['OR65a','OR65b','OR65c'], 'name': 'at4B', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'DL4': {'receptors': ['OR49a','OR85f'], 'name': 'ab10B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DL5': {'receptors': ['OR7a'], 'name': 'ab4A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DM1': {'receptors': ['OR42b'], 'name': 'ab1A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DM2': {'receptors': ['OR22a','OR22b'], 'name': 'ab3A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DM3': {'receptors': ['OR47a','OR33b'], 'name': 'ab5B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DM4': {'receptors': ['OR59b'], 'name': 'ab2A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DM5': {'receptors': ['OR33b','OR85a'], 'name': 'ab2B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DM6': {'receptors': ['OR67a'], 'name': 'ab10B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'DP1l': {'receptors': ['IR75a'], 'name': 'ac2', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'DP1m': {'receptors': ['IR64a'], 'name': 'Sac III', 'co-receptors': ['IR8a'], 'sensillum': 'sacculus', 'sensillum location': 'antenna'},
                'V': {'receptors': ['GR21a','GR63a'], 'name': 'ab1C', 'co-receptors': [], 'sensillum': 'sacculus', 'sensillum location': 'antenna'},
                'DA4l': {'receptors': ['OR43a'], 'name': 'at3', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'VA1d': {'receptors': ['OR88a'], 'name': 'at4C', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'VA1v': {'receptors': ['OR47b'], 'name': 'at4A', 'co-receptors': ['Orco'], 'sensillum': 'trichodea', 'sensillum location': 'antenna'},
                'VA2': {'receptors': ['OR92a'], 'name': 'ab1B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'VA3': {'receptors': ['OR67b'], 'name': 'ab9', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'VA4': {'receptors': ['OR85d'], 'name': 'pb3B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'maxillary pulp'},
                'VA5': {'receptors': ['OR49b'], 'name': 'ab6B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna pulp'},
                'VA6': {'receptors': ['OR82a'], 'name': 'ab5A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna pulp'},
                'VA7l': {'receptors': ['OR46a'], 'name': 'pb2B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'maxillary pulp'},
                'VA7m': {'receptors': [], 'name': '', 'co-receptors': [], 'sensillum': 'unknown', 'sensillum location': 'unknown'},
                'VC1': {'receptors': ['OR33c,OR85e'], 'name': 'pb2A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'maxillary pulp'},
                'VC2': {'receptors': ['OR71a'], 'name': 'pb1B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'maxillary pulp'},
                'VC3l': {'receptors': ['OR35a'], 'name': 'ac1', 'co-receptors': ['Orco'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VC3m': {'receptors': [], 'name': 'unknown', 'co-receptors': ['Orco'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VC4': {'receptors': ['OR67c'], 'name': 'ab7B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'VC5': {'receptors': ['IR41a'], 'name': 'IR25a,IR76b', 'co-receptors': ['Orco'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VL1': {'receptors': ['IR75d'], 'name': 'ac1', 'co-receptors': ['IR25a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VM7d': {'receptors': ['OR42a'], 'name': 'pb1A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'maxillary pulp'},
                'VM7v': {'receptors': ['OR59c'], 'name': 'pb3A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'maxillary pulp'},
                'VM5v': {'receptors': ['OR98a'], 'name': 'ab7A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'VM5d': {'receptors': ['OR85b','OR98b'], 'name': 'ab3B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'VM4': {'receptors': ['IR76a'], 'name': 'ac4', 'co-receptors': ['IR25a','IR76b'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VM3': {'receptors': ['OR9a'], 'name': 'ab8B', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'VM2': {'receptors': ['OR43b'], 'name': 'ab8A', 'co-receptors': ['Orco'], 'sensillum': 'basiconica', 'sensillum location': 'antenna'},
                'VM1': {'receptors': ['OR92a'], 'name': 'ac1', 'co-receptors': ['IR25a','IR76b'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VL2p': {'receptors': ['IR31a'], 'name': 'ac1', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VL2a': {'receptors': ['IR84a'], 'name': 'ac4', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VP3': {'receptors': [], 'name': 'ac4', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VP5': {'receptors': [], 'name': 'ac4', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VP2': {'receptors': [], 'name': 'ac4', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VP1l': {'receptors': [], 'name': 'ac4', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
                'VP1m': {'receptors': [], 'name': 'ac4', 'co-receptors': ['IR8a'], 'sensillum': 'coeloconica', 'sensillum location': 'antenna'},
            }

        idx_a = all_odorant_names.index('putrescine')

        b_array = np.array(list(df.iloc[idx_a,1:]))

        or_names = [i.lower() for i in list(df.columns[1:])]
        bj = []
        or_names_found = []
        or_inds_found = []
        for i, val in enumerate(glom_names):
            receptors = GL_to_OR[val]['receptors']
            found = False
            for j in receptors:
                jreal = j
                j = j.lower()
                if j in or_names:
                    bj.append(b_array[or_names.index(j)])
                    found = True
                    or_names_found.append(jreal)
                    or_inds_found.append(len(bj)-1)
            if found == False:
                bj.append(0)
        self.ngloms = 56
        self.or_inds_found = or_inds_found
        self.or_names_found = or_names_found
        T = 50000
        dt = 1e-4
        bj = np.array(bj)

        I = np.zeros((self.ngloms, T))
        b = bj.reshape((-1,1))
        b = np.repeat(b, T, axis=1)
        I[:,10000:30000] = 5 + 0. * np.repeat(5.*np.random.random((1,20000)), self.ngloms, axis=0) # 1.
        Ib = I*b

        self.OSN = np.diff(np.floor(np.cumsum(Ib,axis=1)/threshold), axis=1)
        self.Ib = Ib
        self.threshold = threshold

        self.base_hpf_gain = 10.
        self.hemi_hpf_gain = 180.
        self.hemi_preLN_gain = 5e3
        self.preLN_linear_inhibition_gain = 1e-2
        self.hemi_amp = 4.
        self.osnf_filter = 20.
        self.alpha = 1.
        self.eps = 1e-1
        self.dt = dt
        self.T = T
                                    
    def sim(self, return_flycircuit=True):
        """Simulates the AL with a simple model for Hemibrain and FlyCircuit.
        
        # Arguments:
            return_flycircuit (bool): Whether to return the FlyCircuit output or not.
        """
        dt = self.dt
        T = self.T
        OSN = np.diff(np.floor(np.cumsum(self.Ib,axis=1)/self.threshold), axis=1)
        W_OSNLN = self.preLN_field_a
        W_PNLN = self.preLN_field_b.T
        preLN = W_PNLN.dot(W_OSNLN.dot(self.OSN))
        preLNbaseline = self.OSN
        OSNf = self.OSN * 0.
        alpha = 1.
        for j in range(1,OSNf.shape[1]):
            OSNf[:,j] = OSNf[:,j-1] + dt * (alpha * self.OSN[:,j]-self.osnf_filter*OSNf[:,j-1])
        XONE = OSN * 0.
        alpha = 1.
        for j in range(1,XONE.shape[1]):
            XONE[:,j] = XONE[:,j-1] + dt * (alpha * (1-XONE[:,j-1])*OSNf[:,j]-10.*XONE[:,j-1])
        XONEf = XONE * 0.
        alpha = 10.
        for j in range(1,OSNf.shape[1]):
            XONEf[:,j] = XONEf[:,j-1] + dt * (alpha * XONE[:,j]-10.0*XONEf[:,j-1])
        OSNh = XONE - XONEf
        OSNh = np.maximum(0., OSNh)
        def lpass(X, alpha=10.):
            Xf = X * 0.
            for j in range(1,X.shape[1]):
                Xf[:,j] = Xf[:,j-1] + dt * (alpha * X[:,j]-alpha*Xf[:,j-1])
            return Xf

        preLN = W_PNLN.dot(W_OSNLN.dot(OSNf))
        preLNbaseline = OSNf
        OSNhnew = W_PNLN.dot(W_OSNLN).dot(OSNf)
        OSNhnew2 = self.postLN_field.dot(OSNh)

        hemimodel = np.maximum(0.,self.hemi_amp * (self.hemi_preLN_gain * OSNf/(self.eps+1.*W_PNLN.dot(W_OSNLN).dot(OSNf))-self.preLN_linear_inhibition_gain*OSNhnew+self.hemi_hpf_gain*OSNhnew2))

        basemodel = preLNbaseline.copy()
        for j in range(1,basemodel.shape[1]):
            if np.sqrt(np.sum((basemodel[:,j])**2))>1e-0:
                basemodel[:,j] = basemodel[:,j] / np.sqrt(np.sum((basemodel[:,j])**2))
        basemodel = lpass(basemodel) + self.base_hpf_gain * OSNh
        if return_flycircuit:
            return hemimodel, basemodel
        else:
            return hemimodel