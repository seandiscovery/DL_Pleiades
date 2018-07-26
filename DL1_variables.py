"""
List of variables used in Preprocessing.

Provided for reference; from the DL1 repository. Note that 
for ETE training, the final 5 variables: 
      'rnnip_pb',
      'rnnip_pc',
      'rnnip_pu',
      'rnnip_ptau',
      'rnnip_isDefaults'
are removed, as in that case the RNN and FNN are a single algorithm. 
"""
# make sure that the label is always the first entry
# the order of the variables is important
var_list = ['HadronConeExclTruthLabelID',
            'eta_uncalib',
            'pt_uncalib',
            'JetFitter_mass',
            'JetFitter_isDefaults',
            'JetFitter_energyFraction',
            'JetFitter_significance3d',
            'JetFitter_nVTX',
            'JetFitter_nSingleTracks',
            'JetFitter_nTracksAtVtx',
            'JetFitter_N2Tpair',
            'JetFitter_deltaR',
            'SV1_NGTinSvx',
            'SV1_masssvx',
            'SV1_isDefaults',
            'SV1_N2Tpair',
            'SV1_efracsvx',
            'SV1_deltaR',
            'SV1_Lxy',
            'SV1_L3d',
            'SV1_significance3d',
            'IP2D_bu',
            'IP2D_isDefaults',
            'IP2D_bc',
            'IP2D_cu',
            'IP3D_bu',
            'IP3D_isDefaults',
            'IP3D_bc',
            'IP3D_cu',
            'secondaryVtx_nTrks',
            'secondaryVtx_isDefaults',
            'secondaryVtx_m',
            'secondaryVtx_E',
            'secondaryVtx_EFrac',
            'secondaryVtx_L3d',
            'secondaryVtx_Lxy',
            'secondaryVtx_max_trk_flightDirRelEta',
            'secondaryVtx_min_trk_flightDirRelEta',
            'secondaryVtx_avg_trk_flightDirRelEta',
            'max_trk_flightDirRelEta',
            'min_trk_flightDirRelEta',
            'avg_trk_flightDirRelEta',
            'rnnip_pb',
            'rnnip_pc',
            'rnnip_pu',
            'rnnip_ptau',
            'rnnip_isDefaults'
            ]

