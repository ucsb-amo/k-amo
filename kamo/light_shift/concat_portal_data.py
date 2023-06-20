import os
import pandas as pd

def concat_portal_data():
    '''
    Generates a single csv containing all the matrix element data from the UDel
    atomic physics portal for K, and saves to the WeldLab server.
    '''
    portal_data_folder = r"B:\_K\Resources\udel_potassium_matrix_elements\source"
    dpaths = os.listdir(portal_data_folder)
    dpaths = [os.path.join(portal_data_folder,dpath) for dpath in dpaths]

    data = pd.read_csv(dpaths[0])
    for dpath in dpaths[1:]:
        new_data = pd.read_csv(dpath)
        data = pd.concat([data, new_data], axis=0)

    data.to_csv(os.path.join(portal_data_folder,"K1MatrixElements_complete.csv"))