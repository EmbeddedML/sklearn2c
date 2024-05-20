import os.path as osp
import numpy as np

def np2str(arr):
    np.set_printoptions(threshold=np.inf)
    if type(arr) in (list, tuple):
        arr = np.array(arr)
    str_arr = np.array2string(arr, separator= ",")
    str_arr = str_arr.replace("[", "{").replace("]","}")
    str_arr = str_arr.replace("'","\"")
    return str_arr

class GenericExporter:
    def __init__(self) -> None:
        self.header_str = ""
        self.source_str = ""
        self.filename = ""
        self.config_name = ""

    def export(self, filename):
        self.filename = filename
        self.config_name = osp.basename(self.filename)
        self.create_header()
        self.create_source()
        with open(f'{filename}.h', "w") as header_file:
            header_file.write(self.header_str)
        with open(f'{filename}.c', "w") as source_file:
            source_file.write(self.source_str)

    def create_header(self):
        self.header_str += f'#ifndef {self.config_name.upper()}_H_INCLUDED\n'
        self.header_str += f'#define {self.config_name.upper()}_H_INCLUDED\n'
    
    def create_source(self):
        self.source_str += f'#include "{self.config_name}.h"\n'