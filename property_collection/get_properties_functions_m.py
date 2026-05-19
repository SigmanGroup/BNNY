#Original code written by Brittany C. Haas and Melissa A. Hardy (adapted from David B. Vogt's get_properties_pandas.py, adapted from Tobias Gensch)

#modifications for bidentate ligands made by Jamie A. Cadge (contributions from Kjell Jorner, Jordan Dotson and Lucy van Dijk)

import pandas as pd
import numpy as np
import re
import math
import multiprocessing
from morfeus import Sterimol
from morfeus import BuriedVolume
from morfeus import Pyramidalization
from morfeus import SASA
from morfeus import BiteAngle
from morfeus import SolidAngle
from morfeus import VisibleVolume

import goodvibes.GoodVibes as gv
import goodvibes.thermo as thermo
import goodvibes.io as io
import goodvibes.pes as pes

from goodvibes.GoodVibes import ATMOS, GAS_CONSTANT
from goodvibes.io import level_of_theory
from goodvibes.thermo import calc_bbe
from goodvibes.vib_scale_factors import scaling_data_dict, scaling_data_dict_mod

from pathlib import Path
import itertools

# import dbstep.Dbstep as db
#import matplotlib.pyplot as plt
#from matplotlib import rcParams

homo_pattern = re.compile("Alpha  occ. eigenvalues")
npa_pattern = re.compile("Summary of Natural Population Analysis:")
nbo_os_pattern = re.compile("beta spin orbitals")
nbo_occup_pattern = re.compile("Natural Bond Orbitals (Summary):")
nmrstart_pattern = " SCF GIAO Magnetic shielding tensor (ppm):\n"
nmrend_pattern = re.compile("End of Minotr F.D.")
nmrend_pattern_os = re.compile("g value of the free electron")
zero_pattern = re.compile("zero-point Energies")
cputime_pattern = re.compile("Job cpu time:")
walltime_pattern = re.compile("Elapsed time:")
volume_pattern = re.compile("Molar volume =")
polarizability_pattern = re.compile("Dipole polarizability, Alpha")
dipole_pattern = "Dipole moment (field-independent basis, Debye)"
frqs_pattern = re.compile("Red. masses")
frqsend_pattern = re.compile("Thermochemistry")
chelpg1_pattern = re.compile("(CHELPG)")
chelpg2_pattern = re.compile("Charges from ESP fit")
hirshfeld_pattern = re.compile("Hirshfeld charges, spin densities, dipoles, and CM5 charges")

def get_geom(streams): #extracts the geometry from the compressed stream
    geom = []
    for item in streams[-1][16:]:
        if item == "":
            break
        geom.append([item.split(",")[0],float(item.split(",")[-3]),float(item.split(",")[-2]),float(item.split(",")[-1])])
    return(geom)

def get_outstreams(log): #gets the compressed stream information at the end of a Gaussian job
    streams = []
    starts,ends = [],[]
    error = ""
    an_error = True
    try:
        with open(log+".log") as f:
            loglines = f.readlines()
    except:
        with open(log+".LOG") as f:
            loglines = f.readlines()

    for line in loglines[::-1]:
        if "Normal termination" in line:
            an_error = False
        if an_error:
            error = "****Failed or incomplete jobs for " + log + ".log"

    for i in range(len(loglines)):
        if "1\\1\\" in loglines[i]:
            starts.append(i)
        if "@" in loglines[i]:
            ends.append(i)
    #    if "Normal termination" in loglines[i]:
    #        error = ""


    if len(starts) != len(ends) or len(starts) == 0: #probably redundant
        error = "****Failed or incomplete jobs for " + log + ".log"
        return(streams,error)
    for i in range(len(starts)):
        tmp = ""
        for j in range(starts[i],ends[i]+1,1):
            tmp = tmp + loglines[j][1:-1]
        streams.append(tmp.split("\\"))
    return(streams,error)

def get_filecont(log): #gets the entire job output
    error = "" #default unless "normal termination" is in file
    an_error = True
    with open(log+".log") as f:
        loglines = f.readlines()
    for line in loglines[::-1]:
        if "Normal termination" in line:
            an_error = False
        if an_error:
            error = "****Failed or incomplete jobs for " + log + ".log"
    return(loglines, error)

def get_sterimol_morfeus(dataframe, sterimol_list): #uses morfeus to calculate sterimol L, B1, B5 for two input atoms for every entry in df
    sterimol_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = []
            for sterimol in sterimol_list:
                atomnum_list = [] #the atom numbers use to collect sterimol values (i.e. [18 16 17 15]) are collected from the df using the input list (i.e. [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]

            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)

            log_file = row['log_name']
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data",
                    'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data",
                    'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
                continue

            geom = get_geom(streams)


            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "Number of atom inputs given for Sterimol is not divisible by two. " + str(len(sterimol)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += " " + atom + ": Only numbers accepted as input for Sterimol"
                    if int(atom) > len(geom):
                        error += " " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)

            elements = np.array([geom[i][0] for i in range(len(geom))])
            coordinates = np.array([np.array(geom[i][1:]) for i in range(len(geom))])

            #this collects Sterimol values for each pair of inputs
            sterimolout = []
            for sterimol in sterimolnums_list:
                sterimol_values = Sterimol(elements, coordinates, int(sterimol[0]), int(sterimol[1])) #calls morfeus
                sterimolout.append(sterimol_values)


            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': sterimolout[a].L_value,
                'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': sterimolout[a].B_1_value,
                'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': sterimolout[a].B_5_value}
                row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire Morfeus Sterimol parameters for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data",
                    'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data",
                    'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_morfeus': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Morfeus Sterimol function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))

def get_sterimol_dbstep(dataframe, sterimol_list): #uses DBSTEP to calculate sterimol L, B1, B5 for two input atoms for every entry in df
    sterimol_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']

            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = []
            for sterimol in sterimol_list:
                atomnum_list = [] #the atom numbers use to collect sterimol values (i.e. [18 16 17 15]) are collected from the df using the input list (i.e. [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]

            #checks for if the wrong number of atoms are input or input is not of the correct form
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "****Number of atom inputs given for Sterimol is not divisible by two. " + str(len(sterimol)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for Sterimol"
                if error != "": print(error)

            #this collects Sterimol values for each pair of inputs
            sterimol_out = []
            fp = log_file + str(".log")
            for sterimol in sterimolnums_list:
                sterimol_values = db.dbstep(fp,atom1=int(sterimol[0]),atom2=int(sterimol[1]),commandline=True,verbose=False,sterimol=True,measure='grid')
                sterimol_out.append(sterimol_values)

            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)

            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                entry = {'Sterimol_B1_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].Bmin,
                         'Sterimol_B5_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].Bmax,
                         'Sterimol_L_' + str(sterimoltitle_list[a]) + "(Å)_dbstep": sterimol_out[a].L}
                row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire DSBTEP Sterimol parameters for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Sterimol_L_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data",
                    'Sterimol_B1_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data",
                    'Sterimol_B5_' + str(sterimoltitle_list[a]) + '(Å)_dbstep': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("DBSTEP Sterimol function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))

def get_sterimol2vec(dataframe, sterimol_list, end_r, step_size): #uses DBSTEP to calculate sterimol Bmin and Bmax for two input atoms at intervals from 0 to end_r at step_size
    sterimol_dataframe = pd.DataFrame(columns=[])
    num_steps = int((end_r)/step_size + 1)
    radii_list = [0 + step_size*i for i in range(num_steps)]

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']

            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = []
            for sterimol in sterimol_list:
                atomnum_list = [] #the atom numbers use to collect sterimol values (i.e. [18 16 17 15]) are collected from the df using the input list (i.e. [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]

            #checks for if the wrong number of atoms are input or input is not of the correct form
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "Number of atom inputs given for Sterimol is not divisible by two. " + str(len(sterimol)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += " " + atom + ": Only numbers accepted as input for Sterimol"
                if error != "": print(error)

            #this collects Sterimol values for each pair of inputs
            sterimol2vec_out = []
            fp = log_file + str(".log")
            for sterimol in sterimolnums_list:
                sterimol2vec_values = db.dbstep(fp,atom1=int(sterimol[0]),atom2=int(sterimol[1]),scan='0.0:{}:{}'.format(end_r,step_size),commandline=True,verbose=False,sterimol=True,measure='grid')
                sterimol2vec_out.append(sterimol2vec_values)

            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)

            scans = radii_list
            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                for i in range(0, len(scans)):
                    entry = {'Sterimol_Bmin_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": sterimol2vec_out[a].Bmin[i],
                             'Sterimol_Bmax_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": sterimol2vec_out[a].Bmax[i]}
                    row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire DSBTEP Sterimol2Vec parameters for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(sterimolnums_list)):
                    for i in range(0, len(scans)):
                        entry = {'Sterimol_Bmin_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": "no data",
                                'Sterimol_Bmax_' + str(sterimoltitle_list[a]) + "_" + str(scans[i]) + "Å(Å)": "no data"}
                        row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("DBSTEP Sterimol2Vec function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))

def get_vbur_scan(dataframe, a_list, start_r, end_r, step_size): #uses morfeus via get_vbur_one_radius to scan vbur across a range of radii
    num_steps = int((end_r-start_r)/step_size + 1)
    radii = [start_r + step_size*i for i in range(num_steps)]
    frames = []
    for radius in radii:
        for a in a_list:
            frames.append(get_vbur_one_radius(dataframe, a, radius))
    vbur_scan_dataframe = pd.concat(frames, axis = 1)
    print("Vbur scan function has completed for", a_list, "from", start_r, " to ", end_r)
    return(pd.concat([dataframe, vbur_scan_dataframe], axis = 1))

def get_vbur_one_radius(dataframe, a1, radius): #uses morfeus to calculate vbur at a single radius for an atom (a1) in df
    atom = str(a1) # if you enter metal_atom, this is zinc should work for whatever's defined
    vbur_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            atom1 = row[str(a1)] #gets numerical value (e.g. 16) for a1 (e.g. metal_atom, N1 etc)
            exclude1 = row["-H1"] # numerical value for column we always want to exclude
            exclude2 = row["-H2"] #numerical value for column we always want to exclude
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": "no data"}
                vbur_dataframe = vbur_dataframe.append(row_i, ignore_index=True)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))]) #this is every element in the file
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))]) #the xyz coordinates for each element
            vbur = BuriedVolume(elements, coordinates, int(atom1), include_hs=True, radius=radius, excluded_atoms=[exclude1, exclude2]) #calls morfeus
            row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": vbur.percent_buried_volume * 100} #dictionary with column name and vbur value
            vbur_dataframe = vbur_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to acquire Vbur parameters for:', row['log_name'], ".log")
            row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": "no data"}
            vbur_dataframe = vbur_dataframe.append(row_i, ignore_index=True)

    #return(pd.concat([dataframe, vbur_dataframe], axis = 1))
    return(vbur_dataframe)

def get_vbur_one_radius_no_metal(dataframe, a1, radius): #uses morfeus to calculate vbur_nm at a single radius for an atom (a1) in df
      atom = str(a1)
      vbur_dataframe_nm = pd.DataFrame(columns=[])

      for index, row in dataframe.iterrows():
          try:
              log_file = row['log_name']
              atom1 = row[str(a1)] #gets numerical value (e.g. 16) for a1 (e.g. C1)
              streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
              if error != "":
                  print(error)
                  row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": "no data"}
                  vbur_dataframe_nm = vbur_dataframe_nm.append(row_i, ignore_index=True)
                  continue

              log_coordinates = get_geom(streams)
              elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
              coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])
              vbur_nm = BuriedVolume(elements, coordinates, int(atom1), include_hs=True, radius=radius) #calls morfeus
              row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": vbur_nm.percent_buried_volume * 100}
              vbur_dataframe_nm = vbur_dataframe_nm.append(row_i, ignore_index=True)
          except:
              print('****Unable to acquire Vbur parameters for:', row['log_name'], ".log")
              row_i = {'%Vbur_'+str(atom)+"_"+str(radius)+"Å": "no data"}
              vbur_dataframe_nm = vbur_dataframe_nm.append(row_i, ignore_index=True)
      return(vbur_dataframe_nm)

def get_vbur_scan_no_metal (dataframe, a_list, start_r, end_r, step_size): #uses morfeus via get_vbur_one_radius to scan vbur_nm across a range of radii
      num_steps = int((end_r-start_r)/step_size + 1)
      radii = [start_r + step_size*i for i in range(num_steps)]
      frames = []
      for radius in radii:
          for a in a_list:
              frames.append(get_vbur_one_radius_no_metal(dataframe, a, radius))
      vbur_scan_dataframe_nm = pd.concat(frames, axis = 1)
      print("Vbur scan function has completed for", a_list, "from", start_r, " to ", end_r)
      return(pd.concat([dataframe, vbur_scan_dataframe_nm], axis = 1))


def get_pyramidalization(dataframe, a_list): #uses morfeus to calculate pyramidalization (based on the 3 atoms in closest proximity to the defined atom) for for all atoms (a_list, of form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    pyr_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            atom_list = []
            for label in a_list:
                atom = row[str(label)] #the atom number (i.e. 16) to add to the list is the df entry of this row for the labeled atom (i.e. "C1")
                atom_list.append(str(atom)) #append that to atom_list to make a list of the form [16, 17, 29]

            log_file = row['log_name']
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(atom_list)):
                    entry = {'pyramidalization_Gavrish_' + str(a_list[a]) + '(°)': "no data",
                             'pyramidalization_Agranat-Radhakrishnan_' + str(a_list[a]): "no data"} #details on these values can be found here: https://kjelljorner.github.io/morfeus/pyramidalization.html
                    row_i.update(entry)
                pyr_dataframe = pyr_dataframe.append(row_i, ignore_index=True)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            pyrout = []
            for atom in atom_list:
                pyr = Pyramidalization(coordinates, int(atom)) #calls morfeus
                pyrout.append(pyr)

            row_i = {}
            for a in range(0, len(atom_list)):
                entry = {'pyramidalization_Gavrish_' + str(a_list[a]) + '(°)': pyrout[a].P_angle,
                'pyramidalization_Agranat-Radhakrishnan_' + str(a_list[a]): pyrout[a].P} #details on these values can be found here: https://kjelljorner.github.io/morfeus/pyramidalization.html
                row_i.update(entry)
            pyr_dataframe = pyr_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire pyramidalizataion parameters for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(atom_list)):
                entry = {'pyramidalization_Gavrish_' + str(a_list[a]) + '(°)': "no data",
                'pyramidalization_Agranat-Radhakrishnan_' + str(a_list[a]): "no data"} #details on these values can be found here: https://kjelljorner.github.io/morfeus/pyramidalization.html
                row_i.update(entry)
            pyr_dataframe = pyr_dataframe.append(row_i, ignore_index=True)
    print("Pyramidalization function has completed for", a_list)
    return(pd.concat([dataframe, pyr_dataframe], axis = 1))

def get_specdata(atoms,prop): #input a list of atom numbers of interest and a list of pairs of all atom numbers and property of interest for use with NMR, NBO, possibly others with similar output structures
    propout = []
    for atom in atoms:
        if atom.isdigit():
            a = int(atom)-1
            if a <= len(prop):
                propout.append(float(prop[a][1]))
            else: continue
        else: continue
    return(propout)

def get_nbo(dataframe, a_list): #a function to get the nbo for all atoms (a_list, form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    nbo_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            atomnum_list = []
            for atom in a_list:
                atomnum = row[str(atom)] #the atom number (i.e. 16) to add to the list is the df entry of this row for the labeled atom (i.e. "C1")
                atomnum_list.append(str(atomnum)) #append that to atomnum_list to make a list of the form [16, 17, 29]

            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'NBO_charge_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                nbo_dataframe = nbo_dataframe.append(row_i, ignore_index=True)
                continue

            nbo,nbostart,nboout,skip = [],0,"",0
            #this section finds the line (nbostart) where the nbo data is located
            for i in range(len(filecont)-1,0,-1): #search the file contents for the phrase "beta spin orbitals" to check for open shell molecules
                if re.search(nbo_os_pattern,filecont[i]) and skip == 0:
                    skip = 2 # retrieve only combined orbitals NPA in open shell molecules
                if npa_pattern.search(filecont[i]): #search the file content for the phrase which indicates the start of the NBO section
                    if skip != 0:
                        skip = skip-1
                        continue
                    nbostart = i + 6 #skips the set number of lines between the search key and the start of the table
                    break
            if nbostart == 0:
                error = "****no Natural Population Analysis found in: " + str(row['log_name']) + ".log"
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'NBO_charge_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                nbo_dataframe = nbo_dataframe.append(row_i, ignore_index=True)
                continue

            #this section splits the table where nbo data is located into just the atom number and charge to generate a list of lists (nbo)
            ls = []
            for line in filecont[nbostart:]:
                if "==" in line: break
                ls = [str.split(line)[1],str.split(line)[2]]
                nbo.append(ls)

            #this uses the nbo list to return only the charges for only the atoms of interest as a list (nboout)
            nboout = get_specdata(atomnum_list,nbo)

            #this adds the data from the nboout into the new property df
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'NBO_charge_'+str(a_list[a]): nboout[a]}
                row_i.update(entry)
            #print(row_i)
            nbo_dataframe = nbo_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire NBO charges for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'NBO_charge_'+str(a_list[a]): "no data"}
                row_i.update(entry)
            nbo_dataframe = nbo_dataframe.append(row_i, ignore_index=True)
    print("NBO function has completed for", a_list)
    return(pd.concat([dataframe, nbo_dataframe], axis = 1))

def get_nmr(dataframe, a_list): # a function to get the nbo for all atoms (a_list, form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    nmr_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe

        #if True:
        try: #try to get the data
            atom_list = []
            for new_a in a_list:
                new_atom = row[str(new_a)] #the atom number (i.e. 16) to add to the list is the df entry of this row for the labeled atom (i.e.) "C1")
                atom_list.append(str(new_atom)) #append that to atom_list to make a list of the form [16, 17, 29]
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'NMR_shift_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                nmr_dataframe = nmr_dataframe.append(row_i, ignore_index=True)
                continue

            #determining the locations/values for start and end of NMR section
            start,end,i = 0,0,0
            if nmrstart_pattern in filecont:
                start = filecont.index(nmrstart_pattern)+1
                for i in range(start,len(filecont),1):
                    if nmrend_pattern.search(filecont[i]) or nmrend_pattern_os.search(filecont[i]):
                        end = i
                        break
            if start == 0:
                error = "****no NMR data found in file: " + str(row['log_name']) + ".log"
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'NMR_shift_'+str(a_list[a]): "no data", 'aniso_NMR_shift_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                nmr_dataframe = nmr_dataframe.append(row_i, ignore_index=True)
                continue

            atoms = int((end - start)/5) #total number of atoms in molecule (there are 5 lines generated per atom)
            nmr = []
            aniso_nmr = []
            for atom in range(atoms):
                #print(filecont[start+5*atom])
                element = str.split(filecont[start+5*atom])[1]
                shift_s = str.split(filecont[start+5*atom])[4]
                aniso_shift = str.split(filecont[start+5*atom])[7]
                #print(aniso_shift)
                nmr.append([element,shift_s])
                aniso_nmr.append([element,aniso_shift])
            #atom_list = ["1", "2", "3"]
            nmrout = get_specdata(atom_list,nmr) #revisit
            aniso_nmrout = get_specdata(atom_list,aniso_nmr)
            #print(nmrout)
            #print(aniso_nmrout)

            #this adds the data from the nboout into the new property df
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'NMR_shift_'+str(a_list[a]): nmrout[a], 'aniso_NMR_shift_'+str(a_list[a]): aniso_nmrout[a]}
                row_i.update(entry)
            nmr_dataframe = nmr_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire NMR shifts for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'NMR_shift_'+str(a_list[a]): "no data", 'aniso_NMR_shift_'+str(a_list[a]): "no data"}
                row_i.update(entry)
            nmr_dataframe = nmr_dataframe.append(row_i, ignore_index=True)
    print("NMR function has completed for", a_list)
    return(pd.concat([dataframe, nmr_dataframe], axis = 1))

def get_angles(dataframe,angle_list): # a function to get the angles for all atoms (angle_list, form [[O3, C1, O2], [C4, C1, O3]]) in a dataframe that contains file name and atom number
    angle_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:
            #parsing the angle list from input line
            anglenums_list = []
            for angle in angle_list:
                atomnum_list = [] #the atom numbers for an angle (i.e. 17 16 18) are collected from the df using the input list (i.e.["O3", "C1", "O2"])
                for atom in angle:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                anglenums_list.append(atomnum_list) #append atomnum_list for each angle to make a list of the form [['17', '16', '18'], ['15', '16', '17']]

            angletitle_list = []
            for angle in angle_list:
                angletitle = str(angle[0]) + "_" + str(angle[1]) + "_" + str(angle[2])
                angletitle_list.append(angletitle)

            log_file = row['log_name'] #read file name from df
            streams, error = get_outstreams(log_file)
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(anglenums_list)):
                    entry = {'angle_'+str(angletitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                angle_dataframe = angle_dataframe.append(row_i, ignore_index=True)
                continue

            geom = get_geom(streams)

            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule.
            error = ""
            for angle in anglenums_list:
                if len(angle)%3 != 0:
                    error = "****Number of atom inputs given for angle is not divisible by three. " + str(len(angle)) + " atoms were given. "
                for atom in angle:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for angles"
                    if int(atom) > len(geom):
                        error += "**** " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)

            anglesout = []
            for angle in anglenums_list:
                a = geom[int(angle[0])-1][:4] # atom coords
                b = geom[int(angle[1])-1][:4]
                c = geom[int(angle[2])-1][:4]
                ba = np.array(a[1:]) - np.array(b[1:])
                bc = np.array(c[1:]) - np.array(b[1:])
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                anglevalue = np.arccos(cosine_angle)

                anglesout.append(float(round(np.degrees(anglevalue),3)))

            #this adds the data from the anglesout into the new property df
            row_i = {}
            for a in range(0, len(anglenums_list)):
                entry = {'angle_'+str(angletitle_list[a]) + '(°)': anglesout[a]}
                row_i.update(entry)
            angle_dataframe = angle_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire angles for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(anglenums_list)):
                    entry = {'angle_'+str(angletitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                angle_dataframe = angle_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Angles function has completed for", angle_list)
    return(pd.concat([dataframe, angle_dataframe], axis = 1))

def get_dihedral(dataframe,dihedral_list): # a function to get the dihedrals for all atoms (dihederal_list, form [[O2, C1, O3, H5], [C4, C1, O3, H5]]) in a dataframe that contains file name and atom number
    dihedral_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:
            #parsing the dihedral list from input line
            dihedralnums_list = []
            for dihedral in dihedral_list:
                atomnum_list = [] #the atom numbers for an dihedral (i.e. 18 16 17 50) are collected from the df using the input list (i.e.["O2", "C1", "O3", "H5"])
                for atom in dihedral:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                dihedralnums_list.append(atomnum_list) #append atomnum_list for each dihedral to make a list of the form [['18', '16', '17', '50'], ['18', '16', '17', '50']]
            dihedraltitle_list = []
            for dihedral in dihedral_list:
                dihedraltitle = str(dihedral[0]) + "_" + str(dihedral[1]) + "_" + str(dihedral[2]) + "_" +str(dihedral[3])
                dihedraltitle_list.append(dihedraltitle)

            log_file = row['log_name'] #read file name from df
            streams, error = get_outstreams(log_file)
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(dihedralnums_list)):
                    entry = {'dihedral_'+str(dihedraltitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                dihedral_dataframe = dihedral_dataframe.append(row_i, ignore_index=True)
                continue
            geom = get_geom(streams)

            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule.
            error = ""
            for dihedral in dihedralnums_list:
                if len(dihedral)%4 != 0:
                    error = "****Number of atom inputs given for dihedral angle is not divisible by four. " + str(len(dihedral)) + " atoms were given. "
                for atom in dihedral:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for dihedral angles"
                    if int(atom) > len(geom):
                        error += "**** " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)

            dihedralsout = []
            for dihedral in dihedralnums_list:
                a = geom[int(dihedral[0])-1][:4] # atom coords
                b = geom[int(dihedral[1])-1][:4]
                c = geom[int(dihedral[2])-1][:4]
                d = geom[int(dihedral[3])-1][:4]

                ab = np.array([a[1]-b[1],a[2]-b[2],a[3]-b[3]]) # vectors
                bc = np.array([b[1]-c[1],b[2]-c[2],b[3]-c[3]])
                cd = np.array([c[1]-d[1],c[2]-d[2],c[3]-d[3]])

                n1 = np.cross(ab,bc) # normal vectors
                n2 = np.cross(bc,cd)

                dihedral = round(np.degrees(np.arccos(np.dot(n1,n2) / (np.linalg.norm(n1)*np.linalg.norm(n2)))),3)
                dihedralsout.append(float(dihedral))

            #this adds the data from the dihedralsout into the new property df
            row_i = {}
            for a in range(0, len(dihedralnums_list)):
                entry = {'dihedral_'+str(dihedraltitle_list[a]) + '(°)': dihedralsout[a]}
                row_i.update(entry)
            dihedral_dataframe = dihedral_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire dihedral angles for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(dihedralnums_list)):
                    entry = {'dihedral_'+str(dihedraltitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                dihedral_dataframe = dihedral_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Dihedral function has completed for", dihedral_list)
    return(pd.concat([dataframe, dihedral_dataframe], axis = 1))

def get_distance(dataframe,dist_list): # a function to get the distances for all atoms (dist_list, form [[C1, O2], [C4, C1]]) in a dataframe that contains file name and atom number
    dist_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:
            #parsing the distances list from input line
            distnums_list = []
            for dist in dist_list:
                atomnum_list = [] #the atom numbers for a distance (i.e. 18 16 16 15) are collected from the df using the input list (i.e.["O2", "C1", "O3", "H5"])
                for atom in dist:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                distnums_list.append(atomnum_list) #append atomnum_list for each distance to make a list of the form [['18', '16'], ['16', '15']]

            disttitle_list = []
            for dist in dist_list:
                disttitle = str(dist[0]) + "_" + str(dist[1])
                disttitle_list.append(disttitle)

            log_file = row['log_name'] #read file name from df
            streams, error = get_outstreams(log_file)
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(distnums_list)):
                    entry = {'distance_' + str(disttitle_list[a]) + '(Å)': "no data"}
                    row_i.update(entry)
                dist_dataframe = dist_dataframe.append(row_i, ignore_index=True)
                continue
            geom = get_geom(streams)


            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule.
            error = ""
            for dist in distnums_list:
                if len(dist)%2 != 0:
                    error = "****Number of atom inputs given for distance is not divisible by two. " + str(len(dist)) + " atoms were given. "
                for atom in dist:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for distances"
                    if int(atom) > len(geom):
                        error += "**** " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)

            distout = []
            for dist in distnums_list:
                a = geom[int(dist[0])-1][:4] # Atomcoords
                b = geom[int(dist[1])-1][:4]
                ba = np.array(a[1:]) - np.array(b[1:])
                dist = round(np.linalg.norm(ba),5)
                distout.append(float(dist))

            #this adds the data from the distout into the new property df

            row_i = {}
            for a in range(0, len(distnums_list)):
                entry = {'distance_' + str(disttitle_list[a]) + '(Å)': distout[a]}
                row_i.update(entry)
            dist_dataframe = dist_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire distance for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(distnums_list)):
                    entry = {'distance_' + str(disttitle_list[a]) + '(Å)': "no data"}
                    row_i.update(entry)
                dist_dataframe = dist_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Distance function has completed for", dist_list)
    return(pd.concat([dataframe, dist_dataframe], axis = 1))

def get_enthalpies(dataframe): # gets thermochemical data from freq jobs
    enthalpy_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont = get_filecont(log_file) #read the contents of the log file

            evals = []
            error = "no thermochemical data found;;"
            e_hf,ezpe,h,g = 0,0,0,0
            for i in range(len(filecont)-1): #uses the zero_pattern that denotes this section to gather relevant energy terms
                if zero_pattern.search(filecont[i]):
                    e_hf = round(-eval(str.split(filecont[i-4])[-2]) + ezpe,6)
                    evals.append(e_hf)
                    ezpe = eval(str.split(filecont[i])[-1])
                    evals.append(ezpe)
                    h = eval(str.split(filecont[i+2])[-1])
                    evals.append(h)
                    g = eval(str.split(filecont[i+3])[-1])
                    evals.append(g)
                    error = ""

            #this adds the data from the energy_values list (evals) into the new property df
            row_i = {'ZP_correction(Hartree)': evals[0], 'E_ZPE(Hartree)': evals[1], 'H(Hartree)': evals[2], 'G(Hartree)': evals[3]}
            #print(row_i)

            enthalpy_dataframe = enthalpy_dataframe.append(row_i, ignore_index=True)
        except:
            print('Unable to acquire enthalpies for:', row['log_name'], ".log")
    print("Enthalpies function has completed")
    return(pd.concat([dataframe, enthalpy_dataframe], axis = 1))

def get_time(dataframe): # gets wall time and CPU for all jobs
    time_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'CPU_time_total(hours)': "no data", 'Wall_time_total(hours)': "no data"}
                time_dataframe = time_dataframe.append(row_i, ignore_index=True)
                continue

            cputime,walltime = 0,0
            timeout = []
            for line in filecont:
                if cputime_pattern.search(line):
                    lsplt = str.split(line)
                    cputime = float(lsplt[-2])/3600 + float(lsplt[-4])/60 + float(lsplt[-6]) + float(lsplt[-8])*24
                    timeout.append(round(cputime,5))
                if walltime_pattern.search(line):
                    lsplt = str.split(line)
                    walltime = float(lsplt[-2])/3600 + float(lsplt[-4])/60 + float(lsplt[-6]) + float(lsplt[-8])*24
                    timeout.append(walltime)
            CPU_time = 0
            Wall_time = 0
            for i in range(len(timeout)):
                if i%2 == 0:
                    CPU_time += timeout[i]
                if i%2 != 0:
                    Wall_time += timeout[i]

            #this adds the data from the CPU_time and Wall_time into the property df
            row_i = {'CPU_time_total(hours)': CPU_time, 'Wall_time_total(hours)': Wall_time}
            time_dataframe = time_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire CPU time and wall time for:', row['log_name'], ".log")
            row_i = {'CPU_time_total(hours)': "no data", 'Wall_time_total(hours)': "no data"}
            time_dataframe = time_dataframe.append(row_i, ignore_index=True)
    print("Time function has completed")
    return(pd.concat([dataframe, time_dataframe], axis = 1))

def get_frontierorbs(dataframe): # homo,lumo energies and derived values of last job in file
    frontierorbs_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'HOMO': "no data", 'LUMO': "no data", "μ": "no data", "η": "no data", "ω": "no data"}
                frontierorbs_dataframe = frontierorbs_dataframe.append(row_i, ignore_index=True)
                continue

            frontierout = []
            index = 0
            for line in filecont[::-1]:
                if homo_pattern.search(line):
                    index += 1 #index ensures only the first entry is included
                    if index == 1:
                        homo = float(str.split(line)[-1])
                        lumo = float(str.split(filecont[filecont.index(line)+1])[4])
                        mu = (homo+lumo)/2 # chemical potential or negative of molecular electronegativity
                        eta = lumo-homo # hardness/softness
                        omega = round(mu**2/(2*eta),5) # electrophilicity index
                        frontierout.append(homo)
                        frontierout.append(lumo)
                        frontierout.append(mu)
                        frontierout.append(eta)
                        frontierout.append(omega)

            #this adds the data from the frontierout into the new property df
            row_i = {'HOMO': frontierout[0], 'LUMO': frontierout[1], "μ": frontierout[2], "η": frontierout[3], "ω": frontierout[4]}
            frontierorbs_dataframe = frontierorbs_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire frontier orbitals for:', row['log_name'], ".log")
            row_i = {'HOMO': "no data", 'LUMO': "no data", "μ": "no data", "η": "no data", "ω": "no data"}
            frontierorbs_dataframe = frontierorbs_dataframe.append(row_i, ignore_index=True)
    print("Frontier orbitals function has completed")
    return(pd.concat([dataframe, frontierorbs_dataframe], axis = 1))

def get_volume(dataframe): #gets the molar volume of the molecule
    volume_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'volume(Bohr_radius³/mol)': "no data"}
                volume_dataframe = volume_dataframe.append(row_i, ignore_index=True)
                continue

            volume = []
            for line in filecont:
                if volume_pattern.search(line):
                    volume.append(line.split()[3])
            #this adds the data into the new property df
            row_i = {'volume(Bohr_radius³/mol)': float(volume[0])}
            volume_dataframe = volume_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to acquire volume for:', row['log_name'], ".log")
            row_i = {'volume(Bohr_radius³/mol)': "no data"}
            volume_dataframe = volume_dataframe.append(row_i, ignore_index=True)
    print("Volume function has completed")
    return(pd.concat([dataframe, volume_dataframe], axis = 1))


def get_polarizability(dataframe): # polarizability isotropic and anisotropic
    polarizability_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'polar_iso(Debye)': "no data", 'polar_aniso(Debye)': "no data"}
                polarizability_dataframe = polarizability_dataframe.append(row_i, ignore_index=True)
                continue

            polarout = []
            for i in range(len(filecont)-1,1,-1):
                if polarizability_pattern.search(filecont[i]):
                    alpha_iso = float(filecont[i+4].split()[1].replace("D","E"))
                    alpha_aniso = float(filecont[i+5].split()[1].replace("D","E"))
                    polarout.append(alpha_iso)
                    polarout.append(alpha_aniso)


            #this adds the data from the polarout into the new property df
            row_i = {'polar_iso(Debye)': polarout[0], 'polar_aniso(Debye)': polarout[1]}
            polarizability_dataframe = polarizability_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to acquire polarizability for:', row['log_name'], ".log")
            row_i = {'polar_iso(Debye)': "no data", 'polar_aniso(Debye)': "no data"}
            polarizability_dataframe = polarizability_dataframe.append(row_i, ignore_index=True)
    print("Polarizability function has completed")
    return(pd.concat([dataframe, polarizability_dataframe], axis = 1))

def get_planeangle(dataframe,planeangle_list): # a function to get the plane angles for all atoms (dihederal_list, form [[O2, C1, O3, H5], [C4, C1, O3, H5]]) in a dataframe that contains file name and atom number
    planeangle_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:
            #parsing the plane angle list from input line
            planeanglenums_list = []
            for planeangle in planeangle_list:
                atomnum_list = [] #the atom numbers for a plane angle (i.e. 18 16 17 50) are collected from the df using the input list (i.e.["O2", "C1", "O3", "H5"])
                for atom in planeangle:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                planeanglenums_list.append(atomnum_list) #append atomnum_list for each plane angle to make a list of the form [['18', '16', '17', '50'], ['18', '16', '17', '50']]

            planeangletitle_list = []
            for planeangle in planeangle_list:
                planeangletitle = str(planeangle[0]) + "_" + str(planeangle[1]) + "_" + str(planeangle[2]) + "_&_" +str(planeangle[3])+ "_" + str(planeangle[4]) + "_" +str(planeangle[5])
                planeangletitle_list.append(planeangletitle)

            log_file = row['log_name'] #read file name from df
            streams, error = get_outstreams(log_file)
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(planeanglenums_list)):
                    entry = {'planeangle_'+str(planeangletitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                planeangle_dataframe = planeangle_dataframe.append(row_i, ignore_index=True)
                continue

            geom = get_geom(streams)


            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule.
            error = ""
            for planeangle in planeanglenums_list:
                if len(planeangle)%6 != 0:
                    error = "****Number of atom inputs given for plane angle is not divisible by six. " + str(len(planeangle)) + " atoms were given. "
                for atom in planeangle:
                    if not atom.isdigit():
                        error += "**** " + atom + ": Only numbers accepted as input for plane angles"
                    if int(atom) > len(geom):
                        error += "**** " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)

            planeanglesout = []
            for planeangle in planeanglenums_list:
                a = geom[int(planeangle[0])-1][:4]
                b = geom[int(planeangle[1])-1][:4]
                c = geom[int(planeangle[2])-1][:4]
                d = geom[int(planeangle[3])-1][:4]
                e = geom[int(planeangle[4])-1][:4]
                f = geom[int(planeangle[5])-1][:4]

                ab = np.array([a[1]-b[1],a[2]-b[2],a[3]-b[3]]) # Vectors
                bc = np.array([b[1]-c[1],b[2]-c[2],b[3]-c[3]])
                de = np.array([d[1]-e[1],d[2]-e[2],d[3]-e[3]])
                ef = np.array([e[1]-f[1],e[2]-f[2],e[3]-f[3]])

                n1 = np.cross(ab,bc) # Normal vectors
                n2 = np.cross(de,ef)

                planeangle_value = round(np.degrees(np.arccos(np.dot(n1,n2) / (np.linalg.norm(n1)*np.linalg.norm(n2)))),3)
                planeangle_value = min(abs(planeangle_value),abs(180-planeangle_value))
                planeanglesout.append(planeangle_value)


            #this adds the data from the planeanglesout into the new property df
            row_i = {}
            for a in range(0, len(planeanglenums_list)):
                entry = {'planeangle_'+str(planeangletitle_list[a]) + '(°)': planeanglesout[a]}
                row_i.update(entry)
            planeangle_dataframe = planeangle_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire plane angle for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(planeanglenums_list)):
                    entry = {'planeangle_'+str(planeangletitle_list[a]) + '(°)': "no data"}
                    row_i.update(entry)
                planeangle_dataframe = planeangle_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Plane angle function has completed for", planeangle_list)
    return(pd.concat([dataframe, planeangle_dataframe], axis = 1))

def get_dipole(dataframe):
    dipole_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try: #try to get the data
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {'dipole(Debye)': "no data"}
                dipole_dataframe = dipole_dataframe.append(row_i, ignore_index=True)
                continue

            dipole = []
            for i in range(len(filecont)-1,0,-1): #search filecont in backwards direction
                if dipole_pattern in filecont[i]:
                    dipole.append(float(str.split(filecont[i+1])[-1]))
            #this adds the data from the first dipole entry (corresponding to the last job in the file) into the new property df
            row_i = {'dipole(Debye)': dipole[0]}
            dipole_dataframe = dipole_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire dipole for:', row['log_name'], ".log")
            row_i = {'dipole(Debye)': "no data"}
            dipole_dataframe = dipole_dataframe.append(row_i, ignore_index=True)
    print("Dipole function has completed")
    return(pd.concat([dataframe, dipole_dataframe], axis = 1))

def get_SASA(dataframe): #uses morfeus to calculate solvent accessible surface area in a dataframe that contains file name
    #if you want to SASA with different probe radii, morfeus has this functionality, but it has not been implemented here
    sasa_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {'SASA_surface_area(Å²)': "no data",
                     'SASA_volume(Å³)': "no data",
                     'SASA_sphericity': "no data"}
                sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            sasa = SASA(elements,coordinates) #calls morfeus

            sphericity = np.cbrt((36*math.pi*sasa.volume**2))/sasa.area

            row_i = {'SASA_surface_area(Å²)': sasa.area,
                     'SASA_volume(Å³)': sasa.volume, #volume inside the solvent accessible surface area
                     'SASA_sphericity': sphericity}
            sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire SASA parameters for:', row['log_name'], ".log")
            row_i = {'SASA_surface_area(Å²)': "no data",
                     'SASA_volume(Å³)': "no data",
                     'SASA_sphericity': "no data"}
            sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True)
    print("SASA function has completed")
    return(pd.concat([dataframe, sasa_dataframe], axis = 1))

def _get_goodvibes_freq_scale_factor(file: Path):
    '''
    Replicate the GoodVibes 3.2 automatic vibrational scale-factor lookup.

    Parameters
    ----------
    filename: str
        Output file to inspect.

    Returns
    -------
    freq_scale_factor: float
        Vibrational scale factor used by GoodVibes.
    '''
    configure_logger(debug=False)

    # Detect the level of theory the same way GoodVibes does.
    level = level_of_theory(file=file).upper()

    # Search the built-in GoodVibes scale-factor tables.
    for data in (scaling_data_dict, scaling_data_dict_mod):
        if level in data:

            # This must be returned as type float because
            # It was specified as float32 (f4) in goodvibes
            return float(data[level].zpe_fac)

    # Match the GoodVibes fallback when no match is found.
    return 1.0


def _get_goodvibes_thermo_data(logfile: Path | str,
                               temp: float = 298.15,
                               spc: str = 'link'):
    '''
    Helper function that mimics the old GoodVibes workflow and returns
    thermochemical data as a dict.

    Parameters
    ----------
    logfile: str
        Gaussian/ORCA output file path. A bare stem is also accepted.

    temp: float
        Temperature in Kelvin.

    spc: str
        Single-point correction mode. Use 'link' to match the old code.

    Returns
    -------
    thermo_data: dict
        Thermochemical data extracted from the GoodVibes calc_bbe object.
    '''
    try:
        # Match the GoodVibes gas-phase default concentration when -c is not supplied.
        conc = ATMOS / (GAS_CONSTANT * temp)

        # Match GoodVibes automatic vibrational scale-factor detection.
        freq_scale_factor = _get_goodvibes_freq_scale_factor(logfile)

        # Call the real GoodVibes 3.2 thermochemistry engine directly.
        bbe = calc_bbe(
            file=logfile,
            QS='grimme',
            QH=False,
            s_freq_cutoff=100.0,
            H_FREQ_CUTOFF=100.0,
            temperature=temp,
            conc=conc,
            freq_scale_factor=freq_scale_factor,
            solv='none',
            spc=spc,
            invert=False,
            d3_term=0.0,
            cosmo=None,
            ssymm=False,
            mm_freq_scale_factor=False,
            inertia='global',
            g4=False,
        )

        # Return the same values the old version
        thermo_data =  pd.Series({
            'log_name': logfile.name,
            'E_spc (Hartree)': bbe.sp_energy,
            'ZPE(Hartree)': bbe.zpe,
            'H_spc(Hartree)': bbe.enthalpy,
            'T*S': bbe.entropy * temp,
            'T*qh_S': bbe.qh_entropy * temp,
            'G(T)_spc(Hartree)': bbe.gibbs_free_energy,
            'qh_G(T)_spc(Hartree)': bbe.qh_gibbs_free_energy,
            'T': temp
        })

    except Exception as e:
        thermo_data =  pd.Series({
            'log_name': logfile.name,
            'E_spc (Hartree)': None,
            'ZPE(Hartree)': None,
            'H_spc(Hartree)': None,
            'T*S': None,
            'T*qh_S': None,
            'G(T)_spc(Hartree)': None,
            'qh_G(T)_spc(Hartree)': None,
            'T': temp
        })

    return pd.DataFrame(thermo_data).transpose()


def get_goodvibes_e(dataframe: pd.DataFrame,
                    data_dir: Path,
                    temp: float = 298.15,
                    procs: int = 1):
    '''
    Extracts the following properties

    - E_spc (Hartree)
    - ZPE(Hartree)
    - H_spc(Hartree)
    - T*S
    - T*qh_S
    - G(T)_spc(Hartree)
    - qh_G(T)_spc(Hartree)
    - T

    Parameters
    ----------
    dataframe: pd.DataFrame
        DataFrame containing `'log_name'` column

    data_dir: Path
        Directory where the files are located

    temp: float
        Temperature in Kelvin

    procs: int
        Number of processors

    Returns
    ----------
    pd.DataFrame
        The DataFrame containing the `'log_name'` column and
        the resultant descriptors
    '''
    files = [Path(data_dir / x) for x in dataframe['log_name'].to_list()]

    args = zip(files,
            itertools.repeat(temp),
            itertools.repeat('link'))

    with multiprocessing.Pool(processes=procs) as p:
        results = p.starmap(_get_goodvibes_thermo_data, args)

    results = pd.concat(results)

    results.set_index('log_name', inplace=True, drop=True)
    dataframe.set_index('log_name', inplace=True, drop=True)

    dataframe = pd.concat([dataframe, results], axis=1)
    dataframe.reset_index(inplace=True)

    print('GoodVibes function completed.')
    return dataframe


class IR:
    def __init__(self,filecont,start,col,len):
        self.freqno = int(filecont[start].split()[-3+col])
        self.freq = float(filecont[start+2].split()[-3+col])
        self.int = float(filecont[start+5].split()[-3+col])
        self.deltas = []
        atomnos = []
        for a in range(len-7):
            atomnos.append(filecont[start+7+a].split()[1])
            x = float(filecont[start+7+a].split()[3*col+2])
            y = float(filecont[start+7+a].split()[3*col+3])
            z = float(filecont[start+7+a].split()[3*col+4])
            self.deltas.append(np.linalg.norm([x,y,z]))


def get_IR(dataframe, a1, a2, freqmin, freqmax, intmin, intmax, threshold): # a function to get IR values for a pair of atoms at a certain freq and intensity
    IR_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in
    pair_label = str(a1)+"_"+str(a2)

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        #if True:
        try:
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file)
            if error != "":
                print(error)
                row_i = {'IR_freq_'+str(pair_label): "no data"}
                IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
                continue
            #this changes a1 and a2 (of the form "C1" and "O3") to atomnum_pair (of the form [17, 18])
            atom1 = row[str(a1)]
            atom2 = row[str(a2)]

            #this section finds where all IR frequencies are located in the log file
            frq_len = 0
            frq_end = 0
            for i in range(len(filecont)):
                if frqs_pattern.search(filecont[i]) and frq_len == 1: #subsequent times it finds the pattern, it recognizes the frq_len
                    frq_len = i -3 - frq_start
                if frqs_pattern.search(filecont[i]) and frq_len == 0: #first time it finds the pattern it will set frq_start
                    frq_start = i-3
                    frq_len = 1
                if frqsend_pattern.search(filecont[i]): #finds the end pattern
                    frq_end = i-3

            nfrq = filecont[frq_end-frq_len+1].split()[-1]
            blocks = int((frq_end + 1 - frq_start)/frq_len)
            irdata = []   # list of objects. IR contains: IR.freq, IR.int, IR.deltas = []

            for i in range(0, blocks):
                for j in range(len(filecont[i*frq_len+frq_start].split())):
                    irdata.append(IR(filecont,i*frq_len+frq_start,j,frq_len))

            irout = []
            for i in range(len(irdata)):
                if irdata[i].freq < freqmax and irdata[i].freq > freqmin and irdata[i].int > intmin and irdata[i].int < intmax and irdata[i].deltas[int(atom1)] >= threshold and irdata[i].deltas[int(atom2)] >= threshold:
                        irout = [irdata[i].freq, irdata[i].int]

            #this adds the frequency data from the irout into the new property df
            row_i = {'IR_freq_'+str(pair_label): irout[0]}
            IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire IR frequencies for:', row['log_name'], ".log")
            row_i = {'IR_freq_'+str(pair_label): "no data"}
            IR_dataframe = IR_dataframe.append(row_i, ignore_index=True)
    print("IR function has completed for", a1, "and", a2)
    return(pd.concat([dataframe, IR_dataframe], axis = 1))

def get_buried_sterimol(dataframe, sterimol_list, r_buried): #uses morfeus to calculate sterimol L, B1, B5 for two input atoms for every entry in df
    sterimol_dataframe = pd.DataFrame(columns=[])
    r_buried -= 0.5 #the function adds

    for index, row in dataframe.iterrows():
        try:
            #parsing the Sterimol axis defined in the list from input line
            sterimolnums_list = []
            for sterimol in sterimol_list:
                atomnum_list = [] #the atom numbers use to collect sterimol values (i.e. [18 16 17 15]) are collected from the df using the input list (i.e. [["O2", "C1"], ["O3", "H5"]])
                for atom in sterimol:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                sterimolnums_list.append(atomnum_list) #append atomnum_list for each sterimol axis defined in the input to make a list of the form [['18', '16'], ['16', '15']]

            #this makes column headers based on Sterimol axis defined in the input line
            sterimoltitle_list = []
            for sterimol in sterimol_list:
                sterimoltitle = str(sterimol[0]) + "_" + str(sterimol[1])
                sterimoltitle_list.append(sterimoltitle)

            log_file = row['log_name']
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Buried_Sterimol_L_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data",
                    'Buried_Sterimol_B1_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data",
                    'Buried_Sterimol_B5_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
                continue

            geom = get_geom(streams)

            #checks for if the wrong number of atoms are input, input is not of the correct form, or calls atom numbers that do not exist in the molecule
            error = ""
            for sterimol in sterimolnums_list:
                if len(sterimol)%2 != 0:
                    error = "Number of atom inputs given for Sterimol is not divisible by two. " + str(len(atoms)) + " atoms were given. "
                for atom in sterimol:
                    if not atom.isdigit():
                        error += " " + atom + ": Only numbers accepted as input for Sterimol"
                    if int(atom) > len(geom):
                        error += " " + atom + " is out of range. Maximum valid atom number: " + str(len(geom)+1) + " "
                if error != "": print(error)

            elements = np.array([geom[i][0] for i in range(len(geom))])
            coordinates = np.array([np.array(geom[i][1:]) for i in range(len(geom))])

            #this collects Sterimol values for each pair of inputs
            sterimolout = []
            for sterimol in sterimolnums_list:
                sterimol_values = Sterimol(elements, coordinates, int(sterimol[0]), int(sterimol[1])) #calls morfeus
                sterimol_values.bury(method="delete", sphere_radius=float(r_buried))
                sterimolout.append(sterimol_values)

            #this adds the data from sterimolout into the new property df
            row_i = {}
            for a in range(0, len(sterimolnums_list)):
                entry = {'Buried_Sterimol_L_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': sterimolout[a].L_value,
                'Buried_Sterimol_B1_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': sterimolout[a].B_1_value,
                'Buried_Sterimol_B5_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': sterimolout[a].B_5_value}
                row_i.update(entry)
            sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire Morfeus Buried Sterimol parameters for:', row['log_name'], ".log")
            row_i = {}
            try:
                for a in range(0, len(sterimolnums_list)):
                    entry = {'Buried_Sterimol_L_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data",
                    'Buried_Sterimol_B1_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data",
                    'Buried_Sterimol_B5_' + str(sterimoltitle_list[a]) + '_' + str(r_buried) + '(Å)': "no data"}
                    row_i.update(entry)
                sterimol_dataframe = sterimol_dataframe.append(row_i, ignore_index=True)
            except:
                print("****Ope, there's a problem with your atom inputs.")
    print("Morfeus Buried Sterimol function has completed for", sterimol_list)
    return(pd.concat([dataframe, sterimol_dataframe], axis = 1))

def get_chelpg(dataframe, a_list): #a function to get the ChelpG ESP charges for all atoms (a_list, form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    chelpg_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:#try to get the data
            atomnum_list = []
            for atom in a_list:
                atomnum = row[str(atom)] #the atom number (i.e. 16) to add to the list is the df entry of this row for the labeled atom (i.e. "C1")
                atomnum_list.append(str(atomnum)) #append that to atomnum_list to make a list of the form [16, 17, 29]
            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'ChelpG_charge_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                chelpg_dataframe = chelpg_dataframe.append(row_i, ignore_index=True)
                continue

            chelpgstart,chelpg,error,chelpgout = 0,False,"",[]

            #this section finds the line (chelpgstart) where the ChelpG data is located
            for i in range(len(filecont)-1,0,-1):
                if chelpg2_pattern.search(filecont[i]):
                    chelpgstart = i
                if chelpg1_pattern.search(filecont[i]):
                    chelpg = True
                    break
            if chelpgstart != 0 and chelpg == False:
                error = "****Other ESP scheme than ChelpG used in: " + str(log_file) + ".log"
            if chelpgstart == 0:
                error = "****no ChelpG ESP charge analysis found in: "+ str(log_file) + ".log"
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'ChelpG_charge_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                chelpg_dataframe = chelpg_dataframe.append(row_i, ignore_index=True)
                continue

            for atom in atomnum_list:
                if atom.isnumeric():
                    chelpgout.append(filecont[chelpgstart+int(atom)+2].split()[-1])

            #this adds the data from the chelpgout into the new property df
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'ChelpG_charge_'+str(a_list[a]): chelpgout[a]}
                row_i.update(entry)
            chelpg_dataframe = chelpg_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire ChelpG charges for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'ChelpG_charge_'+str(a_list[a]): "no data"}
                row_i.update(entry)
            chelpg_dataframe = chelpg_dataframe.append(row_i, ignore_index=True)
    print("ChelpG function has completed for", a_list)
    return(pd.concat([dataframe, chelpg_dataframe], axis = 1))

def get_hirshfeld(dataframe,a_list): #a function to get the Hirshfeld charge, CM5 charge, and atomic dipole for all atoms (a_list, form ["C1", "C4", "O2"]) in a dataframe that contains file name and atom number
    hirsh_dataframe = pd.DataFrame(columns=[]) #define an empty df to place results in

    for index, row in dataframe.iterrows(): #iterate over the dataframe
        try:#try to get the data
            atomnum_list = []
            for atom in a_list:
                atomnum = row[str(atom)] #the atom number (i.e. 16) to add to the list is the df entry of this row for the labeled atom (i.e. "C1")
                atomnum_list.append(str(atomnum)) #append that to atomnum_list to make a list of the form [16, 17, 29]

            log_file = row['log_name'] #read file name from df
            filecont, error = get_filecont(log_file) #read the contents of the log file
            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'Hirsh_charge_'+str(a_list[a]): "no data",
                            'Hirsh_CM5_charge_'+str(a_list[a]): "no data",
                            'Hirsh_atom_dipole_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                hirsh_dataframe = hirsh_dataframe.append(row_i, ignore_index=True)
                continue

            hirshstart,error,hirshout = 0,False,[]

            #this section finds the line (chelpgstart) where the ChelpG data is located
            for i in range(len(filecont)-1,0,-1):
                if hirshfeld_pattern.search(filecont[i]):
                    hirshstart = i
                    break
            if hirshstart == 0:
                error = "****no Hirshfeld Population Analysis found in: " + str(log_file) + ".log"
                print(error)
                row_i = {}
                for a in range(0, len(a_list)):
                    entry = {'Hirsh_charge_'+str(a_list[a]): "no data",
                        'Hirsh_CM5_charge_'+str(a_list[a]): "no data",
                        'Hirsh_atom_dipole_'+str(a_list[a]): "no data"}
                    row_i.update(entry)
                hirsh_dataframe = hirsh_dataframe.append(row_i, ignore_index=True)
                continue

            for atom in atomnum_list:
                if atom.isnumeric():
                    cont = filecont[hirshstart+int(atom)+1].split()
                    qh = cont[2] #using 0-indexing, this gets the value for Hirshfeld charge from the 2nd column
                    qcm5 = cont[7] #using 0-indexing, this gets the value for CM5 charge from the 7th column
                    d = np.linalg.norm(np.array((cont[4:8])))
                    hirshout.append([str(qh),str(qcm5),str(d)])

            #this adds the data from the hirshout into the new property df
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'Hirsh_charge_'+str(a_list[a]): hirshout[a][0],
                        'Hirsh_CM5_charge_'+str(a_list[a]): hirshout[a][1],
                        'Hirsh_atom_dipole_'+str(a_list[a]): hirshout[a][2]}
                row_i.update(entry)
            hirsh_dataframe = hirsh_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire Hirshfeld properties for:', row['log_name'], ".log")
            row_i = {}
            for a in range(0, len(a_list)):
                entry = {'Hirsh_charge_'+str(a_list[a]): "no data",
                        'Hirsh_CM5_charge_'+str(a_list[a]): "no data",
                        'Hirsh_atom_dipole_'+str(a_list[a]): "no data"}
                row_i.update(entry)
            hirsh_dataframe = hirsh_dataframe.append(row_i, ignore_index=True)
    print("Hirshfeld function has completed for", a_list)
    return(pd.concat([dataframe, hirsh_dataframe], axis = 1))

def get_cone_angle(dataframe, a_list): #DOES NOT MATCH VALUES FROM LITERATURE, WORK IN PROGRESS
    cone_angle_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        if True:
        #try:
            atom_list = []
            for label in a_list:
                atom = row[str(label)] #the atom number (i.e. 16) to add to the list is the df entry of this row for the labeled atom (i.e. "C1")
                atom_list.append(str(atom)) #append that to atom_list to make a list of the form [16, 17, 29]

            log_file = row['log_name']
            streams, errors = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            cone_angle_out = []
            for atom in atom_list:
                cone_angle = ConeAngle(elements, coordinates, int(atom)) #calls morfeus
                cone_angle_out.append(cone_angle)
            cone_angle.print_report()

            row_i = {}
            for a in range(0, len(atom_list)):
                entry = {'cone_angle' + str(a_list[a]) + '(°)': cone_angle_out[a].cone_angle} #details on these values can be found here: https://kjelljorner.github.io/morfeus/pyramidalization.html
                row_i.update(entry)
            cone_angle_dataframe = cone_angle_dataframe.append(row_i, ignore_index=True)
        #except:
        #    print('Unable to acquire cone_angle parameters for:', row['log_name'], ".log")
    print("cone_angle function has completed for", a_list)
    return(pd.concat([dataframe, cone_angle_dataframe], axis = 1))

#####all new functions are added below......######

def get_SCF_energy(dataframe):
    SCF_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            filecont, error = get_filecont(log_file)
            if error != "":
                print(error)
                row_i = {'SCF_energy(Hartree)': "no data"}
                SCF_dataframe = SCF_dataframe.append(row_i, ignore_index=True)
                continue

            SCF_energies = []
            SCF_counter = 0
            with open(log_file + '.log') as f:
                line_list = list(f)
                line_list.reverse()

                for SCF_line in line_list:
                    if SCF_line.find("SCF Done") != -1:
                        energy = float(SCF_line.split(' ')[7])
                        SCF_energies.append(energy)
                        break

            row_i = {'SCF_energy(Hartree)': SCF_energies[0]}
            SCF_dataframe = SCF_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to aquire SCF energy for: ', row['log_name'], ".log")
            row_i = {'SCF_energy(Hartree)': "no data"}
            SCF_dataframe = SCF_dataframe.append(row_i, ignore_index=True)
    print("SCF energy collection has completed")
    return(pd.concat([dataframe, SCF_dataframe], axis=1))

def get_bite_angle(dataframe, a1, d1, d2):
    atom = str(a1)
    donor1 = str(d1)
    donor2 = str(d2)
    bite_angle_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            metal_atom = row[atom]
            donor1_atom = row[donor1]
            donor2_atom = row[donor2]
            streams, error = get_outstreams(log_file)

            if error != "":
                print(error)
                row_i = {'Bite_angle_' + str(atom) + '(°)': "no data"}
                bite_angle_dataframe = bite_angle_dataframe.append(row_i, ignore_index=True)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])
            bite_angle = BiteAngle(coordinates, int(metal_atom), int(donor1_atom), int(donor2_atom))
            row_i = {'Bite_angle_' + str(atom) + '(°)': bite_angle.angle}
            bite_angle_dataframe = bite_angle_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to aquire bite angle for: ', row['log_name'], ".log")
            row_i = {'Bite_angle_' + str(atom) + '(°)': "no data"}
            bite_angle_dataframe = bite_angle_dataframe.append(row_i, ignore_index=True)
    print("Bite angle collection has completed")
    return(pd.concat([dataframe, bite_angle_dataframe], axis=1))

def get_solid_angle(dataframe, a1, ex):
    atom = str(a1)
    exclude = str(ex)

    solid_angle_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        log_file = row['log_name']
        atom1 = row[str(a1)]
        streams, error = get_outstreams(log_file)

        if error != "":
            print(error)
            row_i = {'Solid_angle_'+str(atom)+'(sr)': "no data", 'Solid_cone_angle_'+str(atom)+'(°)': "no data", '%G_param_'+str(atom): "no data"}
            solid_angle_dataframe = solid_angle_dataframe.append(row_i, ignore_index=True)
            continue

        log_coordinates = get_geom(streams)
        elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
        coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

        mask = elements != exclude
        elements = elements[mask]
        coordinates = coordinates[mask]
        metal_idx = np.where(elements == atom)[0] + 1

        solid_angle = SolidAngle(elements, coordinates, metal_index=int(metal_idx))
        #solid_angle.draw_3D(size=10)
        row_i = {'Solid_angle_'+str(atom)+'(sr)': solid_angle.solid_angle, 'Solid_cone_angle_'+str(atom)+'(°)': solid_angle.cone_angle, '%G_param_'+str(atom): solid_angle.G}
        solid_angle_dataframe = solid_angle_dataframe.append(row_i, ignore_index=True)

    print("Solid angle collection has completed")
    return(pd.concat([dataframe, solid_angle_dataframe], axis=1))


def get_visible_volume(dataframe, a1, ex):
    atom = str(a1)
    excluded = str(ex)

    visible_volume_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            atom1 = row[str(a1)]
            streams, error = get_outstreams(log_file)

            if error != "":
                print(error)
                row_i = {'Visible_volume_'+str(atom)+'(Å³)': "no data"}
                visible_volume_dataframe = visible_volume_dataframe.append(row_i, ignore_index=True)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            mask = elements != excluded
            elements = elements[mask]
            coordinates = coordinates[mask]
            metal_idx = np.where(elements == atom)[0] + 1

            visible_volume = VisibleVolume(elements, coordinates, int(metal_idx), include_hs=True)
            row_i = {'Visible_volume_'+str(atom)+'(Å³)': visible_volume.visible_volume}
            visible_volume_dataframe = visible_volume_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to aquire visible volume for ', row['log_name'], ".log")
            row_i = {'Visible_volume_'+str(atom)+'(Å³)': "no data"}
            visible_volume_dataframe = visible_volume_dataframe.append(row_i, ignore_index=True)
    print("Visible volume collection has completed")
    return(pd.concat([dataframe, visible_volume_dataframe], axis=1))

def get_bond_occ_en(dataframe, bond_list):  #phosphorus must be first in bond list?
    bond_occ_en_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            bondnums_list = []
            for bond in bond_list:
                atomnum_list = []
                for atom in bond:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                bondnums_list.append(atomnum_list)

            bondtitle_list = []
            for bond in bond_list:
                bondtitle = str(bond[0]) + "_" + str(bond[1])
                bondtitle_list.append(bondtitle)

            log_file = row['log_name']
            filecont, error = get_filecont(log_file)

            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(bondnums_list)):
                    entry = {'NBO_Bond_occup_' + str(bondtitle_list[a]): "no data", 'NBO_Bond_energy_' + str(bondtitle_list[a]): "no data"}
                    row_i.update(entry)
                bond_occ_en_dataframe = bond_occ_en_dataframe.append(row_i, ignore_index=True)
                continue

            nbo_bond_occ_out = []
            nbo_bond_energy_out = []
            for bond in bondnums_list:
                pattern = "BD \(\s+1\) P\s+" + str(bond[0])+"\s+-Pd\s+" + str(bond[1])+"|"+"BD \(\s+1\)Pd\s+" + str(bond[1]) + " - P\s+" + str(bond[0])
                for line in filecont[::-1]:
                    if re.search(pattern, line):
                        bond_occup = str.split(line)[8]
                        nbo_bond_occ_out.append(bond_occup)
                        bond_energy = str.split(line)[9]
                        nbo_bond_energy_out.append(bond_energy)
                        break

            row_i = {}
            for a in range(0, len(bond_list)):
                entry = {'NBO_Bond_occup_' + str(bondtitle_list[a]): nbo_bond_occ_out[a], 'NBO_Bond_energy_' + str(bondtitle_list[a]): nbo_bond_energy_out[a]}
                row_i.update(entry)
            bond_occ_en_dataframe = bond_occ_en_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to aquire NBO bond energies and occupancies for ', row['log_name'], ".log")
            row_i  = {'NBO_Bond_occup_' + str(bondtitle_list[a]): "no data", 'NBO_Bond_energy_' + str(bondtitle_list[a]): "no data"}
            bond_occ_en_dataframe = bond_occ_en_dataframe.append(row_i, ignore_index=True)
    print("NBO bond energy and occupancy collection has completed")
    return(pd.concat([dataframe, bond_occ_en_dataframe], axis=1))

def get_antibond_occ_en(dataframe, bond_list):  #phosphorus must be first in bond list?
    antibond_occ_en_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            bondnums_list = []
            for bond in bond_list:
                atomnum_list = []
                for atom in bond:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                bondnums_list.append(atomnum_list)

            bondtitle_list = []
            for bond in bond_list:
                bondtitle = str(bond[0]) + "_" + str(bond[1])
                bondtitle_list.append(bondtitle)

            log_file = row['log_name']
            filecont, error = get_filecont(log_file)

            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(bondnums_list)):
                    entry = {'NBO_Antibond_occup_' + str(bondtitle_list[a]): "no data", 'NBO_Antibond_energy_' + str(bondtitle_list[a]): "no data"}
                    row_i.update(entry)
                antibond_occ_en_dataframe = antibond_occ_en_dataframe.append(row_i, ignore_index=True)
                continue

            nbo_antibond_occ_out = []
            nbo_antibond_energy_out = []
            for bond in bondnums_list:
                pattern = "BD\*\(\s+1\) [PN]\s+" + str(bond[0]) + " -Pd\s+" + str(bond[1]) + "|"+"BD\*\(\s+1\)Pd\s+" + str(bond[1]) + " - [PN]\s+" + str(bond[0])
                for line in filecont[::-1]:
                    if re.search(pattern, line):
                        antibond_occup = str.split(line)[7]
                        nbo_antibond_occ_out.append(antibond_occup)
                        antibond_energy = str.split(line)[8]
                        nbo_antibond_energy_out.append(antibond_energy)
                        break

            row_i = {}
            for a in range(0, len(bond_list)):
                entry = {'NBO_Antibond_occup_' + str(bondtitle_list[a]): nbo_antibond_occ_out[a], 'NBO_Antibond_energy_' + str(bondtitle_list[a]): nbo_antibond_energy_out[a]}
                row_i.update(entry)
            antibond_occ_en_dataframe = antibond_occ_en_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to aquire NBO antibonding energies and occupancies for ', row['log_name'], ".log")
            row_i  = {'NBO_Antibond_occup_' + str(bondtitle_list[a]): "no data", 'NBO_Antibond_energy_' + str(bondtitle_list[a]): "no data"}
            antibond_occ_en_dataframe = antibond_occ_en_dataframe.append(row_i, ignore_index=True)
    print("NBO antibonding energy and occupancy collection has completed")
    return(pd.concat([dataframe, antibond_occ_en_dataframe], axis=1))


def get_antibond_occ_en_new(dataframe, bond_list):  #phosphorus must be first in bond list?
    antibond_occ_en_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            bondnums_list = []
            for bond in bond_list:
                atomnum_list = []
                for atom in bond:
                    atomnum = row[str(atom)]
                    atomnum_list.append(str(atomnum))
                bondnums_list.append(atomnum_list)

            print(bondnums_list)

            bondtitle_list = []
            for bond in bond_list:
                bondtitle = str(bond[0]) + "_" + str(bond[1])
                bondtitle_list.append(bondtitle)
                atom1_split = [*str(bond[0])]   ##need to state whether this is Pd/Zn or not before splitting the chars ?
                atom2_split = [*str(bond[1])]   ##can this be done by reading in the number of chars ?
                print(atom1_split)
                print(atom2_split)

            #print(bondtitle_list)

            log_file = row['log_name']
            filecont, error = get_filecont(log_file)

            if error != "":
                print(error)
                row_i = {}
                for a in range(0, len(bondnums_list)):
                    entry = {'NBO_Antibond_occup_' + str(bondtitle_list[a]): "no data", 'NBO_Antibond_energy_' + str(bondtitle_list[a]): "no data"}
                    row_i.update(entry)
                antibond_occ_en_dataframe = antibond_occ_en_dataframe.append(row_i, ignore_index=True)
                continue

            nbo_antibond_occ_out = []
            nbo_antibond_energy_out = []
            for bond in bondnums_list:
                pattern = "BD\*\(\s+1\) P\s+" + str(bond[0]) + " -Pd\s+" + str(bond[1]) + "|"+"BD\*\(\s+1\)Pd\s+" + str(bond[1]) + " - P\s+" + str(bond[0])
                for line in filecont[::-1]:
                    if re.search(pattern, line):
                        antibond_occup = str.split(line)[7]
                        nbo_antibond_occ_out.append(antibond_occup)
                        antibond_energy = str.split(line)[8]
                        nbo_antibond_energy_out.append(antibond_energy)
                        break

            row_i = {}
            for a in range(0, len(bond_list)):
                entry = {'NBO_Antibond_occup_' + str(bondtitle_list[a]): nbo_antibond_occ_out[a], 'NBO_Antibond_energy_' + str(bondtitle_list[a]): nbo_antibond_energy_out[a]}
                row_i.update(entry)
            antibond_occ_en_dataframe = antibond_occ_en_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to aquire NBO antibonding energies and occupancies for ', row['log_name'], ".log")
            row_i  = {'NBO_Antibond_occup_' + str(bondtitle_list[a]): "no data", 'NBO_Antibond_energy_' + str(bondtitle_list[a]): "no data"}
            antibond_occ_en_dataframe = antibond_occ_en_dataframe.append(row_i, ignore_index=True)
    print("NBO antibonding energy and occupancy collection has completed")
    return(pd.concat([dataframe, antibond_occ_en_dataframe], axis=1))


def get_SASA_metal(dataframe, a1, ex): #uses morfeus to calculate solvent accessible surface area in a dataframe that contains file name
    #if you want to SASA with different probe radii, morfeus has this functionality, but it has not been implemented here
    atom = str(a1)
    excluded = str(ex)
    sasa_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            atom1 = row[str(a1)]
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)
                row_i = {"SASA_"+str(atom)+"_surface_area(Å²)": "no data"}
                sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])

            mask = elements != excluded
            elements = elements[mask]
            coordinates = coordinates[mask]
            metal_idx = np.where(elements == atom)[0] + 1

            sasa = SASA(elements,coordinates) #calls morfeus

            row_i = {"SASA_"+str(atom)+"_surface_area(Å²)": sasa.atom_areas[int(metal_idx)]}
            sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True)
        except:
            print('****Unable to acquire metal SASA parameters for:', row['log_name'], ".log")
            row_i = {"SASA_"+str(atom)+"_surface_area(Å²)": "no data"}
            sasa_dataframe = sasa_dataframe.append(row_i, ignore_index=True)
    print("SASA metal function has completed")
    return(pd.concat([dataframe, sasa_dataframe], axis = 1))


def get_vbur_quadrants_octants(dataframe, a1, ex1, ex2, z1, z2, radius): #uses morfeus to calculate vbur at a single radius for an atom (a1) in df
    atom = str(a1)
    atom_ex1 = str(ex1)
    atom_ex2 = str(ex2)
    atom_z1 = str(z1)
    atom_z2 = str(z2)

    vbur_quadoct_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            atom1 = row[atom] #gets numerical value (e.g. 16) for a1 (e.g. C1)
            exclude1 = row[atom_ex1]
            exclude2 = row[atom_ex2]
            zaxis1 = row[atom_z1]
            zaxis2 = row[atom_z2]
            xzatom = row[atom_z2]
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)

                row_i = {'%Vbur_' + str(atom) + '_quadrant_+,+': "no data",
                     '%Vbur_' + str(atom) + '_quadrant_–,+': "no data",
                     '%Vbur_' + str(atom) + '_quadrant_–,–': "no data",
                     '%Vbur_' + str(atom) + '_quadrant_+,–': "no data",
                     '%Vbur_' + str(atom) + '_octant_+,+,+': "no data",
                     '%Vbur_' + str(atom) + '_octant_–,+,+': "no data",
                     '%Vbur_' + str(atom) + '_octant_–,–,+': "no data",
                     '%Vbur_' + str(atom) + '_octant_+,–,+': "no data",
                     '%Vbur_' + str(atom) + '_octant_+,–,–': "no data",
                     '%Vbur_' + str(atom) + '_octant_–,–,–': "no data",
                     '%Vbur_' + str(atom) + '_octant_–,+,–': "no data",
                     '%Vbur_' + str(atom) + '_octant_+,+,–': "no data",
                    }

                vbur_dataframe = vbur_dataframe.append(row_i, ignore_index=True)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])
            vbur = BuriedVolume(elements, coordinates, int(atom1), radius=radius, include_hs=True, excluded_atoms=[exclude1, exclude2], z_axis_atoms=[int(zaxis1), int(zaxis2)], xz_plane_atoms=[int(xzatom)]) #calls morfeus

            vbur.octant_analysis()

            quadrants = vbur.quadrants
            bv_quadrants = quadrants['percent_buried_volume']

            octants = vbur.octants
            bv_octants = octants['percent_buried_volume']

            row_i = {'%Vbur_' + str(atom) + '_quadrant_+,+': bv_quadrants[1],
                     '%Vbur_' + str(atom) + '_quadrant_–,+': bv_quadrants[2],
                     '%Vbur_' + str(atom) + '_quadrant_–,–': bv_quadrants[3],
                     '%Vbur_' + str(atom) + '_quadrant_+,–': bv_quadrants[4],
                     '%Vbur_' + str(atom) + '_octant_+,+,+': bv_octants[0],
                     '%Vbur_' + str(atom) + '_octant_–,+,+': bv_octants[1],
                     '%Vbur_' + str(atom) + '_octant_–,–,+': bv_octants[2],
                     '%Vbur_' + str(atom) + '_octant_+,–,+': bv_octants[3],
                     '%Vbur_' + str(atom) + '_octant_+,–,–': bv_octants[4],
                     '%Vbur_' + str(atom) + '_octant_–,–,–': bv_octants[5],
                     '%Vbur_' + str(atom) + '_octant_–,+,–': bv_octants[6],
                     '%Vbur_' + str(atom) + '_octant_+,+,–': bv_octants[7],
                    }

            vbur_quadoct_dataframe = vbur_quadoct_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to acquire Vbur quadrant and octants for:', row['log_name'], ".log")

            row_i = {'%Vbur_' + str(atom) + '_quadrant_+,+': "no data",
                 '%Vbur_' + str(atom) + '_quadrant_–,+': "no data",
                 '%Vbur_' + str(atom) + '_quadrant_–,–': "no data",
                 '%Vbur_' + str(atom) + '_quadrant_+,–': "no data",
                 '%Vbur_' + str(atom) + '_octant_+,+,+': "no data",
                 '%Vbur_' + str(atom) + '_octant_–,+,+': "no data",
                 '%Vbur_' + str(atom) + '_octant_–,–,+': "no data",
                 '%Vbur_' + str(atom) + '_octant_+,–,+': "no data",
                 '%Vbur_' + str(atom) + '_octant_+,–,–': "no data",
                 '%Vbur_' + str(atom) + '_octant_–,–,–': "no data",
                 '%Vbur_' + str(atom) + '_octant_–,+,–': "no data",
                 '%Vbur_' + str(atom) + '_octant_+,+,–': "no data",
                }

            vbur_quadoct_dataframe = vbur_quadoct_dataframe.append(row_i, ignore_index=True)
    print("Buried volume quadrants and octants function has completed")
    return(pd.concat([dataframe, vbur_quadoct_dataframe], axis = 1))

def get_vbur_quadrants_only(dataframe, a1, ex1, ex2, z1, z2, radius): #uses morfeus to calculate vbur at a single radius for an atom (a1) in df
    atom = str(a1)
    atom_ex1 = str(ex1)
    atom_ex2 = str(ex2)
    atom_z1 = str(z1)
    atom_z2 = str(z2)

    vbur_quadoct_dataframe = pd.DataFrame(columns=[])

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            atom1 = row[atom] #gets numerical value (e.g. 16) for a1 (e.g. C1)
            exclude1 = row[atom_ex1]
            exclude2 = row[atom_ex2]
            zaxis1 = row[atom_z1]
            zaxis2 = row[atom_z2]
            xzatom = row[atom_z2]
            streams, error = get_outstreams(log_file) #need to add file path if you're running from a different directory than file
            if error != "":
                print(error)

                row_i = {'%Vbur_' + str(atom) + '_quadrant_+,+': "no data",
                     '%Vbur_' + str(atom) + '_quadrant_–,+': "no data",
                     '%Vbur_' + str(atom) + '_quadrant_–,–': "no data",
                     '%Vbur_' + str(atom) + '_quadrant_+,–': "no data"
                    }

                vbur_dataframe = vbur_dataframe.append(row_i, ignore_index=True)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([log_coordinates[i][0] for i in range(len(log_coordinates))])
            coordinates = np.array([np.array(log_coordinates[i][1:]) for i in range(len(log_coordinates))])
            vbur = BuriedVolume(elements, coordinates, int(atom1), radius=radius, include_hs=True, excluded_atoms=[exclude1, exclude2], z_axis_atoms=[int(zaxis1), int(zaxis2)], xz_plane_atoms=[int(xzatom)]) #calls morfeus

            vbur.octant_analysis()

            quadrants = vbur.quadrants
            bv_quadrants = quadrants['percent_buried_volume']

            row_i = {'%Vbur_' + str(atom) + '_quadrant_+,+': bv_quadrants[1],
                     '%Vbur_' + str(atom) + '_quadrant_–,+': bv_quadrants[2],
                     '%Vbur_' + str(atom) + '_quadrant_–,–': bv_quadrants[3],
                     '%Vbur_' + str(atom) + '_quadrant_+,–': bv_quadrants[4]
                    }

            vbur_quadoct_dataframe = vbur_quadoct_dataframe.append(row_i, ignore_index=True)

        except:
            print('****Unable to acquire Vbur quadrant and octants for:', row['log_name'], ".log")

            row_i = {'%Vbur_' + str(atom) + '_quadrant_+,+': "no data",
                 '%Vbur_' + str(atom) + '_quadrant_–,+': "no data",
                 '%Vbur_' + str(atom) + '_quadrant_–,–': "no data",
                 '%Vbur_' + str(atom) + '_quadrant_+,–': "no data",
                }

            vbur_quadoct_dataframe = vbur_quadoct_dataframe.append(row_i, ignore_index=True)
    print("Buried volume quadrants and octants function has completed")
    return(pd.concat([dataframe, vbur_quadoct_dataframe], axis = 1))

import os
import openbabel
from openbabel import pybel

def get_sdf_from_log(directory=os.getcwd()):
    for filename in os.listdir(directory):
        if filename.endswith(".log"):
            log_path = os.path.join(directory, filename)
            base_name = os.path.splitext(filename)[0]
            sdf_path = os.path.join(directory, f"{base_name}.sdf")
            for mol in pybel.readfile("log", log_path):
                mol.write("sdf", sdf_path, overwrite=True)


### Step 2: Get list of .log files and write to log_ids.txt
def get_log_ids(directory=os.getcwd()):
    with open("log_ids.txt", "w") as a:
        for filename in os.listdir(os.getcwd()):
            if filename.endswith('.log'):
                f = os.path.join(filename)
                a.write(str(f) + os.linesep)

from rdkit import Chem
### Step 3: Determine ligand core type (5- or 6-membered ring)
def get_ligand_core_type(sdf_directory=os.getcwd()):
    mols_5 = {}
    mols_6 = {}
    results = []
    with open('log_ids.txt', 'r') as file:
        log_names = [line.strip() for line in file]

    for log in log_names:
        base_name = log.split('.')[0]
        sdf_name = base_name + '.sdf'
        sdf_path = os.path.join(sdf_directory, sdf_name)
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        original_mol = next((m for m in suppl if m is not None), None)
        if original_mol is None:
            print(f"sdf error for {sdf_name}, cannot automatically get atom numbering")
            continue

        Ni_atom = [atom for atom in original_mol.GetAtoms() if atom.GetAtomicNum() == 28]
        Ni_atom_idx = Ni_atom[0].GetIdx()
        Ni_ringsize_5 = original_mol.GetAtomWithIdx(Ni_atom_idx).IsInRingSize(5)
        Ni_ringsize_6 = original_mol.GetAtomWithIdx(Ni_atom_idx).IsInRingSize(6)

        if Ni_ringsize_5 and not Ni_ringsize_6:
            mols_5[base_name] = original_mol
            results.append({'id': base_name, 'ligand_type': 'core_5'})
        elif not Ni_ringsize_5 and Ni_ringsize_6:
            mols_6[base_name] = original_mol
            results.append({'id': base_name, 'ligand_type': 'core_6'})

    return mols_5, mols_6, pd.DataFrame(results)

### Step 4: Get initial atom numbering based on substructure matching
def get_initial_atom_numbers(mol_dict5, mol_dict6, sdf_directory=os.getcwd()):
    substructure_5 = ['[H][Ni]1([H])[N]C=C[N]1', '[H][Ni]1([H])[N]CC[N]1', '[H][Ni]1([H])[#7]~[#6]~[#6]~[#7]1' ]

    results5 = []
    results6 = []
    for key, value in mol_dict5.items():
        ligID = key
        mol = value

        for substructure in substructure_5:
            substructure_match = mol.GetSubstructMatches(Chem.MolFromSmarts(substructure))
            if len(substructure_match) > 0:
                atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist])
                atoms_idx_in_substructure_gauss = [x+1 for x in atoms_idx_in_substructure_rdkit] #this line changes from 0-indexed to 1-indexed (for Gaussian)
                results5.append({'log_name': ligID, 'ligand_type': 'core_5', 'H1': atoms_idx_in_substructure_gauss[0], 'Ni': atoms_idx_in_substructure_gauss[1], 'H2': atoms_idx_in_substructure_gauss[2], 'N1': atoms_idx_in_substructure_gauss[3], 'C1': atoms_idx_in_substructure_gauss[4], 'C2': atoms_idx_in_substructure_gauss[5], 'N2': atoms_idx_in_substructure_gauss[6]})
                break
        if ligID not in [entry['log_name'] for entry in results5]:
            sdf_name = key + '.sdf'
            print (sdf_name)
            sdf_path = os.path.join(sdf_directory, sdf_name)
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None) #rdkit is weird sometimes just generating the mol again fixes the issue
            for substructure in substructure_5:
                print (substructure)
                substructure_match = mol.GetSubstructMatches(Chem.MolFromSmarts(substructure))
                print (substructure_match)
                if len(substructure_match) > 0:
                    atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist])
                    atoms_idx_in_substructure_gauss = [x+1 for x in atoms_idx_in_substructure_rdkit] #this line changes from 0-indexed to 1-indexed (for Gaussian)
                    results5.append({'log_name': ligID, 'ligand_type': 'core_5', 'H1': atoms_idx_in_substructure_gauss[0], 'Ni': atoms_idx_in_substructure_gauss[1], 'H2': atoms_idx_in_substructure_gauss[2], 'N1': atoms_idx_in_substructure_gauss[3], 'C1': atoms_idx_in_substructure_gauss[4],'C3': atoms_idx_in_substructure_gauss[5], 'C2': atoms_idx_in_substructure_gauss[6], 'N2': atoms_idx_in_substructure_gauss[7]})
                    break

    substructure_6 = ['[H][Ni]1([H])[N]CCC[N]1', '[H][Ni]1([H])[N]CC=C[N]1', '[H][Ni]1([H])[#7]~[#6]~[#6]~[#6]~[#7]1']
    for key, value in mol_dict6.items():
        ligID = key
        mol = value

        for substructure in substructure_6:
            substructure_match = mol.GetSubstructMatches(Chem.MolFromSmarts(substructure))
            if len(substructure_match) > 0:
                atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist])
                atoms_idx_in_substructure_gauss = [x+1 for x in atoms_idx_in_substructure_rdkit] #this line changes from 0-indexed to 1-indexed (for Gaussian)
                results6.append({'log_name': ligID, 'ligand_type': 'core_6', 'H1': atoms_idx_in_substructure_gauss[0], 'Ni': atoms_idx_in_substructure_gauss[1], 'H2': atoms_idx_in_substructure_gauss[2], 'N1': atoms_idx_in_substructure_gauss[3], 'C1': atoms_idx_in_substructure_gauss[4],'C3': atoms_idx_in_substructure_gauss[5], 'C2': atoms_idx_in_substructure_gauss[6], 'N2': atoms_idx_in_substructure_gauss[7]})
                break
        if ligID not in [entry['log_name'] for entry in results6]:
            sdf_name = key + '.sdf'
            print (sdf_name)
            sdf_path = os.path.join(sdf_directory, sdf_name)
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None) #rdkit is weird sometimes just generating the mol again fixes the issue
            for substructure in substructure_6:
                print (substructure)
                substructure_match = mol.GetSubstructMatches(Chem.MolFromSmarts(substructure))
                print (substructure_match)
                if len(substructure_match) > 0:
                    atoms_idx_in_substructure_rdkit = list([item for sublist in substructure_match for item in sublist])
                    atoms_idx_in_substructure_gauss = [x+1 for x in atoms_idx_in_substructure_rdkit] #this line changes from 0-indexed to 1-indexed (for Gaussian)
                    results6.append({'log_name': ligID, 'ligand_type': 'core_6', 'H1': atoms_idx_in_substructure_gauss[0], 'Ni': atoms_idx_in_substructure_gauss[1], 'H2': atoms_idx_in_substructure_gauss[2], 'N1': atoms_idx_in_substructure_gauss[3], 'C1': atoms_idx_in_substructure_gauss[4],'C3': atoms_idx_in_substructure_gauss[5], 'C2': atoms_idx_in_substructure_gauss[6], 'N2': atoms_idx_in_substructure_gauss[7]})
                    break
    if len(results6) > 0 and len(results5) > 0:
        results6 = (pd.DataFrame(results6)).drop(columns=['C3'])
        results5 = pd.DataFrame(results5)
        results = pd.DataFrame(pd.concat([results5, results6], ignore_index=True))
    elif len(results6) > 0 and len(results5) == 0:
        results = pd.DataFrame(results6).drop(columns=['C3'])
    elif len(results6) == 0 and len(results5) > 0:
        results = pd.DataFrame(results5)

    return pd.DataFrame(results)

### Step 5: Make atom numbering consistent across ensemble of conformers for each ligand
def make_ensemble_atom_numbering_consistent (df):
    df['lig_name'] = df['log_name'].str.split('_').str[0]
    ligands = df.groupby('lig_name')
    for prefix, ligand in ligands:
        unique_rows = ligand.drop(columns=['log_name', 'lig_name']).drop_duplicates()
        if len(unique_rows) > 1:
            if (len(unique_rows) == 2 and #if there are only 2 unique sets of values
                unique_rows.iloc[0]['Ni'] == unique_rows.iloc[1]['Ni'] and #and the Ni values are the same
                ((unique_rows.iloc[0]['N1'] == unique_rows.iloc[1]['N2']) and
                    (unique_rows.iloc[0]['N2'] == unique_rows.iloc[1]['N1']))): # and the N1/N2 values are just swapped
                chosen_values = unique_rows.iloc[0]
                for col in unique_rows.columns:
                    df.loc[df['lig_name'] == prefix, col] = chosen_values[col]
                continue
            if len(unique_rows) > 2: #this indicates that there is a more complex issue and potentially different substructures
                print(f"\nFlagged Prefix: {prefix}")
                print("Detected differing sets of values:")

                for i, unique_row in enumerate(unique_rows.iterrows(), start=1):
                    print(f"Option {i}:")
                    print(unique_row[1].to_dict())

                print(f"{len(unique_rows) + 1}: Leave as is")

                choice = int(input(f"\nEnter the option number (1-{len(unique_rows) + 1}) you want to apply for prefix '{prefix}': ").strip())

                if choice <= len(unique_rows):
                    chosen_values = unique_rows.iloc[choice - 1]
                    for col in unique_rows.columns:
                        df.loc[df['lig_name'] == prefix, col] = chosen_values[col]
                else:
                    print(f"Leaving the rows for prefix '{prefix}' as they are.")
    df.sort_values(by=['lig_name'], inplace=True)
    df = df.drop(columns=['lig_name'])
    return df



### Step 7: Verify atom numbering by checking atomic numbers match atom labels
def verify_atom_numbering(df, sdf_dir = os.getcwd()):
    df_columns = list(df.columns)
    for i,row in df.iterrows():
        log_name_base = row['log_name']
        sdf_name = log_name_base + '.sdf'
        sdf_path = os.path.join(sdf_dir,sdf_name)
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        except:
            print (f'error reading sdf for {log_name_base}')
            continue

        N1_index = row['N1'] - 1
        N1_atomic_num = mol.GetAtomWithIdx(N1_index).GetAtomicNum()

        N2_index = row['N2'] - 1
        N2_atomic_num = mol.GetAtomWithIdx(N2_index).GetAtomicNum()

        C1_index = row['C1'] - 1
        C1_atomic_num = mol.GetAtomWithIdx(C1_index).GetAtomicNum()

        C2_index = row['C2'] - 1
        C2_atomic_num = mol.GetAtomWithIdx(C2_index).GetAtomicNum()

        if C2_atomic_num != 6 or C1_atomic_num != 6 or N2_atomic_num !=7 or N1_atomic_num !=7:
            print (f'error in atom numbering for {log_name_base}')

        if 'F1' in df_columns and 'F2' in df_columns:
            F1_index = row['F1'] - 1
            F2_index = row['F2'] - 1

            F1_atomic_num = mol.GetAtomWithIdx(F1_index).GetAtomicNum()
            F2_atomic_num = mol.GetAtomWithIdx(F2_index).GetAtomicNum()

            if F1_atomic_num !=9 or F2_atomic_num !=9:
                print (f'error in atom numbering for {log_name_base}')
        if 'H1' in df_columns and 'H2' in df_columns:
            H1_index = row['H1'] - 1
            H2_index = row['H2'] - 1

            H1_atomic_num = mol.GetAtomWithIdx(H1_index).GetAtomicNum()
            H2_atomic_num = mol.GetAtomWithIdx(H2_index).GetAtomicNum()

            if H1_atomic_num !=1 or H2_atomic_num !=1:
                print (f'error in atom numbering for {log_name_base}')


def find_pyox_ligands_and_renumber(df, sdf_dir=os.getcwd()):
    # first deal with the 5-membered cores
    df5 = df[df['ligand_type'] == 'core_5'].copy()
    pyox_ligands = []
    other_ligands = []
    for i,row in df5.iterrows():
        log_name_base = row['log_name']
        N1_index = row['N1'] - 1 #subtract 1 because mol is 0 indexed and gaussian is 1-indexed
        N2_index = row['N2'] - 1
        C1_index = row['C1'] - 1
        C2_index = row['C2'] - 1

        sdf_name = log_name_base + '.sdf'
        sdf_path = os.path.join(sdf_dir,sdf_name)

        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        except:
            print (f'bad input file for {sdf_path}')
        ringsize_N1_is_6 = mol.GetAtomWithIdx(N1_index).IsInRingSize(6)
        ringsize_N1_is_5 = mol.GetAtomWithIdx(N1_index).IsInRingSize(5)

        ringsize_N2_is_6 = mol.GetAtomWithIdx(N2_index).IsInRingSize(6)
        ringsize_N2_is_5 = mol.GetAtomWithIdx(N2_index).IsInRingSize(5)

        if ringsize_N1_is_6 is False and ringsize_N2_is_6 is False and ringsize_N2_is_5 is True and ringsize_N1_is_5 is True:
            # if there are no 6 membered rings and core is 5 membered, it is biim/biox
            other_ligands.append(log_name_base)
        elif ringsize_N1_is_6 is True and ringsize_N2_is_6 is True and ringsize_N2_is_5 is True and ringsize_N1_is_5 is True:
            # if N1 and N2 are each in a 6 membered ring and also a 5 membered core, it is a bpy or phen
            other_ligands.append(log_name_base)
        else:
            # should be a pyox ligand
            pyox_ligands.append(log_name_base)

    if len(other_ligands) > 0:
        other_lig_df = df5.loc[df5['log_name'].isin(other_ligands)].copy()
        other_lig_df.loc[:, 'ligand_class'] = 'other'
    if len(pyox_ligands) > 0:
        pyox_lig_df = df5.loc[df5['log_name'].isin(pyox_ligands)].copy()
        pyox_lig_df.loc[:, 'ligand_class'] = 'pyox'
        for i,row in pyox_lig_df.iterrows():
                log_name_base = row['log_name']
                N1_index = row['N1'] - 1 #subtract 1 because mol is 0 indexed and gaussian is 1-indexed
                N2_index = row['N2'] - 1
                C1_index = row['C1'] - 1
                C2_index = row['C2'] - 1

                sdf_name = log_name_base + '.sdf'
                sdf_path = os.path.join(sdf_dir,sdf_name)

                try:
                    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
                    mol = next((m for m in suppl if m is not None), None)
                except:
                    print (f'bad input file for {sdf_path}')

                ringsize_N1_is_6 = mol.GetAtomWithIdx(N1_index).IsInRingSize(6)
                ringsize_N1_is_5 = mol.GetAtomWithIdx(N1_index).IsInRingSize(5)

                ringsize_N2_is_6 = mol.GetAtomWithIdx(N2_index).IsInRingSize(6)
                ringsize_N2_is_5 = mol.GetAtomWithIdx(N2_index).IsInRingSize(5)

                if ringsize_N1_is_6 is True and ringsize_N2_is_5 is True and ringsize_N2_is_6 is False: # is N1 is already pyridine, df is correct as is
                    # write back corrected values (+1 because dataframe is 1-based)
                    pyox_lig_df.loc[i, 'N1'], pyox_lig_df.loc[i, 'N2']  = N1_index + 1, N2_index + 1
                    pyox_lig_df.loc[i, 'C1'], pyox_lig_df.loc[i, 'C2'] = C1_index + 1, C2_index + 1

                elif ringsize_N2_is_6 is True and ringsize_N1_is_5 is True and ringsize_N1_is_6 is False:
                    pyox_lig_df.loc[i, 'N1'], pyox_lig_df.loc[i, 'N2'] = N2_index + 1, N1_index + 1
                    pyox_lig_df.loc[i, 'C1'], pyox_lig_df.loc[i, 'C2'] = C2_index + 1, C1_index + 1

    if len(other_ligands) > 0 and len(pyox_ligands) > 0:
        df5_pyox_corrected = pd.concat([pyox_lig_df, other_lig_df], ignore_index=True)
    elif len(other_ligands) == 0 and len(pyox_ligands) > 0:
        df5_pyox_corrected = pyox_lig_df
    elif len(other_ligands) > 0 and len(pyox_ligands) == 0:
        df5_pyox_corrected = other_lig_df
    if len(other_ligands) == 0 and len(pyox_ligands) == 0:
        df5_pyox_corrected = pd.DataFrame()

    # now deal with the 6-membered cores
    df6 = df[df['ligand_type'] == 'core_6'].copy()
    pyr6_ligands = []
    box_ligands = []
    for i,row in df6.iterrows():
        log_name_base = row['log_name']
        N1_index = row['N1'] - 1
        N2_index = row['N2'] - 1
        C1_index = row['C1'] - 1
        C2_index = row['C2'] - 1

        sdf_name = log_name_base + '.sdf'
        sdf_path = os.path.join(sdf_dir,sdf_name)

        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = next((m for m in suppl if m is not None), None)
        except:
            print (f'bad input file for {sdf_path}')
        ringsize_N1_is_6 = mol.GetAtomWithIdx(N1_index).IsInRingSize(6)
        ringsize_N1_is_5 = mol.GetAtomWithIdx(N1_index).IsInRingSize(5)

        ringsize_N2_is_6 = mol.GetAtomWithIdx(N2_index).IsInRingSize(6)
        ringsize_N2_is_5 = mol.GetAtomWithIdx(N2_index).IsInRingSize(5)

        if ringsize_N2_is_6 is True and ringsize_N2_is_5 is True and ringsize_N1_is_6 is True and ringsize_N1_is_5 is True:
            # if N1 and N2 are each in a 5 membered ring and also a 6 membered core, it is a box ligand
            box_ligands.append(log_name_base)
        else:
            # should be a pyr6 ligand
            pyr6_ligands.append(log_name_base)
    if len(box_ligands) > 0:
        box_lig_df = df6.loc[df6['log_name'].isin(box_ligands)].copy()
        box_lig_df.loc[:, 'ligand_class'] = 'other'
    if len(pyr6_ligands) > 0:
        pyr6_lig_df = df6.loc[df6['log_name'].isin(pyr6_ligands)].copy()
        pyr6_lig_df.loc[:, 'ligand_class'] = 'pyr6'
        for i,row in pyr6_lig_df.iterrows():
                log_name_base = row['log_name']
                N1_index = row['N1'] - 1 #subtract 1 because mol is 0 indexed and gaussian is 1-indexed
                N2_index = row['N2'] - 1
                C1_index = row['C1'] - 1
                C2_index = row['C2'] - 1

                sdf_name = log_name_base + '.sdf'
                sdf_path = os.path.join(sdf_dir,sdf_name)

                try:
                    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
                    mol = next((m for m in suppl if m is not None), None)
                except:
                    print (f'bad input file for {sdf_path}')

                ringsize_N1_is_6 = mol.GetAtomWithIdx(N1_index).IsInRingSize(6)
                ringsize_N1_is_5 = mol.GetAtomWithIdx(N1_index).IsInRingSize(5)

                ringsize_N2_is_6 = mol.GetAtomWithIdx(N2_index).IsInRingSize(6)
                ringsize_N2_is_5 = mol.GetAtomWithIdx(N2_index).IsInRingSize(5)

                if ringsize_N1_is_6 is True and ringsize_N1_is_5 is True and ringsize_N2_is_6 is True and ringsize_N2_is_5 is False: # is N1 is already pyridine, df is correct as is
                    # then N1 is the oxazoline and N2 is the pyridine so we need to swap them
                    pyr6_lig_df.loc[i, 'N1'], pyr6_lig_df.loc[i, 'N2']  = N2_index + 1, N1_index + 1
                    pyr6_lig_df.loc[i, 'C1'], pyr6_lig_df.loc[i, 'C2'] = C2_index + 1, C1_index + 1

                elif ringsize_N1_is_6 is True and ringsize_N1_is_5 is False and ringsize_N2_is_6 is True and ringsize_N2_is_5 is True:
                    # then N2 is the oxazoline and N1 is the pyridine, which is correct
                    pyr6_lig_df.loc[i, 'N1'], pyr6_lig_df.loc[i, 'N2'] = N1_index + 1, N2_index + 1
                    pyr6_lig_df.loc[i, 'C1'], pyr6_lig_df.loc[i, 'C2'] = C1_index + 1, C2_index + 1
    if len(box_ligands) > 0 and len(pyr6_ligands) > 0:
        df6_pyox_corrected = pd.concat([box_lig_df, pyr6_lig_df], ignore_index=True)
    elif len(box_ligands) == 0 and len(pyr6_ligands) > 0:
        df6_pyox_corrected = pyr6_lig_df
    elif len(box_ligands) > 0 and len(pyr6_ligands) == 0:
        df6_pyox_corrected = box_lig_df
    if len(box_ligands) == 0 and len(pyr6_ligands) == 0:
        df6_pyox_corrected = pd.DataFrame()
    df_pyox_corrected = pd.concat([df5_pyox_corrected, df6_pyox_corrected], ignore_index=True)
    return df_pyox_corrected


def get_vbur_quadrants_only(dataframe, a1, ex1, ex2, z1, z2, radius):
    """
    Uses Morfeus to calculate %Vbur at a single radius for atom (a1) in df.
    """
    atom = str(a1)
    atom_ex1 = str(ex1)
    atom_ex2 = str(ex2)
    atom_z1 = str(z1)
    atom_z2 = str(z2)

    rows = []  # Collect result rows here

    for index, row in dataframe.iterrows():
        try:
            log_file = row['log_name']
            atom1 = row[atom]
            exclude1 = row[atom_ex1]
            exclude2 = row[atom_ex2]
            zaxis1 = row[atom_z1]
            zaxis2 = row[atom_z2]
            xzatom = row[atom_z2]

            streams, error = get_outstreams(log_file)
            if error:
                print(error)
                row_i = {
                    f'%Vbur_{atom}_quadrant_+,+': "no data",
                    f'%Vbur_{atom}_quadrant_–,+': "no data",
                    f'%Vbur_{atom}_quadrant_–,–': "no data",
                    f'%Vbur_{atom}_quadrant_+,–': "no data",
                }
                rows.append(row_i)
                continue

            log_coordinates = get_geom(streams)
            elements = np.array([entry[0] for entry in log_coordinates])
            coordinates = np.array([entry[1:] for entry in log_coordinates], dtype=float)

            vbur = BuriedVolume(
                elements,
                coordinates,
                int(atom1),
                radius=radius,
                include_hs=True,
                excluded_atoms=[exclude1, exclude2],
                z_axis_atoms=[int(zaxis1), int(zaxis2)],
                xz_plane_atoms=[int(xzatom)],
            )

            vbur.octant_analysis()
            bv_quadrants = vbur.quadrants["percent_buried_volume"]

            row_i = {
                f'%Vbur_{atom}_quadrant_+,+': bv_quadrants[1],
                f'%Vbur_{atom}_quadrant_–,+': bv_quadrants[2],
                f'%Vbur_{atom}_quadrant_–,–': bv_quadrants[3],
                f'%Vbur_{atom}_quadrant_+,–': bv_quadrants[4],
            }

        except Exception as e:
            print(f"**** Unable to acquire Vbur quadrants for: {row.get('log_name', 'unknown')}.log")
            row_i = {
                f'%Vbur_{atom}_quadrant_+,+': "no data",
                f'%Vbur_{atom}_quadrant_–,+': "no data",
                f'%Vbur_{atom}_quadrant_–,–': "no data",
                f'%Vbur_{atom}_quadrant_+,–': "no data",
            }

        rows.append(row_i)
    vbur_quadoct_dataframe = pd.DataFrame(rows)
    print("Buried volume quadrants and octants function has completed")
    return pd.concat([dataframe.reset_index(drop=True), vbur_quadoct_dataframe], axis=1)


### Step 9: For the non-pyridine containing ligands, label the N1/N2 according to a property value
def renumber_non_pyox_ligands(df, prefix='Lig', suffix='_'):
    other_ligands_df = df[df['ligand_class'] == 'other'].copy()
    other_ligands_list = other_ligands_df['log_name'].tolist()
    if other_ligands_list:
        pyox_df = df[~df['log_name'].isin(other_ligands_list)].copy()
    else:
        pyox_df = pd.DataFrame()
    columns = list(other_ligands_df.columns)
    if 'F1' in columns and 'F2' in columns:
        other_ligands_df = get_goodvibes_e(other_ligands_df, data_dir=Path('.'), temp=298.15)
        other_ligands_df = get_vbur_quadrants_only(other_ligands_df, a1="Ni", ex1="F1", ex2="F2", z1="N1", z2="N2", radius=6.5)
        pass
    if 'H1' in columns and 'H2' in columns:
        other_ligands_df = get_goodvibes_e(other_ligands_df, data_dir=Path('.'), temp=298.15)
        other_ligands_df = get_vbur_quadrants_only(other_ligands_df, a1="Ni", ex1="H1", ex2="H2", z1="N1", z2="N2", radius=6.5)
        pass

    other_ligands_df['north_hemisphere'] = other_ligands_df['%Vbur_Ni_quadrant_+,+'] + other_ligands_df['%Vbur_Ni_quadrant_+,–']
    other_ligands_df['south_hemisphere'] = other_ligands_df['%Vbur_Ni_quadrant_–,–'] + other_ligands_df['%Vbur_Ni_quadrant_–,+']

    prefix = prefix
    suffix = suffix

    energy_col_header = "G(T)_spc(Hartree)"
    compound_list = []

    for index, row in other_ligands_df.iterrows():
        log_file = row['log_name']
        prefix_and_compound = log_file.split(str(suffix))
        compound = prefix_and_compound[0].split(str(prefix))
        compound_list.append(compound[1])

    compound_list = list(set(compound_list))
    compound_list.sort()

    # if north hemisphere is bigger than N2 is correctly assigned and is south hemisphere is bigger it needs to swap (i.e. north hemisphere is N2 and south hemisphere is N1)
    property_to_compare = ["north_hemisphere", "south_hemisphere"]
    dict_of_N2 = {}
    dict_of_N1 = {}
    dict_of_C2 = {}
    dict_of_C1 = {}

    for compound in compound_list:
        substring = str(prefix) + str(compound) + str(suffix)
        compounddf = other_ligands_df[other_ligands_df["log_name"].str.startswith(substring)]
        compounddf = compounddf.reset_index(drop = True)
        compounddf["∆G(Hartree)"] = compounddf[energy_col_header] - compounddf[energy_col_header].min()
        low_e_index = compounddf[compounddf["∆G(Hartree)"] == 0].index.tolist()
        prop_1 = compounddf[str(property_to_compare[0])][low_e_index[0]] #first property listed above, should be N2 (north)
        prop_2 = compounddf[str(property_to_compare[1])][low_e_index[0]] #second property listed above, should be N1
        if prop_1 >= prop_2: # if the N2 property is greater than the N1 property, N2 should stay as N2 and N1 should stay as N1 - and then C2 and C1 stay as is
            N2 = compounddf["N2"][low_e_index[0]] #N2 is bigger
            C2 = compounddf["C2"][low_e_index[0]] #C2 has to move with N2
            N1 = compounddf["N1"][low_e_index[0]] #N1 is smaller
            C1 = compounddf["C1"][low_e_index[0]] #C1 has to move with N1
        elif prop_1 < prop_2: # if N1 is larger than N2, reassign N1 as N2 and C1 as C2
            N2 = compounddf["N1"][low_e_index[0]]
            C2 = compounddf["C1"][low_e_index[0]]
            N1 = compounddf["N2"][low_e_index[0]]
            C1 = compounddf["C2"][low_e_index[0]]

        for index, row in compounddf.iterrows():
            key = row['log_name']
            dict_of_N2[key] = N2
            dict_of_N1[key] = N1
            dict_of_C2[key] = C2
            dict_of_C1[key] = C1


    other_ligands_df['N2'] = other_ligands_df['log_name'].map(dict_of_N2)
    other_ligands_df['N1'] = other_ligands_df['log_name'].map(dict_of_N1)
    other_ligands_df['C2'] = other_ligands_df['log_name'].map(dict_of_C2)
    other_ligands_df['C1'] = other_ligands_df['log_name'].map(dict_of_C1)
    other_ligands_df = other_ligands_df[columns]
    full_readjusted_df = pd.concat([pyox_df, other_ligands_df], ignore_index=True)
    full_readjusted_df.drop(columns=['ligand_class', 'ligand_type'], inplace=True)
    return full_readjusted_df
