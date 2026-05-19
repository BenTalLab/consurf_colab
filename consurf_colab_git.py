

import sys
import traceback
import socket
import getpass
import re
import os
import json
import shutil
import gzip
import requests
import subprocess
import time
import tempfile
import urllib
import tarfile
import shutil

from zipfile import ZipFile
from pathlib import Path
from IPython.display import HTML, display, IFrame


bayesInterval = 3
ColorScale = {0 : 9, 1 : 8, 2 : 7, 3 : 6, 4 : 5, 5 : 4, 6 : 3, 7 : 2, 8 : 1}


vars = {}
form = {}
LOG = None

def install():

    # get fonts
    if not os.path.isfile("FONTS_READY"):

        print("Getting fonts.")
        os.system("wget \"https://sourceforge.net/projects/dejavu/files/dejavu/2.37/dejavu-fonts-ttf-2.37.tar.bz2\" -O \"FONTS.tar.bz2\"")
        FONTS = tarfile.open("FONTS.tar.bz2")
        FONTS.extractall()
        FONTS.close()
        os.system("touch FONTS_READY")

    # install prank
    if not os.path.isfile("PRANK_READY"):

        print("Installing prank.")
        os.system("apt-get install prank")
        os.system("touch PRANK_READY")

    # install clustalw
    if not os.path.isfile("CLUSTALW_READY"):

        print("Installing CLUSTALW.")
        os.system("apt-get -qq install -y clustalw")
        os.system("touch CLUSTALW_READY")

    # install prottest
    if not os.path.isfile("PROTTEST_READY"):

        print("Installing prottest.")
        #os.system("git clone https://github.com/ddarriba/prottest3.git")
        response = urllib.request.urlopen("https://github.com/ddarriba/prottest3/releases/download/3.4.2-release/prottest-3.4.2-20160508.tar.gz").read()

        prottest_zip = "prottest.tar.gz"
        PROTTEST_ZIP = open(prottest_zip, 'wb')
        PROTTEST_ZIP.write(response)
        PROTTEST_ZIP.close()

        PROTTEST_EXTRACT = tarfile.open(prottest_zip)
        PROTTEST_EXTRACT.extractall()
        PROTTEST_EXTRACT.close()
        os.system("touch PROTTEST_READY")

    # install muscle
    if not os.path.isfile("MUSCLE_READY"):

        print("Installing muscle.")
        os.system("apt-get -qq install muscle")
        os.system("touch MUSCLE_READY")

    # install mafft
    if not os.path.isfile("MAFFT_READY"):

        print("Installing mafft.")
        os.system("apt-get install mafft")
        os.system("touch MAFFT_READY")

    # install rate4site
    if not os.path.isfile("RATE4SITE_READY"):

        print("Installing rate4site.")

        # create directory for rate4site
        os.system("git clone https://github.com/barakav/r4s_for_collab.git")

        # create directory for rate4site slow
        shutil.copytree(vars['rate4site_dir'], vars['rate4site_slow_dir'])

        # make rate4site
        try:

            os.chdir(vars['rate4site_dir'])
            os.system("make")
            os.chdir(vars['root_dir'])

        except Exception as e:

            print(e)
            os.chdir(vars['root_dir'])
            raise("Installing rate4site failed.")

        # change the make file and make rate4site slow
        try:

            os.chdir(vars['rate4site_slow_dir'])
            os.remove("Makefile") # delete regular file
            os.rename("Makefile_slow", "Makefile") # change to slow file
            os.system("make")
            os.chdir(vars['root_dir'])

        except Exception as e:

            print(e)
            os.chdir(vars['root_dir'])
            raise("Installing rate4site failed.")

        os.system("touch RATE4SITE_READY")


    # install cd-hit
    if not os.path.isfile("CDHIT_READY"):

        print("installing cd-hit.")
        os.system("git clone https://github.com/weizhongli/cdhit.git")
        """
        response = urllib.request.urlopen("https://github.com/weizhongli/cdhit/archive/refs/heads/master.zip").read()

        cd_hit_zip = "cd-hit.zip"
        CD_HIT_ZIP = open(cd_hit_zip, 'wb')
        CD_HIT_ZIP.write(response)
        CD_HIT_ZIP.close()

        CD_HIT_EXTRACT = ZipFile(cd_hit_zip, 'r')
        CD_HIT_EXTRACT.extractall()
        CD_HIT_EXTRACT.close()
        """
        try:

            os.chdir("cdhit")
            os.system("make")
            os.chdir(vars['root_dir'])

        except Exception as e:

            print(e)
            os.chdir(vars['root_dir'])
            raise("Installing cd-hit failed.")

        os.system("touch CDHIT_READY")

    # install mmseqs2
    if not os.path.isfile("COLABFOLD_READY"):

        print("Installing colabfold.")
        os.system("pip install -q --no-warn-conflicts 'colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold'")
        os.system("pip install --upgrade dm-haiku")
        os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabfold colabfold")
        os.system("ln -s /usr/local/lib/python3.*/dist-packages/alphafold alphafold")
        # patch for jax > 0.3.25
        os.system("sed -i 's/weights = jax.nn.softmax(logits)/logits=jnp.clip(logits,-1e8,1e8);weights=jax.nn.softmax(logits)/g' alphafold/model/modules.py")
        os.system("pip install -q biopython==1.81")
        os.system("touch COLABFOLD_READY")

# install biopython
if not os.path.isfile("BIOPYTHON_READY"):

    print("Installing biopython.")
    os.system("pip install biopython")
    os.system("touch BIOPYTHON_READY")

# install fpdf
if not os.path.isfile("FPDF_READY"):

    print("Installing fpdf.")
    os.system("pip install fpdf")
    os.system("touch FPDF_READY")
    
# install py3dmol
if not os.path.isfile("PY3DMOL_READY"):
    
    print("Installing py3dmol.")
    os.system("pip install py3Dmol")
    os.system("touch PY3DMOL_READY")

from datetime import date
from datetime import datetime
from Bio import AlignIO
from Bio import SeqIO
from Bio import SearchIO
from Bio import Phylo
from Bio import Align
from Bio.Align import substitution_matrices
from google.colab import files, output
from Bio.Blast import NCBIWWW
import Bio
import fpdf
import math
import py3Dmol
import uuid


def choose_chain(PDB_name):

    # in the case of nmr we choose the first model
    try:

        PDB_FILE = open(PDB_name, 'r')

    except:

        raise Exception("Can't read the PDB file.")

    # later we find the format (PDB/mmCIF) and we change the files name
    temp_pdb = "temp_file.txt"
    try:

        PDB_TEMP = open(temp_pdb, 'w')

    except:

        raise Exception("Error: choose_chain - Can't open file for writing.")

    line = PDB_FILE.readline()
    while line != "":

        if re.match(r'^MODEL\s+2', line):

            # nmr found
            break

        else:

            PDB_TEMP.write(line)

        line = PDB_FILE.readline()

    PDB_TEMP.close()
    PDB_FILE.close()

    # we now find the chains
    PDB_FILE = open(temp_pdb, 'r')



    chains = []
    last_chain_found = ""
    chain_index = 0
    nmr_struct = False

    # for mmCIF format
    found_auth_comp_id_column = False
    found_auth_asym_id_column = False
    found_label_comp_id_column = False
    found_label_asym_id_column = False
    found_column_numbers = False
    auth_comp_id_column = 0
    auth_asym_id_column = 0
    label_comp_id_column = 0
    label_asym_id_column = 0

    line = PDB_FILE.readline()
    while line != "":

        if line[:4] == "ATOM":

            if (found_auth_comp_id_column or found_label_comp_id_column) and (found_auth_asym_id_column or found_label_asym_id_column):

                # mmCIF format
                found_column_numbers = True
                words = line.split()
                num_columns = len(words)
                if found_auth_comp_id_column:
                    
                    acid = words[num_columns - auth_comp_id_column]
                    
                else:
                    
                    acid = words[num_columns - label_comp_id_column]
                    
                if len(acid) == 3:

                    # it's an amino acid
                    if found_auth_asym_id_column:
                        
                        chain = words[num_columns - auth_asym_id_column]
                        
                    else:
                        
                        chain = words[num_columns - label_asym_id_column]
                        
                    if chain != last_chain_found:

                        chains.append(chain)
                        last_chain_found = chain

            else:

                # PDB format
                acid = line[17:20]
                if len(acid) == 3:

                    # it's an amino acid
                    chain = line[21:22]
                    if chain == " ":

                        chains[0] = "NONE"

                    elif chain != last_chain_found:

                        chains.append(chain)
                        last_chain_found = chain

        # ATOM not found
        # the format maybe mmCIF
        # in this case we need to know what each column means
        elif line.strip() == "_atom_site.auth_comp_id":

            found_auth_comp_id_column = True

        elif line.strip() == "_atom_site.auth_asym_id":

            found_auth_asym_id_column = True

        elif line.strip() == "_atom_site.label_comp_id":

            found_label_comp_id_column = True

        elif line.strip() == "_atom_site.label_asym_id":

            found_label_asym_id_column = True
            
        if found_auth_comp_id_column and not found_column_numbers:

            auth_comp_id_column += 1

        if found_auth_asym_id_column and not found_column_numbers:

            auth_asym_id_column += 1

        if found_label_comp_id_column and not found_column_numbers:

            label_comp_id_column += 1

        if found_label_asym_id_column and not found_column_numbers:

            label_asym_id_column += 1
            
        line = PDB_FILE.readline()

    # we found the chains
    # we change the file name according to the format
    if found_column_numbers:

        vars['cif_or_pdb'] = "cif"
        vars['pdb_file_name'] = "file.cif"

    else:

        vars['cif_or_pdb'] = "pdb"
        vars['pdb_file_name'] = "pdb_file.ent"

    os.rename(temp_pdb, vars['pdb_file_name'])
    return chains

def check_msa_tree_match(ref_msa_seqs, ref_tree_nodes):

    for node in ref_tree_nodes:

        if not node in ref_msa_seqs:

            raise Exception("The node %s is in the tree and not in the MSA." %node)

    for seq_name in ref_msa_seqs: #check that all the msa nodes are in the tree

        if not seq_name in ref_tree_nodes:

            raise Exception("The sequence %s is in the msa and not in the tree" %seq_name)

    vars['unique_seqs'] = len(ref_msa_seqs)

def check_validity_tree_file():

	  # checks validity of tree file and returns an array with the names of the nodes
    try:

        TREEFILE = open(vars['tree_file'], 'r')

    except:

        raise Exception("Can't read the tree file.")

    tree = TREEFILE.read()
    TREEFILE.close()
    tree.replace("\n", "")
    if tree[-1] != ';':

        tree += ';'

    leftBrackets = 0
    rightBrackets = 0
    noRegularFormatChar = ""
    nodes = []
    node_name = ""
    in_node_name = False
    in_node_score = False
    for char in tree:

        if char == ':':

            if in_node_name:

                nodes.append(node_name)

            node_name = ""
            in_node_name = False
            in_node_score = True

        elif char == '(':

            leftBrackets += 1

        elif char == ')':

            rightBrackets += 1
            in_node_score = False

        elif char == ',':

            in_node_score = False

        elif char != ';':

            if char in "!@#$^&*~`{}'?<>\\" and not char in noRegularFormatChar:

                noRegularFormatChar += " '" + char + "', "

            if not in_node_score:

                node_name += char
                in_node_name = True

    if leftBrackets != rightBrackets:

        raise Exception("The tree is missing parentheses.")

    if noRegularFormatChar != "":

        raise Exception("The tree contains the following characters " + noRegularFormatChar[:-2])

    return nodes


def check_msa_format():

    MSA = open(vars['user_msa_file_name'], 'r')
    line = MSA.readline()
    while line != "":

        line = line.strip()
        if line == "":

            line = MSA.readline()
            continue

        if line[:4] == "MSF:":

            format = "msf"
            break

        elif line[0] == '>':

            format = "fasta"
            break

        elif line[0] == '#':

            format = "nexus"
            break

        elif line[0] == 'C':

            format = "clustal"
            break

        elif line[0] == 'P':

            format = "gcg"
            break

        else:

            MSA.close()
            raise Exception("Unknown format.")

        line = MSA.readline()

    MSA.close()
    return format

def multiple_chains(chains):

    print("Please choose a chain from this list:")
    i = 1
    for chain in chains:

        print("%d. %s" %(i, chain))
        i += 1

    while True:

        chain_index = input("Press the number of chain.\n")
        if chain_index.isdigit():

            chain_index = int(chain_index)
            if chain_index > 0 and chain_index < i:

                form['PDB_chain'] = chains[chain_index - 1]
                print("You chose the chain %s.\n" %form['PDB_chain'])
                break

        print("Wrong input.")

def upload_PDB():

    # we get the PDB file
    while True:

        PDB_uniprot = input("Do you have a PDB/uniprot ID? (Y/N) Please type 'Y' if you have an ID, and 'N' if you don't have an ID:\n")
        if PDB_uniprot.upper() == "Y":

            ID = input("Please enter your ID. If you have a uniprot ID, colab will fetch the structure from the alphafold database:\n")
            ID = ID.upper().strip()
            form['pdb_ID'] = ID
            if len(ID) == 4:

                # PDB ID
                pdb_url = "https://files.rcsb.org/download/%s.pdb" %ID
                cif_url= "https://files.rcsb.org/download/%s.cif" %ID

            else:

                # uniprot ID
                pdb_url = "https://alphafold.ebi.ac.uk/files/AF-%s-F1-model_v6.pdb" %ID
                cif_url= "https://alphafold.ebi.ac.uk/files/AF-%s-F1-model_v6.cif" %ID

            try:

                response = urllib.request.urlopen(pdb_url).read()

            except:

                try:

                    response = urllib.request.urlopen(cif_url).read()

                except:

                    raise Exception("Could not download model file.")

            PDB_name = "temp_PDB.txt"
            PDB_FILE = open(PDB_name, 'wb')
            PDB_FILE.write(response)
            PDB_FILE.close()
            vars['Used_PDB_Name'] = vars['job_name'] + "_" + ID
            print()
            break

        elif PDB_uniprot.upper() == "N":

            print("Please upload your model.\n")
            PDB_file = files.upload()
            PDB_name = (list(PDB_file))[0]
            vars['Used_PDB_Name'] = PDB_name.replace(" ", "")
            match = re.search(r'(\S+)\.', vars['Used_PDB_Name'])
            if match:

                vars['Used_PDB_Name'] = vars['job_name'] + "_" + match.group(1)

            break

        else:

            print("Wrong input. Pease type 'Y' if you have an ID, and 'N' if you don't have an ID.")

    # the user must choose a chain
    chains = choose_chain(PDB_name)
    if len(chains) == 1:

        form['PDB_chain'] = chains[0]
        print("The PDB has only one chain %s\n" %form['PDB_chain'])

    else:

        multiple_chains(chains)

    os.rename(PDB_name, vars['pdb_file_name'])

    if form['PDB_chain'] != "NONE":

        vars['Used_PDB_Name'] += "_" + form['PDB_chain']

def upload_sequence():

    vars['query_string'] = "Input_seq"
    vars['SEQRES_seq'] = ""
    vars['ATOM_without_X_seq'] = ""
    while True:

        vars['protein_seq_string'] = input("Please enter your sequence.\n")
        if '>' in vars['protein_seq_string']:

            print("You should upload only one sequence, without the sequence name. If you want to upload a MSA, use the correct mode.")

        else:

            break

    # delete sequence name and white spaces
    #vars['protein_seq_string'] = re.sub(r'>\S*', "", vars['protein_seq_string'])
    vars['protein_seq_string'] = re.sub(r'\s', "", vars['protein_seq_string'])
    print()
    if re.match(r'^[actguACTGUNn]+$', vars['protein_seq_string']):

        raise Exception("Your sequence is only composed of Nucleotides (i.e. :A,T,C,G).")

def upload_MSA():

    print("Please upload your MSA.")
    MSA_file = files.upload()
    MSA_name = (list(MSA_file))[0]

    vars['user_msa_file_name'] = MSA_name
    format = check_msa_format()
    alignment = AlignIO.read(MSA_name, format)

    MSA_FASTA = open(vars['msa_fasta'], 'w')
    seq_names = []
    seqs = []
    num_of_seq = 0
    for record in alignment:

        num_of_seq += 1

        seq_name = record.id
        seq = record.seq

        # we save the sequences and their name for two reasons
        # to let the user choose the query
        # to check if they much the tree, if there is one.
        seq_names.append(seq_name)
        seqs.append(seq)

        # we write the msa in fasta format
        MSA_FASTA.write(">%s\n%s\n" %(seq_name, seq))

    MSA_FASTA.close()

    if num_of_seq < 5:

        raise Exception("There are %d sequences in the msa. There must be at least five." %num_of_seq)

    vars['unique_seqs'] = num_of_seq
    vars['final_number_of_homologoues'] = num_of_seq

    print("\nThe names of the sequences are:\n")
    i = 1
    for seq_name in seq_names:

        print("%d. %s" %(i, seq_name))
        i += 1

    while True:

        query_number = input("\nEnter the number of the query.\n")
        if query_number.isdigit():

            query_number = int(query_number)
            if query_number > 0 and query_number < i:

                vars['query_string'] = seq_names[query_number - 1]
                vars['protein_seq_string'] = seqs[query_number - 1]
                vars['protein_seq_string'] = vars['protein_seq_string'].replace("-", "")
                vars['protein_seq_string'] = vars['protein_seq_string'].upper()
                break

        print("Wrong input.")

    print("Chosen query is %s\n" %vars['query_string'])

    try:

        MSA_FASTA = open(vars['user_msa_file_name'], 'w')

    except:

        raise Exception("Can't open file for writing")

    for i in range(len(seq_names)):

        MSA_FASTA.write(">%s\n%s\n" %(seq_names[i], seqs[i]))

    MSA_FASTA.close()

    vars['msa_SEQNAME'] = vars['query_string']
    
    # include Tree
    if vars['running_mode'] == "_mode_pdb_msa_tree" or vars['running_mode'] == "_mode_msa_tree":

        print("Please upload your tree.")
        Tree_file = files.upload()
        Tree_name = (list(Tree_file))[0]
        os.rename(Tree_name, vars['tree_file'])
        nodes = check_validity_tree_file()
        check_msa_tree_match(seq_names, nodes)
        print()

def create_MSA_parameters():

    # the user may want to change the default parameters
    form['E_VALUE'] = 0.0001
    form['MAX_NUM_HOMOL'] = "150"
    vars['hit_redundancy'] = 95
    form['MSAprogram'] = "MAFFT"
    form['MIN_IDENTITY'] = 35
    form['best_uniform_sequences'] = "sample"
    while True:

        print("\nMSA parameters:")
        print("1. Maximum number of homologs - %s" %form['MAX_NUM_HOMOL'])
        print("2. Maximum redundancy - %d" %vars['hit_redundancy'])
        print("3. MSA building program - %s" %form['MSAprogram'])
        print("4. Minimum identity - %d" %form['MIN_IDENTITY'])
        print("5. Homolog selection - %s" %form['best_uniform_sequences'])
        print("6. E-value cutoff - %.16g" %form['E_VALUE'])
        params = input("Would you like to change? (Y/N) Pease type 'Y' if you do want to change the parameters, and 'N' if you don't want to change them:\n")
        if params.upper() == "Y":

            number = input("Enter the number of the field you want to change.\n")
            if number == "1":

                MAX_NUM_HOMOL = input("What should the maximum number of homologs be?\n")
                if MAX_NUM_HOMOL.isdigit() and int(MAX_NUM_HOMOL) > 0:

                    form['MAX_NUM_HOMOL'] = MAX_NUM_HOMOL

                else:

                    print("Wrong input")

            elif number == "2":

                MAX_REDUNDANCY = input("What should the maximum redundancy be?\n")
                if MAX_REDUNDANCY.isdigit() and int(MAX_REDUNDANCY) > 0:

                    if int(MAX_REDUNDANCY) <= 100:

                        vars['hit_redundancy'] = int(MAX_REDUNDANCY)

                    else:

                        vars['hit_redundancy'] = 99.99999999999

                else:

                    print("Wrong input")

            elif number == "3":

                print("The program to build the msa are:\n1. mafft.\n2. muscle.\n3. CLUSTALW.\n4. prank.")
                MSAprogram = input("Choose the number of the program.\n")
                if MSAprogram == "1":

                    form['MSAprogram'] = "MAFFT"

                elif MSAprogram == "2":

                    form['MSAprogram'] = "MUSCLE"

                elif MSAprogram == "3":

                    form['MSAprogram'] = "CLUSTALW"

                elif MSAprogram == "4":

                    form['MSAprogram'] = "PRANK"

                else:

                    print("Wrong input")

            elif number == "4":

                MIN_IDENTITY = input("What should the minimum identity be?\n")
                if MIN_IDENTITY.isdigit() and int(MIN_IDENTITY) > 0:

                    form['MIN_IDENTITY'] = int(MIN_IDENTITY)

                else:

                    print("Wrong input.")

            elif number == "5":

                print("We either sample from the list of homologs (sample) or take the homologs closest to the query (closest).")
                best_uniform = input("What the homolog selection method should be? (sample/closest):\n")
                if best_uniform == "closest":

                    form['best_uniform_sequences'] = "closest"
                    
                elif best_uniform == "sample":
                    
                    form['best_uniform_sequences'] = "sample"

                else:

                    print("Wrong input.")
                    
            elif number == "6":
                
                print("Sequences with higher e-value will be rejected.")
                form['E_VALUE'] = float(input("What should the e-value be?\n"))

            else:

                print("Wrong input.")


        elif params.upper() == "N":

            print()
            break

        else:

            print("Wrong input. Pease type 'Y' if you do want to change the parameters, and 'N' if you don't want to change them.")


def choose_mode(mode):
    
    if mode == "PDB":

        vars['running_mode'] = "_mode_pdb_no_msa"

    elif mode == "PDB_MSA":

        vars['running_mode'] = "_mode_pdb_msa"

    elif mode == "PDB_MSA_Tree":

        vars['running_mode'] = "_mode_pdb_msa_tree"

    elif mode == "MSA":

        vars['running_mode'] = "_mode_msa"

    elif mode == "MSA_Tree":

        vars['running_mode'] = "_mode_msa_tree"

    elif mode == "Sequence":

        vars['running_mode'] = "_mode_no_pdb_no_msa"

def print_instructions(pdb_with_grades, cif_pdb, pdb_with_grades_isd = False):
    
    print("To create pymol with consurf:")
    if pdb_with_grades_isd:
        
        create_download_link(pdb_with_grades_isd, "Download the modified %s file showing insufficient data" %cif_pdb)
        print("or")
        create_download_link(pdb_with_grades, "the %s file hiding insufficient data" %cif_pdb)

    else:
        
        create_download_link(pdb_with_grades, "Download the modified %s file" %cif_pdb)
        
    print("which contains ConSurf's color grades.\nDownload the coloring script")
    create_download_link(vars['pymol'], "regular version")
    print("or")
    create_download_link(vars['pymol_CBS'], "color blind")
    print("1) Start the PyMOL program.\n2) Drag the %s file to the pymol window.\n3) Drag the pymol coloring acript to the window.\n" %cif_pdb)


def print_legend(cbs):
    
    text = """<style>
.scaleRowStrecher .scaleRowStrecherInner{display: flex;}
.scaleRowStrecher .scaleRowStrecherInner .scaleColorRect{width: 50px;text-align: center;line-height: 32px;font-weight: 500;font-size: 20px;}
.scaleRowStrecher .scaleRowStrecherInner .scaleColorRect.white{color: #ffffff;}
.scaleRowStrecher .scaleRowStrecherInner .scaleColorRect span{width: 100%;}
.scaleRowStrecher .label{display: flex;padding-right: 0;}
.scaleRowStrecher .label div{width: 33.33%;text-align: center;font-weight: 500;font-size: 18px}
.scaleRowStrecher .label .leftLabel{text-align: left;}
.scaleRowStrecher .label .rightLabel{text-align: right;}
#consrvScaleDiv a.download{margin-top: 20px; color: #400080;text-decoration: underline;transition: 0.5s all ease;-webkit-transition: 0.5s all ease;-o-transition: 0.5s all ease;-moz-transition: 0.5s all ease;}
.scaleRowStrecher .insu_data{font-weight: 500;font-size: 18px;margin-top: 20px;}
.scaleRowStrecher .insu_data span{display: inline-block;vertical-align: middle;height: 25px;width: 50px;background-color: #f8f499;margin-right: 10px;}
.scaleRowStrecher {float: left; margin-top: 20px;margin-bottom: 20px;}
.scaleRowStrecher .scaleRowStrecherInner{display: flex;}
.scaleRowStrecher .scaleRowStrecherInner .scaleColorRect.white{color: #ffffff;}
.scaleRowStrecher .scaleRowStrecherInner .scaleColorRect span{width: 100%;}
.scaleRowStrecher .label{display: flex;}
.scaleRowStrecher .label div{width: 33.33%;text-align: center;font-weight: 500;font-size: 18px}
.scaleRowStrecher .label .leftLabel{text-align: left;}
.scaleRowStrecher .label .rightLabel{text-align: right;}
#consrvScaleDiv a.download{margin-top: 20px; color: #400080;text-decoration: underline;transition: 0.5s all ease;-webkit-transition: 0.5s all ease;-o-transition: 0.5s all ease;-moz-transition: 0.5s all ease;}
.scaleRowStrecher .insu_data{font-weight: 500;font-size: 18px;margin-top: 20px;}
.scaleRowStrecher .insu_data span{display: inline-block;vertical-align: middle;height: 25px;width: 50px;background-color: #f8f499;margin-right: 10px;}
.result_c1{background-color: #0a7d82;color: #ffffff;}
.result_c2{background-color: #44afbf;}
.result_c3{background-color: #a5dce6;}
.result_c4{background-color: #d7f0f0;}
.result_c5{background-color: #ffffff;}
.result_c6{background-color: #faebf5;}
.result_c7{background-color: #fac8dc;}
.result_c8{background-color: #f07daa;}
.result_c9{background-color: #a0285f;color: #ffffff;}
.result_c1_CBS{background-color: #0f5a23;color: #ffffff;}
.result_c2_CBS{background-color: #5aaf5f;}
.result_c3_CBS{background-color: #a5dca0;}
.result_c4_CBS{background-color: #d7f0d2;}
.result_c5_CBS{background-color: #ffffff;}
.result_c6_CBS{background-color: #e6d2e6;}
.result_c7_CBS{background-color: #c3a5cd;}
.result_c8_CBS{background-color: #9b6eaa;}
.result_c9_CBS{background-color: #782882;color: #ffffff;}  
    </style>
    """
    if cbs:
        
        text += """
    <div>
        <div class="scaleRowStrecher">
            <div class="scaleRowStrecherInner">
            <div class="scaleColorRect result_c1_CBS"><span>1</span></div>
            <div class="scaleColorRect result_c2_CBS"><span>2</span></div>
            <div class="scaleColorRect result_c3_CBS"><span>3</span></div>
            <div class="scaleColorRect result_c4_CBS"><span>4</span></div>
            <div class="scaleColorRect result_c5_CBS"><span>5</span></div>
            <div class="scaleColorRect result_c6_CBS"><span>6</span></div>
            <div class="scaleColorRect result_c7_CBS"><span>7</span></div>
            <div class="scaleColorRect result_c8_CBS"><span>8</span></div>
            <div class="scaleColorRect result_c9_CBS"><span>9</span></div>
        </div>
        <div class="label">
            <div class="leftLabel">Variable</div> 
            <div class="centerLabel">Average</div>
            <div class="rightLabel">Conserved</div>
        </div>
        <div class="insu_data">
            <span></span>Insufficient Data
        </div>
    </div>
            """
        
    else:
        
        text += """
    <div>
        <div class="scaleRowStrecher">
            <div class="scaleRowStrecherInner">
            <div class="scaleColorRect result_c1"><span>1</span></div>
            <div class="scaleColorRect result_c2"><span>2</span></div>
            <div class="scaleColorRect result_c3"><span>3</span></div>
            <div class="scaleColorRect result_c4"><span>4</span></div>
            <div class="scaleColorRect result_c5"><span>5</span></div>
            <div class="scaleColorRect result_c6"><span>6</span></div>
            <div class="scaleColorRect result_c7"><span>7</span></div>
            <div class="scaleColorRect result_c8"><span>8</span></div>
            <div class="scaleColorRect result_c9"><span>9</span></div>
        </div>
        <div class="label">
            <div class="leftLabel">Variable</div> 
            <div class="centerLabel">Average</div>
            <div class="rightLabel">Conserved</div>
        </div>
        <div class="insu_data">
            <span></span>Insufficient Data
        </div>
    </div>
            """
        
    display(HTML(text))
    
    
def print_msa_colors_FASTA_clustalwLike(grades, MSA, SeqName, cbs):

    # Print the results: the colored sequence of the msa acording to the query sequence
    
    Header = "ConSurf Color-Coded MSA for Job:%s Date:%s" %(vars['job_name'], vars['date'])
    
    blockSize = 50

    if form['DNA_AA'] == "AA":

        unknownChar = 'X'

    else:

        unknownChar = 'N'

    if cbs:
        
        Out_file = vars['job_name'] + "_colored_MSA_CBS.html"
        
    else:
        
        Out_file = vars['job_name'] + "_colored_MSA.html"
        
    try:

        OUT = open(Out_file, 'w')

    except:

        exit_on_error("sys_error", "could not open the file " + Out_file + " for writing.")

    OUT.write("<!DOCTYPE html>\n")
    OUT.write("<html lang=\"en\">\n")
    OUT.write("<head>\n")
    OUT.write("<meta charset=\"utf-8\">\n")
    OUT.write("<meta http-equiv=\"X-UA-Compatible\" content=\"IE=edge\">\n")
    OUT.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    #OUT.write("<link rel=\"stylesheet\" type=\"text/css\" href=\"%s\">\n" %CSS_File)
    OUT.write("<style>")
    if cbs:
        
        OUT.write("""table {
	/*	table-layout: auto; */
		table-layout: fixed;
        margin-left: 0em;
        margin-right: 0em;
        padding:1em 1em 1em 1em;
	    margin:1em 1em 1em 1em;
        border-collapse: collapse;
      }
td {
        font-family: "Courier New", Courier, monospace;
        font-size:1em;
        font-weight: bold;
        text-align: center;
        overflow:hidden;
     /*   white-space:nowrap; */
		 white-space:pre;
/*	 	padding:0.1em 0.1em 0.1em 0.1em; */
/*	 	margin:0.5em 0.5em 0.5em 0.5em;*/
		width: 1em;
      }
td.Seq_Name{
        text-align: left;
        width: 15em;
        padding-right:1em;
      }
td.Score9{
        color: #FFFFFF;
        background: #782882;
  	}
td.Score8{
        background: #9B6EAA;
        }
td.Score7{
        background: #C3A5CD;
        }
td.Score6{
        background: #E6D2E6;
        }
td.Score5{
        background: #FFFFFF;
        }
td.Score4{
        background: #D7F0D2;
	}
td.Score3{
        background: #A5DCA0;
	}
td.Score2{
        background: #5AAF5F;
        }
td.Score1{
		color: #FFFFFF;
        background: #0F5A23;
        }
td.ScoreNaN{
	    background: #808080;
	}

/* ISD COLORES */
td.Score_ISD{
        background: #FFFF96;
  	}

td.white{
        background: #FFFFFF;
        }
/* GRAPH STYLING */

.barGraph {
/*	background: url(images/horizontal_grid_line_50_pixel.png) bottom left;*/
/*	border-bottom: 3px solid #333;*/
/*	font: 9px Helvetica, Geneva, sans-serif; */
	height: 20em;
	margin: 0em 0em;
	padding: 0em;
	position: relative;
	}

.barGraph p {
	font-size:1em;
	}

.barGraph li {
/*	background: #666 url(images/bar_50_percent_highlight.png) repeat-y top right;*/
/*	border: 0.2em solid #555;*/
	border-bottom: none;
	bottom: 0em;
	color: #FFF;
	margin: 0em;
	padding: 0em 0em 0em 0em;
	position: absolute;
	list-style: none;
	text-align: center;
	width: 1em;
	}

.barGraph li.p1{ background-color:#666666; }
.barGraph li:hover {font-weight:bold;}

                  """)
        
    else:
        
        OUT.write("""table {
	/*	table-layout: auto; */
		table-layout: fixed;
        margin-left: 0em;
        margin-right: 0em;
        padding:1em 1em 1em 1em;
	    margin:1em 1em 1em 1em;
        border-collapse: collapse;
      }
td {
        font-family: "Courier New", Courier, monospace;
        font-size:1em;
        font-weight: bold;
        text-align: center;
        overflow:hidden;
     /*   white-space:nowrap; */
		 white-space:pre;
/*	 	padding:0.1em 0.1em 0.1em 0.1em; */
/*	 	margin:0.5em 0.5em 0.5em 0.5em;*/
		width: 1em;
      }
td.Seq_Name{
        text-align: left;
        width: 15em;
        padding-right:1em;
      }
td.Score9{
        color: #FFFFFF;
        background: #A0285F;
  	}
td.Score8{
        background: #F07DAA;
        }
td.Score7{
        background: #FAC8DC;
        }
td.Score6{
        background: #FAEBF5;
        }
td.Score5{
        background: #FFFFFF;
        }
td.Score4{
        background: #D7F0F0;
	}
td.Score3{
        background: #A5DCE6;
	}
td.Score2{
        background: #4BAFBE;
        }
td.Score1{
		color: #FFFFFF;
        background: #0A7D82;
        }
td.ScoreNaN{
	    background: #808080;
	}

/* ISD COLORES */
td.Score_ISD{
        background: #FFFF96;
  	}

td.white{
        background: #FFFFFF;
        }
/* GRAPH STYLING */

.barGraph {
/*	background: url(images/horizontal_grid_line_50_pixel.png) bottom left;*/
/*	border-bottom: 3px solid #333;*/
/*	font: 9px Helvetica, Geneva, sans-serif; */
	height: 20em;
	margin: 0em 0em;
	padding: 0em;
	position: relative;
	}

.barGraph p {
	font-size:1em;
	}

.barGraph li {
/*	background: #666 url(images/bar_50_percent_highlight.png) repeat-y top right;*/
/*	border: 0.2em solid #555;*/
	border-bottom: none;
	bottom: 0em;
	color: #FFF;
	margin: 0em;
	padding: 0em 0em 0em 0em;
	position: absolute;
	list-style: none;
	text-align: center;
	width: 1em;
	}

.barGraph li.p1{ background-color:#666666; }
.barGraph li:hover {font-weight:bold;}

                  """)
                  
    OUT.write("</style>")
    OUT.write("<title>%s</title>\n" %Header)
    OUT.write("</head>\n")
    OUT.write("<H1 align=center><u>%s</u></H1>\n\n" %Header) # MSA color-coded by GAIN probability

    [MSA_Hash, Seq_Names_In_Order] = ReadMSA(MSA)
    NumOfBlocks = int(len(MSA_Hash[SeqName]) / blockSize) + 1

    seq_pos = 0
    ind = 0
    for block in range(0, NumOfBlocks):

        MSA_Pos = blockSize * block
        OUT.write("<table>\n")
        seqNum = 0
        seq_pos += ind
        for name in Seq_Names_In_Order:

            seqNum += 1
            OUT.write("<tr>\n")
            if name == SeqName:

                OUT.write("<td class=\"Seq_Name\"><u><b>%d %s</u></b></td>\n" %(seqNum, name))

            else:

                OUT.write("<td class=\"Seq_Name\">%d %s</td>\n" %(seqNum, name))

            ind = 0
            for (pos, char) in list(zip(MSA_Hash[SeqName], MSA_Hash[name]))[MSA_Pos : (block + 1) * blockSize]:

                if pos != "-" and pos.upper() != unknownChar:

                    ScoreClass = "Score" + str(grades[seq_pos + ind]['COLOR'])
                    if grades[seq_pos + ind]['ISD'] == 1:

                        ScoreClass = "Score_ISD"

                    OUT.write("<td class=\"%s\">%s</td>" %(ScoreClass, char))
                    ind += 1

                else:

                    OUT.write("<td class=\"white\">%s</td>" %char)

            OUT.write("</tr>\n")

        OUT.write("</table><br><br>\n")

    # print the color scale
    OUT.write("<table style = 'table-layout: auto;margin-left: 0em;margin-right: 0em;padding:1px 1px 1px 1px; margin:1px 1px 1px 1px; border-collapse: collapse;' border=0 cols=1 width=310>\n<tr><td align=center>\n<font face='Courier New' color='black' size=+1><center>\n<tr>")
    for i in range(1,10):

        OUT.write("<td class=\"%s\">%d</td>\n" %("Score" + str(i), i))

    OUT.write("</tr></font></center>\n<center><table style = 'table-layout: auto;margin-left:0em;margin-right: 0em;padding:1px 1px 1px 1px; margin:1px 1px 1px 1px; border-collapse: collapse;' border=0 cols=3 width=310>\n<tr>\n<td align=left><td align=left><b>Variable</b></td><td></td><td align=center><b>Average</b></td><td></td>\n<td align=right><b>Conserved</b></td>\n</tr><tr></tr><tr></tr>\n</table></center>\n")
    OUT.write("<table><tr><b><td class=\"Score_ISD\">X</td><td class=\"white\"> - Insufficient data - the calculation for this site was performed on less than 10% of the sequences.</b><br></td></tr></table>\n")
    OUT.write("</body>\n</table>\n")
    OUT.close()
    
    create_download_link(Out_file, "Download colored MSA")
    
    vars['zip_list'].append(Out_file)

def ReadMSA(msa):

    Seq_Names_In_Order = [] # array to hold sequences names in order
    MSA_Hash = {} # hash to hold sequnces
    Seq = ""
    Seq_Name = ""

    try:

        MSA = open(msa, 'r')

    except:

        exit_on_error("sys_error", "ReadMSA: Can't read the MSA: " + msa)

    line = MSA.readline()
    while line != "":

        line = line.rstrip()
        match = re.match(r'^>(.*)', line)
        if match:

            if Seq != "":

                MSA_Hash[Seq_Name] = Seq
                Seq = ""
                Seq_Name = ""

            Seq_Name = match.group(1)
            Seq_Names_In_Order.append(Seq_Name)

        else:

            Seq += line

        line = MSA.readline()

    MSA.close()
    MSA_Hash[Seq_Name] = Seq # last sequence

    return(MSA_Hash, Seq_Names_In_Order)


def create_download_link(file, text):
    callback_id = f"download_{uuid.uuid4().hex}"

    def download_file():
        files.download(vars['working_dir'] + file)

    output.register_callback(callback_id, download_file)

    display(HTML(f'''
        <a href="#" onclick="google.colab.kernel.invokeFunction('{callback_id}', [], {{}}); return false;">
           {text}
        </a>
    '''))
    

def consurf_HTML_Output(cbs):

    # Print the results: the colored sequence and the B/E information
    if cbs:

        consurf_html_colors = vars['color_array_CBS']
    else:

        consurf_html_colors = vars['color_array']

    COLORS ="<html>\n<title>ConSurf Results</title>\n"
    COLORS += "<head>\n<style>\nb { float: left;}\n</style>\n</head>\n"
    COLORS += "<body bgcolor='white'>\n"
    COLORS += "\n<table border=0 width=100%>\n"
    COLORS += "<tr><td>\n"

    # print the colored sequence

    count = 1
    letter_str = ""

    number_of_pos = len(vars['gradesPE_Output'])
    for elem in vars['gradesPE_Output']:

        # print the counter above the beginning of each 10 characters
        if count % 50 == 1:

            count_num = count
            while count_num < count + 50:

                if count_num <= number_of_pos:

                    space_num = 11 - len(str(count_num))
                    spaces = ""
                    for i in range(0, space_num):

                        spaces += "&nbsp;"

                    COLORS += "<font face='Courier New' color='black' size=+1>" + str(count_num) + spaces + "</font>"

                count_num += 10

            COLORS += "<br>\n"

        # print the colored letters and 'e' for the exposed residues

        # after 50 characters - print newline
        if count % 50 == 0 or count == number_of_pos:

            if elem['ISD'] == 1: # INSUFFICIENT DATA

                letter_str += "<b><font face='Courier New' color='black' size=+1><span style='background: %s;'>%s</span></font></b><br>" %(consurf_html_colors['ISD'], elem['SEQ'])

            elif elem['COLOR'] == 9 or elem['COLOR'] == 1: # MOST OR LEAST CONSERVED

                letter_str += "<b><font face='Courier New' color='white' size=+1><span style='background: %s;'>%s</span></font></b><br>\n" %(consurf_html_colors[elem['COLOR']], elem['SEQ'])

            else:

                letter_str += "<b><font face='Courier New' color='black' size=+1><span style='background: %s;'>%s</span></font></b><br>\n" %(consurf_html_colors[elem['COLOR']], elem['SEQ'])

            COLORS += letter_str
            COLORS += "</td></tr>\n"
            COLORS += "<tr><td>\n"

            letter_str = ""

        elif count % 10 == 1 and count % 50 != 1: # after 10 characters - print a space ('&nbsp;')

            letter_str += "<b><font face='Courier New' color='black' size=+1>&nbsp;</font></b>"

            if elem['ISD'] == 1:

                letter_str += "<b><font face='Courier New' color='black' size=+1><span style='background: %s;'>%s</span> </font></b>\n" %(consurf_html_colors['ISD'], elem['SEQ'])

            elif elem['COLOR'] == 9 or elem['COLOR'] == 1: # MOST OR LEAST CONSERVED

                letter_str += "<b><font face='Courier New' color='white' size=+1><span style='background: %s;'>%s</span> </font></b>\n" %(consurf_html_colors[elem['COLOR']], elem['SEQ'])

            else:

                letter_str += "<b><font face='Courier New' color='black' size=+1><span style='background: %s;'>%s</span> </font></b>\n" %(consurf_html_colors[elem['COLOR']], elem['SEQ'])

        else:

            if elem['ISD'] == 1:

                letter_str += "<b><font face='Courier New' color='black' size=+1><span style='background: %s;'>%s</span> </font></b>\n" %(consurf_html_colors['ISD'], elem['SEQ'])

            elif elem['COLOR'] == 9 or elem['COLOR'] == 1: # MOST OR LEAST CONSERVED

                letter_str += "<b><font face='Courier New' color='white' size=+1><span style='background: %s;'>%s</span> </font></b>\n" %(consurf_html_colors[elem['COLOR']], elem['SEQ'])

            else:

                letter_str += "<b><font face='Courier New' color='black' size=+1><span style='background: %s;'>%s</span> </font></b>\n" %(consurf_html_colors[elem['COLOR']], elem['SEQ'])

        count += 1

    COLORS += "</td></tr>\n</table><br>\n"
    COLORS += "</body>\n</html>\n"
    display(HTML(COLORS))

def no_model_view(cbs = False):
    
    if cbs:
        
        print("------------------------------------------------------------------------------------------------------------------------")
        print("Color Blind View")
        # ConSurf color palette
        pdf_file = vars['Colored_Seq_CBS_PDF']
        
    else:
        
        print("------------------------------------------------------------------------------------------------------------------------")
        print("Regular View")
        # ConSurf color palette 
        pdf_file = vars['Colored_Seq_PDF']


    print_legend(cbs)
    consurf_HTML_Output(cbs)
    create_download_link(pdf_file, "Download colored sequence pdf file")
    print_msa_colors_FASTA_clustalwLike(vars['gradesPE_Output'], vars['msa_fasta'], vars['msa_SEQNAME'], cbs)
    
    if not cbs:
        
        no_model_view(True)
        print("------------------------------------------------------------------------------------------------------------------------")
        
    

def show_py3dmol(file, file_type, cbs = False):
    
    # Read your ConSurf-colored PDB file
    with open(file) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, file_type)
    view.setStyle({'cartoon': {}})

    if cbs:
        
        print("------------------------------------------------------------------------------------------------------------------------")
        print("Color Blind View")
        # ConSurf color palette
        colors = vars['color_array_CBS']
        pdf_file = vars['Colored_Seq_CBS_PDF']
        
    else:
        
        print("------------------------------------------------------------------------------------------------------------------------")
        print("Regular View")
        # ConSurf color palette 
        colors = vars['color_array']
        pdf_file = vars['Colored_Seq_PDF']

    # Apply each color to the appropriate B-factor range
    for i in range(1, 10):
        
        view.setStyle({'b': i}, {'cartoon': {'color': colors[i]}})

    # Color residues with B = 10 (insufficient data) in yellow
    view.setStyle({'b': 10}, {'cartoon': {'color': 'yellow'}})

    view.zoomTo()
    view.show()
    print_legend(cbs)
    consurf_HTML_Output(cbs)
    create_download_link(pdf_file, "Download colored sequence pdf file")
    print_msa_colors_FASTA_clustalwLike(vars['gradesPE_Output'], vars['msa_fasta'], vars['msa_SEQNAME'], cbs)
    
    if not cbs:
        
        show_py3dmol(file, file_type, True)
        print("------------------------------------------------------------------------------------------------------------------------")
        
    
def print_selected(arr_ref, print_for_pipe):

    total_text = ""
    string = ""
    if print_for_pipe == "yes":

        string = "! select "

    else:

        string = "select "

    total_length = len(string)

    if len(arr_ref) > 0:

        for aa in arr_ref:

            aa = aa.replace(":", "")
            total_length += len(aa)
            if total_length > 80:

                if re.search(r', $', string):

                    string = string[:-2]

                total_text += string + "\n"
                if print_for_pipe == "yes":

                    string = "! select selected or %s, " %aa

                else:

                    string = "select selected or %s, " %aa

                total_length = len(string)

            else:

                string += aa + ", "
                total_length += 2

    else:

        total_text += string + "none"


    if re.search(r', $', string):

        string = string[:-2]
        total_text += string

    return total_text

def print_rasmol(job_name, out_file, isd, ref_colors_array, chain, cbs):

    # print out new format of rasmol


    consurf_rasmol_colors = ["", "[16,200,209]", "[140,255,255]", "[215,255,255]", "[234,255,255]", "[255,255,255]", "[252,237,244]", "[250,201,222]", "[240,125,171]", "[160,37,96]", "[255,255,150]"]
    consurf_rasmol_colors_CBS = ["", "[27,120,55]", "[90,174,97]", "[166,219,160]", "[217,240,211]", "[247,247,247]", "[231,212,232]", "[194,165,207]", "[153,112,171]", "[118,42,131]", "[255,255,150]"]

    try:

        OUT = open(out_file, 'w')

    except:

        exit_on_error('sys_error', "print_rasmol : Could not open the file " + out_file + " for writing.")

    OUT.write("ConSurfDB.tau.ac.il   %s   %s\nAPD N.NN\n" %(vars['date'], job_name))
    OUT.write("select all\ncolor [200,200,200]\n\n")
    
    i = len(ref_colors_array) - 1
    while i > 0:

        if i == 10 and not isd:

            i -= 1
            continue

        if len(ref_colors_array[i]) > 0:

            OUT.write(print_selected(ref_colors_array[i], "no"))
            OUT.write("\nselect selected and :%s\n" %chain)
            if cbs:

                OUT.write("color %s\nspacefill\n" %consurf_rasmol_colors_CBS[i])

            else:

                OUT.write("color %s\nspacefill\n" %consurf_rasmol_colors[i])

            OUT.write("define CON%d selected\n\n" %i)

        i -= 1

    OUT.close()

def create_rasmol(job_name, chain, ref_colors_array, ref_colors_array_isd):
    
    RasMol_file = job_name + "_jmol_consurf_colors.spt"
    RasMol_file_isd = job_name + "_jmol_consurf_colors_isd.spt"
    RasMol_file_CBS = job_name + "_jmol_consurf_colors_CBS.spt"
    RasMol_file_CBS_isd = job_name + "_jmol_consurf_colors_CBS_isd.spt"
	
    if chain == "NONE":
        
        chain = " "
        
    print_rasmol(job_name, RasMol_file, False, ref_colors_array, chain, False)
    print_rasmol(job_name, RasMol_file_CBS, False, ref_colors_array, chain, True)

    if len(ref_colors_array_isd[10]) > 0: # there is isd
        
        print_rasmol(job_name, RasMol_file_isd, True, ref_colors_array_isd, chain, False)
        print_rasmol(job_name, RasMol_file_CBS_isd, True, ref_colors_array_isd, chain, True)

def read_number(file):
    
    # reads numbers from a numbers file without storing the whole file in a string
    number = ""
    char = file.read(1)
    while char and char.isspace(): # first we skip white spaces
        
        char = file.read(1)
        
    while char and not char.isspace(): # we read the number
        
        number += char
        char = file.read(1)
        
    if number != "":
        
        number = float(number)
        
    return number


def print_pnet_file(num_pos):
    
    # for each position in the msa we print percentage per positions for 6 positions before and after
    window = 6
    pad = []
    pnet_file = open("p.net", 'w')
    acids = ["V", "L", "I", "M", "F", "W", "Y", "G", "A", "P", "S", "T", "C", "H", "R", "K", "Q", "E", "N", "D"]

    for i in range(window):
        
        pad.append({})
        for acid in acids:
            
            pad[-1][acid] = 0
            
    padded_percentage_per_pos = pad + vars['percentage_per_pos'] + pad
    for i in range(window, num_pos + window):
        
        j = -window
        while j < window + 1:
            
            for acid  in acids:
                
                if acid in padded_percentage_per_pos[i + j]:
                
                    pnet_file.write(str(int(padded_percentage_per_pos[i + j][acid])) + " ")
                    
                else:
                    
                    pnet_file.write("0 ")
                    
            j += 1

        pnet_file.write("\n")
            
    pnet_file.close() 

def reveal_buried_exposed(buried_exposed):
    
    # this function reveals the buried and exposed atoms.
    # We look at 6 positions before and 6 position after the current position, 260 numbers in total. 
    # We multiply each number by a unique weight and then we sum the numbers. 
    # The results to put in the formula 1 / (1 + e^(-x)). 
    # We do this twenty times with different weights. Now we have 20 numbers. 
    # We again multiply the numbers by different weights and sum them up and put them in the formula  1 / (1 + e^(-x)). 
    # We do this twice, again with different weights. Now we have two numbers. 
    # If number 1 > number2 we say the position is exposed, if not it’s buried.
    
    num_pos = len(vars['percentage_per_pos'])
    print_pnet_file(num_pos)
    G_nl = 2
    G_N = [260, 20, 2]
    
    weights_file = open(vars['git_dir'] + "WEIGHTS.BIN", 'r')
    pnet_file = open("p.net", 'r')
    

    for i in range(num_pos):
        
        G_o = [[], [],[]]
        for j in range(G_N[0]):
            
            G_o[0].append(read_number(pnet_file) / 100.0)
            
        for s in range(1, G_nl + 1):
            
            for k in range(G_N[s]):
                
                weight = read_number(weights_file)
                for j in range(G_N[s - 1]):
                    
                    weight += read_number(weights_file) * G_o[s - 1][j]
                    
                G_o[s].append(1 / (1 + math.exp(-weight)))


        weights_file.seek(0) # each time we use the same weights, so we move the pointer to the start
        if G_o[G_nl][0] < G_o[G_nl][1]:
         
            buried_exposed.append("e")
        
        else:
        
            buried_exposed.append("b")
            
    weights_file.close()
    pnet_file.close()
    
    return buried_exposed


class pdbParser:

    def __init__(self):

        self.SEQRES = ""
        self.ATOM = ""
        self.ATOM_withoutX = {}
        self.type = ""
        self.MODIFIED_COUNT = 0
        self.MODIFIED_LIST = ""
        self.positions = {}
        self.max_res_details = 0
        self.num_known_atoms = 0
        self.num_known_seqs = 0




    #def read(self, file, query_chain, DNA_AA, atom_position_filename):
    def read(self, file, query_chain, DNA_AA):

        #conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "a", "T" : "t", "C" : "c", "G" : "g", "U" : "u", "I" : "i", "DA" : "a", "DT" : "t", "DC" : "c", "DG" : "g", "DU" : "u", "DI" : "i", "5CM" : "c", "5MU" : "t", "N" : "n"}
        #conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "a", "T" : "t", "C" : "c", "G" : "g", "U" : "u", "I" : "i", "DA" : "a", "DT" : "t", "DC" : "c", "DG" : "g", "DU" : "u", "DI" : "i", "5CM" : "c", "N" : "n"}
        conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "A", "T" : "T", "C" : "C", "G" : "G", "U" : "U", "I" : "I", "DA" : "A", "DT" : "T", "DC" : "C", "DG" : "G", "DU" : "U", "DI" : "I", "5CM" : "C", "N" : "N"}
        modified_residues = {"MSE" : "MET", "MLY" : "LYS", "HYP" : "PRO", "CME" : "CYS", "CGU" : "GLU", "SEP" : "SER", "KCX" : "LYS", "MLE" : "LEU", "TPO" : "THR", "CSO" : "CYS", "PTR" : "TYR", "DLE" : "LEU", "LLP" : "LYS", "DVA" : "VAL", "TYS" : "TYR", "AIB" : "ALA", "OCS" : "CYS", "NLE" : "LEU", "MVA" : "VAL", "SEC" : "CYS", "PYL" : "LYS"}
        localMODRES = {}
        FIRST = [] # first residue in chain
        fas_pos = 0
        chain = "" # current chain
        ENDS = [] # end of chain reached and remaining HETATM should be skipped
        last_residue_number = ""

        if DNA_AA == "Nuc":

            UnknownChar = "N"

        else:

            UnknownChar = "X"
                        
        # open file to read MODRES
        try:

            PDBFILE = open(file, 'r')

        except:

            return 0

        try:

            MODRES_FILE = open(file + ".MODRES", 'w')

        except:

            return 0

        # read the MODRES
        line = PDBFILE.readline()
        while line != "" and not re.match(r'^ATOM', line):

            if re.match(r'^MODRES', line):

                MODRES = line[12:15].strip() # strip spaces to support NUC
                CHAIN = line[16:17]
                if CHAIN == " ":
                    
                    CHAIN = "NONE"
                    
                # we only look at the query chain
                if CHAIN != query_chain:
                    
                    line = PDBFILE.readline()
                    continue
                
                RES = line[24:27].strip() # strip spaces to support NUC

                if not MODRES in localMODRES:

                    localMODRES[MODRES] = RES
                    MODRES_FILE.write(MODRES + "\t" + RES + "\n")

                elif localMODRES[MODRES] != RES:

                    localMODRES[MODRES] = "" # two different values to the same residue

            line = PDBFILE.readline()


        MODRES_FILE.close()
        PDBFILE.close()
        
        # reopen file to read all the file
        try:

            PDBFILE = open(file, 'r')

        except:

            return 0

        line = PDBFILE.readline()
        while line != "":

            line = line.strip()

            if re.search(r'^SEQRES', line): # SEQRES record

                chain_seqres = line[11:12] # get chain id
                if chain_seqres == " ":

                    chain_seqres = "NONE"

                # we skip the chain if it is not the query
                if query_chain != chain_seqres:
                    
                    line = PDBFILE.readline()
                    continue

                # convert to one letter format
                for acid in line[19:70].split():

                    # regular conversion
                    if acid in conversion_table:

                        # add to chain
                        self.SEQRES += conversion_table[acid]
                        self.num_known_seqs += 1

                    # modified residue
                    else:

                        # count this modified residue
                        self.MODIFIED_COUNT += 1

                        # check if residue is identified
                        if acid in modified_residues and modified_residues[acid] in conversion_table:

                            self.SEQRES += conversion_table[modified_residues[acid]]
                            self.num_known_seqs += 1

                            # add to modified residue list
                            if not acid + " > " in self.MODIFIED_LIST:

                                self.MODIFIED_LIST += acid + " > " + conversion_table[modified_residues[acid]] + "\n"

                        elif acid in localMODRES and localMODRES[acid] != "" and localMODRES[acid] in conversion_table:

                            self.SEQRES += conversion_table[localMODRES[acid]]
                            self.num_known_seqs += 1

                            # add to modified residue list
                            if not acid + " > " in self.MODIFIED_LIST:

                                self.MODIFIED_LIST += acid + " > " + conversion_table[localMODRES[acid]] + "\n"

                        else:

                            # set residue name to X or N
                            self.SEQRES += UnknownChar

                            # add message to front of modified residue list
                            modified_changed_to_X_or_N_msg = "Modified residue(s) in this chain were converted to the one letter representation '" + UnknownChar + "'\n"

                            if not "Modified residue" in self.MODIFIED_LIST:

                                self.MODIFIED_LIST = modified_changed_to_X_or_N_msg + self.MODIFIED_LIST

            elif re.search(r'^ATOM', line):

                # extract atom data
                res = line[17:20].strip() # for DNA files there is only one or two letter code
                chain = line[21:22]
                pos = line[22:27].strip()

                if chain == " ":

                    chain = "NONE"

                # find modified residue
                if res in modified_residues:
                    
                    mod_res = modified_residues[res]
                    
                elif res in localMODRES and localMODRES[res] != "" and localMODRES[res] in conversion_table:
                    
                    mod_res = localMODRES[res]

                else:
                    
                    mod_res = res
                  
                # convert residue to one letter
                if mod_res in conversion_table:
                    
                    oneLetter = conversion_table[mod_res]
                    
                else:
                    
                    oneLetter = UnknownChar
                  
                # check if we reached a new residue
                if not chain in FIRST:
                    
                    FIRST.append(chain)
                    last_pos = pos
                    self.ATOM_withoutX[chain] = oneLetter
                    
                elif pos != last_pos:
                    
                    last_pos = pos
                    self.ATOM_withoutX[chain] += oneLetter

                else:

                    line = PDBFILE.readline()
                    continue 
                     
                # if the chain is not the query we only extract the sequence 
                if query_chain != chain:
                         
                    line = PDBFILE.readline()
                    continue 
                  
                self.num_known_atoms += 1
                
                # writing atom position file
                fas_pos += 1
                res_details = "%s:%s:%s" %(res, pos, chain)
                self.positions[fas_pos] = res_details
                if len(res_details) > self.max_res_details:
                    
                    self.max_res_details = len(res_details)
                    
                #CORR.write("%s\t%d\t%s\n" %(res, fas_pos, pos))
                        
                #residue_number = int(line[22:26].strip())
                        
                # check type 
                if self.type == "":

                    if len(mod_res) < 3:

                        self.type = "Nuc"

                    else:

                        self.type = "AA"
                """    
                if FIRST[chain]:
                    
                    FIRST[chain] = False

                elif last_residue_number < residue_number:
                    
                    while residue_number != last_residue_number + 1: # For Disorder regions
                    
                        self.ATOM += UnknownChar
                        last_residue_number += 1
                              
                self.ATOM += oneLetter
                last_residue_number = residue_number
                """
            elif re.search(r'^HETATM', line):

                # extract hetatm data
                res = line[17:20].strip() # for DNA files there is only one or two letter code
                chain = line[21:22]
                pos = line[22:27].strip()

                if chain == " ":

                    chain = "NONE"
                    
                if chain in ENDS:
                         
                    line = PDBFILE.readline()
                    continue 
                    
                # find modified residue
                if res in modified_residues:
                    
                    mod_res = modified_residues[res]
                    
                elif res in localMODRES and localMODRES[res] != "" and localMODRES[res] in conversion_table:
                    
                    mod_res = localMODRES[res]

                else:
                    
                    mod_res = res
                  
                # convert residue to one letter
                if mod_res in conversion_table:
                    
                    oneLetter = conversion_table[mod_res]
                    
                else:
                    
                    oneLetter = UnknownChar
                  
                # check if we reached a new residue
                if not chain in FIRST:
                    
                    FIRST.append(chain)
                    last_pos = pos
                    self.ATOM_withoutX[chain] = oneLetter
                    
                elif pos != last_pos:
                    
                    last_pos = pos
                    self.ATOM_withoutX[chain] += oneLetter

                else:

                    line = PDBFILE.readline()
                    continue                      
                
                # if the chain is not the query we only extract the sequence 
                if query_chain != chain:
                         
                    line = PDBFILE.readline()
                    continue 
                
                self.num_known_atoms += 1
                
                # writing atom position file
                fas_pos += 1
                res_details = "%s:%s:%s" %(res, pos, chain)
                self.positions[fas_pos] = res_details
                if len(res_details) > self.max_res_details:
                    
                    self.max_res_details = len(res_details)
                    
                """                        
                residue_number = int(line[22:26].strip())
                
                if FIRST[chain]:
                    
                    last_residue_number = residue_number
                    FIRST[chain] = False

                elif last_residue_number < residue_number:
                    
                    while residue_number != last_residue_number + 1: # For Disorder regions
                    
                        self.ATOM += UnknownChar
                        last_residue_number += 1
                              
                self.ATOM += oneLetter
                last_residue_number = residue_number   
                """
            elif re.search(r'^TER', line):
                
                if not chain in ENDS:
                    
                    ENDS.append(chain)
                
            line = PDBFILE.readline()

        PDBFILE.close()
        #CORR.close()
        return 1

    def get_num_known_atoms(self):
        
        return self.num_known_atoms

    def get_num_known_seqs(self):
        
        return self.num_known_seqs

    def get_max_res_details(self):
        
        return self.max_res_details

    def get_positions(self):
        
        return self.positions

    def get_type(self):

        return self.type

    def get_SEQRES(self):

        return self.SEQRES

    def get_ATOM_withoutX(self):

        return self.ATOM_withoutX

    def get_MODIFIED_COUNT(self):
        

        return self.MODIFIED_COUNT
        


    def get_MODIFIED_LIST(self):

        return self.MODIFIED_LIST



class cifParser:

    def __init__(self):

        self.SEQRES = ""
        #self.ATOM = ""
        self.ATOM_withoutX = {}
        self.type = ""
        self.MODIFIED_COUNT = 0
        self.MODIFIED_LIST = ""
        self.positions = {}
        self.max_res_details = 0
        self.auth_seq_id_column = 0
        self.auth_comp_id_column = 0
        self.auth_asym_id_column = 0
        self.B_iso_or_equiv = 0


    #def read(self, file, query_chain, DNA_AA, atom_position_filename):
    def read(self, file, query_chain, DNA_AA):

        #conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "a", "T" : "t", "C" : "c", "G" : "g", "U" : "u", "I" : "i", "DA" : "a", "DT" : "t", "DC" : "c", "DG" : "g", "DU" : "u", "DI" : "i", "5CM" : "c", "N" : "n"}
        conversion_table = {"ALA" : "A", "ARG" : "R", "ASN" : "N", "ASP" : "D", "CYS" : "C", "GLN" : "Q", "GLU" : "E", "GLY" : "G", "HIS" : "H", "ILE" : "I", "LEU" : "L", "LYS" : "K", "MET" : "M", "PHE" : "F", "PRO" : "P", "SER" : "S", "THR" : "T", "TRP" : "W", "TYR" : "Y", "VAL" : "V", "A" : "A", "T" : "T", "C" : "C", "G" : "G", "U" : "U", "I" : "I", "DA" : "A", "DT" : "T", "DC" : "C", "DG" : "G", "DU" : "U", "DI" : "I", "5CM" : "C", "N" : "N"}
        #modified_residues = {"MSE" : "MET", "MLY" : "LYS", "HYP" : "PRO", "CME" : "CYS", "CGU" : "GLU", "SEP" : "SER", "KCX" : "LYS", "MLE" : "LEU", "TPO" : "THR", "CSO" : "CYS", "PTR" : "TYR", "DLE" : "LEU", "LLP" : "LYS", "DVA" : "VAL", "TYS" : "TYR", "AIB" : "ALA", "OCS" : "CYS", "NLE" : "LEU", "MVA" : "VAL", "SEC" : "CYS", "PYL" : "LYS"}
        
        # find the portion of the file that contains the seqres
        SEQRES_string = ""
        SEQRES_string_found = False
        in_fasta = False
        fas_pos = 0
        last_pos = 0
        current_chain = ""
        hetatm = "" # the part of the sequence that in the HETATM lines
        hetatm_withoutX = "" # X is not added to fill the breaks in the sequence
        hetatm_pos = {} # the positions of the residues in the HETATM
        hetatm_max_res_details = 0 # maximum length of the details of the residues in the HETATM
        
        if DNA_AA == "Nuc":

            UnknownChar = "N"

        else:

            UnknownChar = "X"
            
        try:

            CIF = open(file, 'r')

        except:

            return 0
        
        line = CIF.readline()
        while line != "":

            if re.match(r'^_atom_site.', line):
                
                # we reached the atoms
                break
                
            if re.match(r'^_entity_poly.entity_id', line):

                while line != "":

                    if ';' in line:

                        in_fasta = not in_fasta

                    line = line.replace(";", "")

                    if '#' in line:

                        # end of _entity_poly.entity_id
                        SEQRES_string_found = True
                        break

                    elif in_fasta:

                        # delete white spaces in fasta
                        SEQRES_string += line.strip()

                    else:

                        SEQRES_string += line

                    line = CIF.readline()

            if SEQRES_string_found:

                break

            line = CIF.readline()

        if re.match(r'^_entity_poly.entity_id\s+1', SEQRES_string):

            # one seqres
            match1 = re.search(r'_entity_poly.pdbx_seq_one_letter_code_can\s+(\S+)', SEQRES_string)
            match2 = re.search(r'_entity_poly.pdbx_strand_id\s+(\S+)', SEQRES_string)
            if match1 and match2:
			
                seqres = match1.group(1)
                for chain in (match2.group(1)).split(','):

                    if chain == query_chain:
                    
                        self.SEQRES = seqres

        else:

            # more than one seqres
            SEQRES_substrings = re.split(r'\d+\s+\'?poly', SEQRES_string)

            POLY = open("poly", 'w')
            for string in SEQRES_substrings:

                POLY.write(string + "\n>\n")

            POLY.close()

            SEQRES_substrings = SEQRES_substrings[1:] # delete titles
            for substring in SEQRES_substrings:

                words = substring.split()
                for chain in (words[5]).split(','):
                    
                    if chain == query_chain:

                        self.SEQRES = words[4]
                    
        number_of_columns = 0
        # we find which columns has what value
        auth_seq_id_column = 0
        auth_comp_id_column = 0
        auth_asym_id_column = 0
        label_seq_id_column = 0
        label_comp_id_column = 0
        label_asym_id_column = 0
        B_iso_or_equiv = 0
        found_auth_seq_id_column = False
        found_auth_comp_id_column = False
        found_auth_asym_id_column = False
        found_label_seq_id_column = False
        found_label_comp_id_column = False
        found_label_asym_id_column = False
        found_B_iso_or_equiv = False
        while line != "":

            line = line.strip()

            if line == "_atom_site.B_iso_or_equiv":

                found_B_iso_or_equiv = True

            if line == "_atom_site.auth_seq_id":

                found_auth_seq_id_column = True

            if line == "_atom_site.auth_comp_id":

                found_auth_comp_id_column = True

            if line == "_atom_site.auth_asym_id":

                found_auth_asym_id_column = True
				
            if line == "_atom_site.label_seq_id":

                found_label_seq_id_column = True

            if line == "_atom_site.label_comp_id":

                found_label_comp_id_column = True

            if line == "_atom_site.label_asym_id":

                found_label_asym_id_column = True

            if not re.match(r'^_atom_site.', line) and found_B_iso_or_equiv and (found_auth_seq_id_column or found_label_seq_id_column) and (found_auth_comp_id_column or found_label_comp_id_column) and (found_auth_asym_id_column or found_label_asym_id_column):

                # we identified the necessary columns
                break

            if found_B_iso_or_equiv:

                B_iso_or_equiv -= 1

            if found_auth_seq_id_column:

                auth_seq_id_column -= 1

            if found_auth_comp_id_column:

                auth_comp_id_column -= 1

            if found_auth_asym_id_column:

                auth_asym_id_column -= 1
				
            if found_label_seq_id_column:

                label_seq_id_column -= 1

            if found_label_comp_id_column:

                label_comp_id_column -= 1

            if found_label_asym_id_column:

                label_asym_id_column -= 1

            line = CIF.readline()

        if not found_auth_seq_id_column:
		
            auth_seq_id_column = label_seq_id_column
			
        if not found_auth_comp_id_column:
		
            auth_comp_id_column = label_comp_id_column
			
        if not found_auth_asym_id_column:
		
            auth_asym_id_column = label_asym_id_column
			
        FIRST = []
        while line.strip() != "":

            words = line.split()
            if words[0] == "ATOM" and words[1].isnumeric():

                number_of_columns = len(words)

                # extract atom data
                pos = int(words[auth_seq_id_column])
                res = words[auth_comp_id_column]
                chain = words[auth_asym_id_column]

                # if HETATM is not in the end of the chain we add it to the sequence
                if chain == current_chain:
                    
                    self.ATOM_withoutX[chain] += hetatm_withoutX

                else:
                    
                    current_chain = chain
                    
                hetatm_withoutX = ""
                    
                # convert residue to one letter
                if res in conversion_table:
                    
                    oneLetter = conversion_table[res]
                    
                else:
                    
                    oneLetter = UnknownChar
                  
                # check if we reached a new residue
                if not chain in FIRST:
                    
                    FIRST.append(chain)
                    last_pos = pos
                    self.ATOM_withoutX[chain] = oneLetter
                    
                elif pos != last_pos:
                    
                    self.ATOM_withoutX[chain] += oneLetter

                else:

                    line = CIF.readline()
                    continue 
                     
                # if the chain is not the query we only extract the sequence 
                if query_chain != chain:
                         
                    #hetatm = ""
                    line = CIF.readline()
                    continue 
                """ 
                else:
                    
                    self.ATOM += hetatm
                    hetatm = ""
                """  
                    

                # writing atom position file
                
                self.positions.update(hetatm_pos)
                if hetatm_max_res_details > self.max_res_details:
                    
                    self.max_res_details = hetatm_max_res_details
                    
                hetatm_pos = {}
                hetatm_max_res_details = 0
                
                #CORR.write(hetatm_pos)
                #hetatm_pos = ""

                
                fas_pos += 1
                
                res_details = "%s:%d:%s" %(res, pos, chain)
                self.positions[fas_pos] = res_details
                if len(res_details) > self.max_res_details:
                    
                    self.max_res_details = len(res_details)
                    
                #CORR.write("%s\t%d\t%s\n" %(res, fas_pos, pos))
                                                
                # check type 
                if self.type == "":

                    if len(res) < 3:

                        self.type = "Nuc"

                    elif len(res) == 3:

                        self.type = "AA"
                """           
                if FIRST[chain]:
                    
                    FIRST[chain] = False

                elif last_pos < pos:
                    
                    while pos != last_pos + 1: # For Disorder regions
                    
                        self.ATOM += UnknownChar
                        last_pos += 1
                            
                self.ATOM += oneLetter
                """
                last_pos = pos
                
            elif words[0] == "HETATM" and words[1].isnumeric():
                

                # extract atom data
                pos = int(words[auth_seq_id_column])
                res = words[auth_comp_id_column]
                chain = words[auth_asym_id_column]

                    
                # convert residue to one letter
                if res in conversion_table:
                    
                    oneLetter = conversion_table[res]
                    
                else:
                    
                    oneLetter = UnknownChar
                  
                # check if we reached a new residue
                if not chain in FIRST:
                    
                    FIRST.append(chain)
                    last_pos = pos
                    self.ATOM_withoutX[chain] = ""
                    hetatm_withoutX = oneLetter
                    
                elif pos != last_pos:
                    
                    hetatm_withoutX += oneLetter

                else:

                    line = CIF.readline()
                    continue 
                     
                # if the chain is not the query we only extract the sequence 
                if query_chain != chain:
                         
                    line = CIF.readline()
                    continue 
                
                # writing atom position file
                fas_pos += 1
                
                res_details = "%s:%d:%s" %(res, pos, chain)
                hetatm_pos[fas_pos] = res_details
                if len(res_details) > hetatm_max_res_details:
                    
                    hetatm_max_res_details = len(res_details)
                    
                #hetatm_pos += "%s\t%d\t%s\n" %(res, fas_pos, pos)
                                                
                # check type 
                if self.type == "":

                    if len(res) < 3:

                        self.type = "Nuc"

                    elif len(res) == 3:

                        self.type = "AA"
                """            
                if FIRST[chain]:
                    
                    FIRST[chain] = False

                elif last_pos < pos:
                    
                    while pos != last_pos + 1: # For Disorder regions
                    
                        hetatm += UnknownChar
                        last_pos += 1
                            
                hetatm += oneLetter
                """
                last_pos = pos    
                
            line = CIF.readline()

        CIF.close()
        #CORR.close()
        
        self.auth_seq_id_column = number_of_columns + auth_seq_id_column
        self.auth_comp_id_column = number_of_columns + auth_comp_id_column
        self.auth_asym_id_column = number_of_columns + auth_asym_id_column
        self.B_iso_or_equiv = number_of_columns + B_iso_or_equiv
        LOG.write("residue number column - %s\nresidue name column - %s\nchain id column - %s\nb-factor column - %s\n" %(self.auth_seq_id_column + 1, self.auth_comp_id_column + 1, self.auth_asym_id_column + 1, self.B_iso_or_equiv + 1))
        
        return 1
        

    def get_max_res_details(self):
        
        return self.max_res_details

    def get_columns(self):
        
        return self.auth_seq_id_column, self.auth_comp_id_column, self.auth_asym_id_column, self.B_iso_or_equiv

    def get_type(self):

        return self.type

    def get_SEQRES(self):

        return self.SEQRES

    def get_ATOM_withoutX(self):

        return self.ATOM_withoutX

    def get_MODIFIED_COUNT(self):
        

        return self.MODIFIED_COUNT
        


    def get_MODIFIED_LIST(self):

        return self.MODIFIED_LIST


    def get_positions(self):
        
        return self.positions





    
bayesInterval = 3
ColorScale = {0 : 9, 1 : 8, 2 : 7, 3 : 6, 4 : 5, 5 : 4, 6 : 3, 7 : 2, 8 : 1}



def insert_spaces(original, spaced_lines, B_iso_or_equiv):

    # insert extra spaces after the tmp column, to allow room for the rate4site scores.

    try:

        READ_PDB = open(original, 'r')

    except:

        return("cp_rasmol_gradesPE_and_pipe.replace_tempFactor: Can't open '" + original + "' for reading.")

    lines = READ_PDB.readlines()
    READ_PDB.close()


    max_start = 0
    max_end = 0
    for line in lines:

        line = line.strip()
        if re.match(r'^ATOM', line):

            match = re.match(r'^((\S+\s+){'+ str(B_iso_or_equiv) + r'}\S+)\s+(\S.*)', line)
            length_start = len(match.group(1))
            length_end = len(match.group(3))
            if length_start > max_start:

                max_start = length_start

            if length_end > max_end:

                max_end = length_end

    for line in lines:

        if re.match(r'^ATOM', line):

            line = line.strip()
            match = re.match(r'^((\S+\s+){'+ str(B_iso_or_equiv) + r'}\S+)\s+(\S.*)', line)
            line_start = match.group(1)
            line_end = match.group(3)
            while len(line_start) < max_start:

                line_start = line_start + " "

            while len(line_end) < max_end:

                line_end = " " + line_end

            spaced_lines.append(line_start + "     " + line_end + "\n")

        else:

            spaced_lines.append(line)




def check_if_rate4site_failed(r4s_log):

    res_flag = vars['r4s_out']
    if not os.path.exists(res_flag) or os.path.getsize(res_flag) == 0: # 1

        LOG.write("check_if_rate4site_failed : the file " + res_flag + " either does not exist or is empty. \n")
        return True

    try:

        R4S_RES = open(res_flag, 'r')

    except:

        LOG.write("check_if_rate4site_failed : can not open file: " + res_flag + ". aborting.\n")
        return True

    error = False
    line = R4S_RES.readline()
    while line != "":

        if "likelihood of pos was zero" in line:

            LOG.write("check_if_rate4site_failed : the line: \"likelihood of pos was zero\" was found in %s.\n" %r4s_log)
            error = True
            break

        if re.match(r'rate of pos\:\s\d\s=', line):

            # output found
            break

        if "Bad format in tree file" in line:

            exit_on_error('user_error', "check_if_rate4site_failed : There is an error in the tree file format. Please check that your tree is in the <a href = \"" + GENERAL_CONSTANTS.CONSURF_TREE_FAQ + "\">requested format</a> and reupload it to the server.<br>")

        line = R4S_RES.readline()

    R4S_RES.close()
    return error



"""
def check_validity_tree_file(nodes):

	# checks validity of tree file and returns an array with the names of the nodes
    try:

        TREEFILE = open(form['tree_name'], 'r')

    except:

        exit_on_error('sys_error', "check_validity_tree_file : can't open the file " + form['tree_name'] + " for reading.")

    tree = TREEFILE.read()
    TREEFILE.close()
    tree.replace("\n", "")
    if tree[-1] != ';':
	
	    tree += ';'
		
    try:

        TREEFILE = open(vars['working_dir'] + vars['tree_file'], 'w')

    except:

        exit_on_error('sys_error', "check_validity_tree_file : can't open the file " + vars['working_dir'] + vars['tree_file'] + " for writing.")
        
    TREEFILE.write(tree)
    TREEFILE.close()

    leftBrackets = 0
    rightBrackets = 0
    noRegularFormatChar = ""
    #nodes = []
    node_name = ""
    in_node_name = False
    in_node_score = False
    for char in tree:
        
        if char == ':':
            
            if in_node_name:
                
                nodes.append(node_name)
                
            node_name = ""
            in_node_name = False
            in_node_score = True

        elif char == '(':

            leftBrackets += 1                
            
        elif char == ')':

            rightBrackets += 1
            in_node_score = False
            
        elif char == ',':
            
            in_node_score = False
            
        elif char != ';':

            if char in "!@#$^&*~`{}'?<>" and not char in noRegularFormatChar: 

                noRegularFormatChar += " '" + char + "', "
                
            if not in_node_score:
                
                node_name += char
                in_node_name = True

    if leftBrackets != rightBrackets:

        msg = "The uploaded tree file, which appears to be in Newick format, is missing parentheses."
        exit_on_error('user_error', msg)

    if noRegularFormatChar != "":

        msg = "The uploaded tree file, which appears to be in Newick format, ontains the following non-standard characters: " + noRegularFormatChar[:-2]
        exit_on_error('user_error', msg)
        
    LOG.write("check_validity_tree_file : tree is valid\n")

    #return nodes
"""



    
def get_query_seq_in_MSA():
    
    # returns the query sequence with gaps as it's in the MSA
    try:
        
        FASTA = open(vars['msa_fasta'], 'r')
        
    except:
        
        exit_on_error('sys_error', "get_query_seq_in_MSA : can't open the file " + vars['msa_fasta'] + " for reading.")
    
    found = False
    seq = ""
    line = FASTA.readline()
    while line != "":
        
        first_word = line.split()[0]
        if found:
            
            if first_word[0] == '>':
                
                break
            
            else:
                
                seq += first_word
                
        elif first_word == ">"  + vars['msa_SEQNAME']:
                
                found = True
            
        line = FASTA.readline()   
        
    FASTA.close()
    return seq
    
def get_positions_in_MSA():
    
    # returns the positions in the MSA were the query is not a gap
    seq = get_query_seq_in_MSA()
    positions = get_seq_legal_positions(seq)
    return positions

def get_seq_legal_positions(seq):

    # return a aaray with the positions of the legal chars
    positions = []
    if form['DNA_AA'] == "AA":
        
        legal_chars = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]
        
    else:
        
        legal_chars = ["A", "C", "G", "T", "U", "N"]
	
    for i in range(len(seq)):
        
        if seq[i].upper() in legal_chars:
            
            positions.append(i)
            
    return positions
            

    
def write_MSA_percentage_file():
    
    # writes a file with the precentage of each acid in each position 
    
    #index_of_pos = get_positions_in_MSA() 
    percentage_per_pos = [] # precentage of each acid in each position in the MSA
    #unknown_per_pos = [] # precentage of unknown acid in each position in the MSA
    number_of_positions = len(vars['protein_seq_string'])
    #number_of_positions = len(index_of_pos)
    query_with_gaps = get_query_seq_in_MSA() # query sequence with gaps

    if form['DNA_AA'] == "AA":
        
        acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]
        unknown = "X"

        for i in range(number_of_positions):
        
            percentage_per_pos.append({})
            #unknown_per_pos.append(0)
            
    else:
        
        acids = ["A", "C", "G", "T", "U", "N"]
        unknown = "N"

        for i in range(number_of_positions):
        
            percentage_per_pos.append({})
            #unknown_per_pos.append(0)

    try:
        
        FASTA = open(vars['msa_fasta'], 'r')
        
    except:
        
        exit_on_error('sys_error', "write_MSA_percentage_file : can't open the file " + vars['msa_fasta'] + " for reading.")
    
    try:
        
        PRECENTAGE_FILE = open(vars['Msa_percentageFILE'], 'w')
        
    except:
        
        exit_on_error('sys_error', "write_MSA_percentage_file : can't open the file " + vars['Msa_percentageFILE'] + " for writing.")
    
    # we find the precentage of each acid in each position 
    seq = ""
    first = True
    line = FASTA.readline()
    while True:
        
        if line == "" or line[0] == '>':
            
            if first:
                
                first = False 
                
            else:
            
                #pos = 0
                #for i in index_of_pos:
                i = 0
                for pos in range(number_of_positions):
                
                    while i < len(query_with_gaps) and query_with_gaps[i] == '-':
					
                        i += 1
						
                    char = seq[i]
                    if char in  acids:
                    
                        if char in percentage_per_pos[pos]:
                            
                            percentage_per_pos[pos][char] += 1
                            
                        else:
                    
                            percentage_per_pos[pos][char] = 1
							
                    """
                    elif char != '-':
                    
                        unknown_per_pos[pos] += 1
                    """
                    i += 1                       
            
                seq = ""
            
            # this is for the last sequence in the msa
            if line == "":

                break
			
        else:
            
            seq += line.strip().upper()
			
        line = FASTA.readline()   

    FASTA.close()
    
    # sort dictionaries
    for pos in range(number_of_positions):
		
        # we sort the amino acids but not the unknown character 
        unknown_percent = percentage_per_pos[pos].pop(unknown, None)
        percentage_per_pos[pos] = dict(sorted(percentage_per_pos[pos].items(), key=lambda item: item[1], reverse=True))
        if unknown_percent:
		
            percentage_per_pos[pos][unknown] = unknown_percent
		
    # calculate the percentage 
    for pos in range(number_of_positions):
			
        sum = 0.0
        for char in percentage_per_pos[pos]:
            
            sum += percentage_per_pos[pos][char]
            
        #sum += unknown_per_pos[pos]
			
        for char in percentage_per_pos[pos]:
            
            percentage_per_pos[pos][char] = 100 * (percentage_per_pos[pos][char] / sum)
            
        #unknown_per_pos[pos] = 100 * (unknown_per_pos[pos] / sum)
            
    # we write the file
    PRECENTAGE_FILE.write("\"The table details the residue variety in % for each position in the query sequence.\"\n\"Each column shows the % for that amino-acid, found in position ('pos') in the MSA.\"\n\"In case there are residues which are not a standard amino-acid in the MSA, they are represented under column 'OTHER'\"\n\npos,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,OTHER,MAX ACID,ConSurf grade\n\n")   
    pos = 0
    for i in range(number_of_positions):
        
        if vars['protein_seq_string'][i] == unknown:
		
            continue
			
		# position
        PRECENTAGE_FILE.write("%d" %(pos + 1))

		# known acids
        for char in acids:
            
            PRECENTAGE_FILE.write(",")
            if char in percentage_per_pos[pos]:
                
                PRECENTAGE_FILE.write("%.3f" %percentage_per_pos[pos][char])
                
            else:
                
                PRECENTAGE_FILE.write("0")
                
		# unknown acids
        #PRECENTAGE_FILE.write(",%.3f" %unknown_per_pos[pos])
        
		# max acid
        keys = list(percentage_per_pos[pos].keys())
        max_acid = ",%s %.3f" %(keys[0], percentage_per_pos[pos][keys[0]])
        PRECENTAGE_FILE.write(max_acid.rstrip("0").rstrip("."))

        # ConSurf grade
        PRECENTAGE_FILE.write(",%s\n" %vars['gradesPE_Output'][pos]['COLOR'])

        pos += 1
		
    PRECENTAGE_FILE.close()

    vars['percentage_per_pos'] = percentage_per_pos
    #vars['unknown_per_pos'] = unknown_per_pos
    
    
    if form['DNA_AA'] != "AA": 
	
        vars['B/E'] = False
        return
		
    vars['B/E'] = True
    buried_exposed = []
    reveal_buried_exposed(buried_exposed)
	
    pos = 0 
    for element in vars['gradesPE_Output']:
    
        while pos < number_of_positions and vars['protein_seq_string'][pos] == unknown:

            pos += 1
			
        element['B/E'] = buried_exposed[pos]
        pos += 1
        if element['B/E'] == "e":

            if element['COLOR'] == 9 or element['COLOR'] == 8:

                element['F/S'] = "f"

            else:

                element['F/S'] = " "

        elif element['COLOR'] == 9:

            element['F/S'] = "s"

        else:

            element['F/S'] = " "
            

def database_address(name): 
    
    parts = name.split('|')
    if parts[0] == "ur":

        return("https://www.uniprot.org/uniref/UniRef90_" + parts[1])

    elif parts[0] == "up":

        return("https://www.uniprot.org/uniprot/" + parts[1])
    
    else:
        
        return("https://www.ncbi.nlm.nih.gov/nuccore/" + parts[1])

def get_details(first_line, second_line, third_line):

    query_seq = (first_line.split())[2]
    db_seq = (third_line.split())[2]

    gaps = 0
    positives = 0
    identities = 0
    length = len(query_seq)

    for char in query_seq:

        if char == "-":

            gaps += 1

    for char in db_seq:

        if char == "-":

            gaps += 1

    for char in second_line:

        if char == "+":

            positives += 1

        if char.isalpha():

            identities += 1
            
    details = ""
    
    if gaps != 0:
        
        details += ", Gaps = %d/%d (%d%%)" %(gaps, length, int((float(gaps) / length) * 100))
        
    else:
        
        details += ", Gaps = 0"
        
    if positives != 0:
        
        details += ", Positives = %d/%d (%d%%)" %(positives, length, int((float(positives) / length) * 100))
        
    else:
        
        details += ", Positives = 0"
        
    if identities != 0:
        
        details += ", Identities = %d/%d (%d%%)" %(identities, length, int((float(identities) / length) * 100))
        
    else:
        
        details += ", Identities = 0"

    return details


def replace_TmpFactor_Rate4Site_Scores_CIF(PdbFile, chain, HashRef, Out):

    # replace the tempFactor column in the PDB file

    [auth_seq_id_column, auth_comp_id_column, auth_asym_id_column, B_iso_or_equiv] = vars['pdb_object'].get_columns()
    
    spaced_lines = []
    insert_spaces(PdbFile, spaced_lines, B_iso_or_equiv)

    # write the PDB file and replace the tempFactor column
    # with the new one.
    try:

        WRITE_PDB = open(Out, 'w')

    except:

        exit_on_error('sys_error', "replace_TmpFactor_Rate4Site_Scores_CIF: Can't open '" + Out + "' for writing.")

    for line in spaced_lines:

        if re.match(r'^ATOM', line):

            words = line.split()
            PDBchain = words[auth_asym_id_column]
            ResNum = words[auth_seq_id_column]
            if PDBchain == chain and ResNum in HashRef:

                score = HashRef[ResNum]
                match = re.match(r'^((\S+\s+){'+ str(B_iso_or_equiv) + r'})(\S+\s+)(.+)', line)
                length_temp_fact = len(match.group(3))
                while len(score) < length_temp_fact:

                    score = score + " "

                WRITE_PDB.write(match.group(1) + score + match.group(4) + "\n")

            else:

                WRITE_PDB.write(line)

        else:

            WRITE_PDB.write(line)

    WRITE_PDB.close()



def replace_TmpFactor_Rate4Site_Scores_PDB(PdbFile, chain, HashRef, Out):

    # replace the tempFactor column in the PDB file

    # read the PDB file to an array
    try:

        READ_PDB = open(PdbFile, 'r')

    except:

        exit_on_error('sys_error', "replace_TmpFactor_Rate4Site_Scores_PDB : Can't open '" + PdbFile + "' for reading.")

    # write the PDB file and replace the tempFactor column
    # with the new one.
    try:

        WRITE_PDB = open(Out, 'w')

    except:

        exit_on_error('sys_error', "replace_TmpFactor_Rate4Site_Scores_PDB : Can't open '" + Out + "' for writing.")

    line = READ_PDB.readline()
    while line != "":

        if re.match(r'^ATOM', line):

            PDBchain = line[21:22]
            ResNum = (line[22:27]).strip()

            if PDBchain == chain and ResNum in HashRef:

                while len(HashRef[ResNum]) < 6:

                    HashRef[ResNum] = " " + HashRef[ResNum]

                WRITE_PDB.write(line[:60] + HashRef[ResNum] + line[66:])

            else:

                WRITE_PDB.write(line[:60] + "      " + line[66:])

        else:

            WRITE_PDB.write(line)

        line = READ_PDB.readline()

    READ_PDB.close()
    WRITE_PDB.close()




def read_Rate4Site_gradesPE(gradesPE_file, gradesPE_hash_ref):

    # the routine matches each position in the gradesPE file its Rate4Site grade.

    try:

        GRADES = open(gradesPE_file, 'r')

    except:

        exit_on_error("sys_error", "read_Rate4Site_gradesPE : Can't open '" + gradesPE_file + "' for reading.")

    line = GRADES.readline()
    while line != "":

        if re.match(r'^\s*\d+\s+\w', line):

            grades = line.split()
            #ResNum = re.sub(r'[a-z]', "", grades[2], flags=re.IGNORECASE)
            if grades[2] != "-":
            
                 grades[2] = (grades[2]).split(":")[1]
                 
            gradesPE_hash_ref[grades[2]] = grades[3]

        line = GRADES.readline()

    GRADES.close()




def read_ConSurf_gradesPE(gradesPE_file, gradesPE_hash_ref, gradesPE_ISD_hash_ref):

    # the routine matches each position in the gradesPE file its ConSurf grade. In case there was a grade mark with *, we put it in a seperate hash with the grade 0.
    # the routine returns "yes" if a * was found and "no" otherwise

    insufficient = False

    try:

        GRADES = open(gradesPE_file, 'r')

    except:

        exit_on_error('sys_error', "read_ConSurf_gradesPE can't open '" + gradesPE_file + "' for reading.")

    line = GRADES.readline()
    while line != "":

        if re.match(r'^\s*\d+\s+\w', line):

            grades = line.split()
            if grades[2] != "-":
            
                 grades[2] = ((grades[2]).split(':'))[1]
                 
            #grades[2] = re.sub(r'[a-z]', "", grades[2], flags=re.IGNORECASE)
            if re.match(r'\d\*?', grades[4]):

                # if it is insufficient color - we change its grade to 0, which will be read as light yellow
                if re.match(r'(\d)\*', grades[4]):

                    gradesPE_hash_ref[grades[2]] = grades[4]
                    gradesPE_ISD_hash_ref[grades[2]] = "10"
                    insufficient = True

                else:

                    gradesPE_hash_ref[grades[2]] = grades[4]
                    gradesPE_ISD_hash_ref[grades[2]] = grades[4]

        line = GRADES.readline()

    GRADES.close()

    return(insufficient)



def replace_TmpFactor_Consurf_Scores_CIF(atom_grades, query_chain, pdb_file, prefix):

    # Creates The ATOM section with ConSurf grades instead of the TempFactor column, creates PDB file with ConSurf grades

    pdb_with_grades = prefix + "_ATOMS_section_With_ConSurf.cif"
    pdb_with_grades_isd = prefix + "_ATOMS_section_With_ConSurf_isd.cif"
    pdb_with_scores = prefix + "_With_Conservation_Scores.cif"
	
    [auth_seq_id_column, auth_comp_id_column, auth_asym_id_column, B_iso_or_equiv] = vars['pdb_object'].get_columns()

    try:

        PDB = open(pdb_file, 'r')

    except:

        exit_on_error('sys_error', "could not open file '" + pdb_file + "' for reading.\n")

    try:

        GRADES = open(pdb_with_grades, 'w')

    except:

        exit_on_error('sys_error', "could not open the file '" + pdb_with_grades + "' for writing.\n")

    try:

        SCORES = open(pdb_with_scores, 'w')

    except:

        exit_on_error('sys_error', "could not open the file '" + pdb_with_scores + "' for writing.\n")
		
    if vars['insufficient_data']:

        try:

            GRADES_ISD = open(pdb_with_grades_isd, 'w')

        except:

            exit_on_error('sys_error', "could not open the file '" + pdb_with_grades_isd + "' for writing.\n")

    line = PDB.readline()
    while line != "":

        if line[:4] == "ATOM" or line[:6] == "HETATM":

            words = line.split()
            chain = words[auth_asym_id_column]
            residue_number = words[auth_seq_id_column]

            grade = "0"
            score = "0"
            grade_isd = "0"

            if residue_number in atom_grades and chain == query_chain:

                # getting the grade
                
                [grade, isd, score] = atom_grades[residue_number]

                if vars['insufficient_data']:

                    if isd == 1:
					
                        grade_isd = "10"
						
                    else:
					
                        grade_isd = grade
					
            match = re.match(r'^((\S+\s+){' + str(B_iso_or_equiv) + r'})(\S+\s+)(.+)', line)
            length_temp_fact = len(match.group(3))
            line_start = match.group(1)
            line_end = match.group(4)
            while len(grade) < length_temp_fact:

                grade = grade + " "
				
            while len(score) < length_temp_fact:

                score = score + " "

            while len(grade_isd) < length_temp_fact:

                grade_isd = grade_isd + " "

            GRADES.write(line_start + grade + line_end + "\n")
            SCORES.write(line_start + score + line_end + "\n")
            if vars['insufficient_data']:

                GRADES_ISD.write(line_start + grade_isd + line_end + "\n")
                

        else:

            GRADES.write(line)
            SCORES.write(line)
            if vars['insufficient_data']:

                GRADES_ISD.write(line)

        line = PDB.readline()

    GRADES.close()
    SCORES.close()	
    vars['zip_list'].append(pdb_with_grades)
    vars['zip_list'].append(pdb_with_scores)
    if vars['insufficient_data']:
        
        show_py3dmol(pdb_with_grades_isd, "cif")
        print_instructions(pdb_with_grades, "CIF", pdb_with_grades_isd)
    
    else:
        
        show_py3dmol(pdb_with_grades, "cif")
        print_instructions(pdb_with_grades, "CIF")

def add_remark():
    
    remark = """REMARK 998                                                                          
REMARK 998   This file has been modified by ConSurf. 
REMARK 998   Publications: https://pubmed.ncbi.nlm.nih.gov/?term=consurf
REMARK 998   https://consurf.tau.ac.il    https://consurfdb.tau.ac.il 
REMARK 998   https://colab.research.google.com/drive/1PhDXX7k12oUsV6T_xkXC3Rm9R99e7tHz
REMARK 998                                                                              
REMARK 998   B-factor/temperature values have been replaced by conservation grades
REMARK 998   1.0 (variable) to 9.0 (conserved), or 10.0 when the MSA had insufficient
REMARK 998   data for a reliable grade.
REMARK 998                                                                                      
REMARK 998   Chain %s was processed
""" %(form['PDB_chain'])

    if vars['running_mode'] == "_mode_pdb_no_msa" or vars['running_mode'] == "_mode_no_pdb_no_msa":
        
        remark += """REMARK 998   %s E value: %s
REMARK 998   Maximum sequence identity: %s%%
REMARK 998   Minimum sequence identity: %s%%
REMARK 998   Unique sequences found: %s
REMARK 998   Sequences used in MSA: %s
REMARK 998   MSA Algorithm: %s
""" %(form['Homolog_search_algorithm'], form['E_VALUE'], form['MIN_IDENTITY'], vars['hit_redundancy'], vars['unique_seqs'], vars['final_number_of_homologoues'], form['MSAprogram'])

        if form['best_uniform_sequences'] == "best":
            
            remark += "REMARK 998   MSA sequences closest to the query were chosen\n"
            
        else:
            
            remark += "REMARK 998   MSA sequences were Sampled across all unique sequences found\n"
            
    if form['ALGORITHM'] == "Bayes":
        
        remark += "REMARK 998   Rate4site Algorithm: Bayesian\n"
        
    else:
        
        remark += "REMARK 998   Rate4site Algorithm: Maximum likelihood\n"
        
    if form['SUB_MATRIX'] == "JC_Nuc":
        
        substitution_model = "JC"
        
    else:
        
        substitution_model = form['SUB_MATRIX']
        
    remark += """REMARK 998   Substitution model: %s
REMARK 998   MSA Average Pairwise Distance (APD): %.2f
REMARK 998                                                                                
""" %(substitution_model, float(vars['Average pairwise distance']))
            
    return remark

def replace_TmpFactor_Consurf_Scores_PDB(atom_grades, query_chain, pdb_file, prefix):

    # Creates The ATOM section with ConSurf grades instead of the TempFactor column, creates PDB file with ConSurf grades


    pdb_with_grades = prefix + "_ATOMS_section_With_ConSurf.pdb"
    pdb_with_grades_isd = prefix + "_ATOMS_section_With_ConSurf_isd.pdb"
    pdb_with_scores = prefix + "_With_Conservation_Scores.pdb"

    remark_added = False
    remark_found = False

    try:

        PDB = open(pdb_file, 'r')

    except:

        exit_on_error('sys_error', "replace_TmpFactor_Consurf_Scores_PDB : could not open file '" + pdb_file + "' for reading.\n")

    try:

        GRADES = open(pdb_with_grades, 'w')

    except:

        exit_on_error('sys_error', "replace_TmpFactor_Consurf_Scores_PDB : could not open the file '" + pdb_with_grades + "' for writing.\n")

    try:

        SCORES = open(pdb_with_scores, 'w')

    except:

        exit_on_error('sys_error', "replace_TmpFactor_Consurf_Scores_PDB : could not open the file '" + pdb_with_scores + "' for writing.\n")

    if vars['insufficient_data']:

        try:

            GRADES_ISD = open(pdb_with_grades_isd, 'w')

        except:

            exit_on_error('sys_error', "replace_TmpFactor_Consurf_Scores_PDB : could not open the file '" + pdb_with_grades_isd + "' for writing.\n")

    line = PDB.readline()
    while line != "":

        if line.strip() == "":

            # line is empty
            line = PDB.readline()
            continue
			
        if line[:4] == "ATOM" or line[:6] == "HETATM":

            if not remark_added:
                        
                remark = add_remark()
                GRADES.write(remark)
                SCORES.write(remark)
                if vars['insufficient_data']:

                    GRADES_ISD.write(remark)
                            
                remark_added = True
                        
            chain = line[21:22]
            if chain == " ":
			
                chain = "NONE"
				
            residue = (line[22:27]).strip()
            if residue in atom_grades and chain == query_chain:
  
                [grade, isd, score] = atom_grades[residue]
                while len(score) < 6:

                    score = " " + score
                    
                while len(grade) < 6:
                    
                    grade = " " + grade
					
                # the TF is updated with the grades and scores
                GRADES.write(line[:60] + grade + line[66:])
                SCORES.write(line[:60] + score + line[66:])

                if vars['insufficient_data']:

                    # the TF is updated with the number from gradesPE showing isd
                    if isd == 1:
					
                        white_space = " " * 4
                        GRADES_ISD.write(line[:60] + white_space + "10" + line[66:])
						
                    else:
					
                        GRADES_ISD.write(line[:60] + grade + line[66:])

            else:
			
                white_space = " " * 6
                GRADES.write(line[:60] + white_space + line[66:])
                SCORES.write(line[:60] + white_space + line[66:])
                if vars['insufficient_data']:

                    GRADES_ISD.write(line[:60] + white_space + line[66:])

        else:

            if not remark_added:
                
                if line.startswith("REMARK 999"):

                    remark = add_remark()
                    GRADES.write(remark)
                    SCORES.write(remark)
                    if vars['insufficient_data']:
    
                        GRADES_ISD.write(remark)
                                
                    remark_added = True  
                    
                elif line.startswith("REMARK"):
                    
                    remark_found = True
                    
                elif remark_found:
                                                
                    remark = add_remark()
                    GRADES.write(remark)
                    SCORES.write(remark)
                    if vars['insufficient_data']:
    
                        GRADES_ISD.write(remark)
                                
                    remark_added = True    
                    
                else:
                    
                    for field in ["DBREF", 
                                  "SEQADV", 
                                  "SEQRES", 
                                  "MODRES", 
                                  "HET", 
                                  "HETNAM", 
                                  "HETSYN", 
                                  "FORMUL",
                                  "HELIX",
                                  "SHEET",
                                  "SSBOND",
                                  "LINK",
                                  "CISPEP",
                                  "SITE",
                                  "CRYST1",
                                  "ORIGX",
                                  "SCALE",
                                  "MTRIX"]:
                    
                        if line.startswith(field):
                            
                            remark = add_remark()
                            GRADES.write(remark)
                            SCORES.write(remark)
                            if vars['insufficient_data']:
    
                                GRADES_ISD.write(remark)
                                
                            remark_added = True
                        
            GRADES.write(line)
            SCORES.write(line)
            if vars['insufficient_data']:

                GRADES_ISD.write(line)

        line = PDB.readline()

    GRADES.close()
    SCORES.close()	
	

    vars['zip_list'].append(pdb_with_grades)
    vars['zip_list'].append(pdb_with_scores)
    if vars['insufficient_data']:
        
        show_py3dmol(pdb_with_grades_isd, "pdb")
        print_instructions(pdb_with_grades, "PDB", pdb_with_grades_isd)
    
    else:
        
        show_py3dmol(pdb_with_grades, "pdb")
        print_instructions(pdb_with_grades, "PDB")

def design_string_with_spaces_for_pipe(part_input):

    if part_input.strip() == "":
        
        return ""
    
    words = part_input.split()
    newPart = "! \"" +words[0]
    part = ""
    for word in words[1:]:

        # if adding another word to the string will yeild a too long string - we cut it.
        if len(word) + 1 + len(newPart) > 76:

            part += newPart + " \" +\n"
            newPart = "! \"" + word

        else:

            newPart += " " + word

    part += newPart + "\" ;"

    return part



def add_pdb_data_to_pipe(pdb_file, pipe_file):

    # create the file to be shown using FGiJ. read the pdb file and concat header pipe to it.

    try:

        PIPE = open(pipe_file, 'a')

    except:

        exit_on_error('sys_error', "add_pdb_data_to_pipe: cannot open " + pipe_file + " for writing.")

    try:

        PDB_FILE = open(pdb_file, 'r')

    except:

        exit_on_error('sys_error', "add_pdb_data_to_pipe: cannot open the " + pdb_file + " for reading.")

    line = PDB_FILE.readline()
    while line != "":

        if not re.match(r'^HEADER', line):

            PIPE.write(line)

        line = PDB_FILE.readline()

    PIPE.close()
    PDB_FILE.close()



def print_selected(arr_ref, print_for_pipe):

    total_text = ""
    string = ""
    if print_for_pipe == "yes":

        string = "! select "

    else:

        string = "select "

    total_length = len(string)

    if len(arr_ref) > 0:

        for aa in arr_ref:

            aa = aa.replace(":", "")
            total_length += len(aa)
            if total_length > 80:

                if re.search(r', $', string):

                    string = string[:-2]

                total_text += string + "\n"
                if print_for_pipe == "yes":

                    string = "! select selected or %s, " %aa

                else:

                    string = "select selected or %s, " %aa

                total_length = len(string)

            else:

                string += aa + ", "
                total_length += 2

    else:

        total_text += string + "none"


    if re.search(r', $', string):

        string = string[:-2]
        total_text += string

    return total_text




def create_consurf_pipe_new(results_dir, IN_pdb_id_capital, chain, ref_header_title, final_pipe_file, identical_chains, partOfPipe, current_dir, run_number, msa_filename, query_name_in_msa = "", tree_filename = "", submission_time = "", completion_time = "", run_date = ""):


    # Create the pipe file for FGiJ

    if chain == 'NONE':

        chain = ""
        identical_chains = ""

    # read info from the pdb file
    [header_line, title_line] = ref_header_title

    if title_line == "":

        title_line = "! \"No title or compound description was found in the PDB file\";"

    else:

        title_line = design_string_with_spaces_for_pipe(title_line)

    # design the identical chains line
    identical_chains_line = "! consurf_identical_chains = \"%s\";" %identical_chains

    current_dir += "_sourcedir"
    # in case there is a source dir - we determine the var query_name_in_msa
    if os.path.exists(current_dir):

        try:

            SOURCEDIR = open(current_dir, 'r')

        except:

            exit_on_error('sys_error', "create_consurf_pipe : cannot open " + current_dir + " for reading.")

        match = re.match(r'(\d[\d\w]{3})\/(\w)', SOURCEDIR.readline())
        if match:

            query_name_in_msa = SOURCEDIR.group(1) + SOURCEDIR.group(2)
            SOURCEDIR.close()

    if query_name_in_msa == "":

        query_name_in_msa = IN_pdb_id_capital + chain.upper()

    # write to the pipe file
    try:

        PIPE_PART = open(partOfPipe, 'r')

    except:

        exit_on_error('sys_error', "create_consurf_pipe : cannot open " + partOfPipe + " for reading.")

    try:

        PIPE = open(final_pipe_file, 'w')

    except:

        exit_on_error('sys_error', "create_consurf_pipe : cannot open " + final_pipe_file + " for writing.")

    if header_line != "":

        PIPE.write(header_line + "\n")

    else:

        PIPE.write("HEADER                                 [THIS LINE ADDED FOR JMOL COMPATIBILITY]\n")

    PIPE.write("""!! ====== IDENTIFICATION SECTION ======
!js.init
! consurf_server = "consurf";
! consurf_version = "3.0";
! consurf_run_number = \"%s\";
! consurf_run_date = \"%s\";
! consurf_run_submission_time = \"%s\";
! consurf_run_completion_time = \"%s\";
!
! consurf_pdb_id = \"%s\";
! consurf_chain = \"%s\";
%s
! consurf_msa_filename = \"%s\";
! consurf_msa_query_seq_name = \"%s\";
! consurf_tree_filename = \"%s\";
!
""" %(run_number, run_date, submission_time, completion_time, IN_pdb_id_capital, chain, identical_chains_line, msa_filename, query_name_in_msa, tree_filename))

    titleFlag = 0
    line = PIPE_PART.readline()
    while line != "":

        if re.match(r'^~~~+', line):

            if titleFlag == 0:

                PIPE.write("! pipe_title = \"<i>ConSurf View:</i> %s chain %s.\"\n!! pipe_subtitle is from TITLE else COMPND\n!!\n! pipe_subtitle =\n%s\n" %(IN_pdb_id_capital, chain, title_line))
                titleFlag = 1

            elif chain != "":

                PIPE.write("! select selected and :%s\n" %chain)

            else:

                PIPE.write("! select selected and protein\n")

        else:

            PIPE.write(line)

        line = PIPE_PART.readline()

    PIPE_PART.close()
    PIPE.close()




def freq_array(isd_residue_color, no_isd_residue_color):

    # design the frequencies array

    consurf_grade_freqs_isd = "Array(" + str(len(isd_residue_color[10]))
    i = 1
    while i < 10:

        consurf_grade_freqs_isd += "," + str(len(isd_residue_color[i]))
        i += 1

    consurf_grade_freqs_isd += ")"

    consurf_grade_freqs = "Array(0"
    i = 1
    while i < 10:

        consurf_grade_freqs += "," + str(len(no_isd_residue_color[i]))
        i += 1

    consurf_grade_freqs += ")"

    return(consurf_grade_freqs_isd, consurf_grade_freqs)


def design_string_for_pipe(string_to_format):

    # take a string aaaaaaa and returns it in this format: ! "aaa" +\n! "aa";\n

    part = string_to_format
    newPart = ""

    while len(part) > 73:

        newPart += "! \"" + part[:73] + "\" +\n"
        part = part[73:]

    newPart += "! \"" + part + "\" ;"

    return newPart


def extract_data_from_pdb(input_pdb_file):

    header = ""
    title = ""
    compnd = ""

    try:

        PDB = open(input_pdb_file, 'r')

    except:

        exit_on_error('sys_error', "extract_data_from_pdb : Could not open the file " + input_pdb_file + " for reading.")

    line = PDB.readline()
    while line != "":

        match1 = re.match(r'^HEADER', line)
        if match1:

            header = line.rstrip()

        else:

            match2 =re.match(r'^TITLE\s+\d*\s(.*)', line)
            if match2:

                title += match2.group(1) + " "

            else:

                match3 = re.match(r'^COMPND\s+\d*\s(.*)', line)
                if match3:

                    compnd += match3.group(1) + " "

                elif re.match(r'^SOURCE', line) or re.match(r'^KEYWDS', line) or re.match(r'^AUTHOR', line) or re.match(r'^SEQRES', line) or re.match(r'^ATOM', line):

                    break # no nead to go over all the pdb

        line = PDB.readline()

    PDB.close()
    if title == "":
	
        return header, compnd
		
    else:
	 
        return header, title


def create_part_of_pipe_new(pipe_file, unique_seqs, db, seq3d_grades_isd, seq3d_grades, length_of_seqres, length_of_atom, ref_isd_residue_color, ref_no_isd_residue_color, E_score, iterations, max_num_homol, MSAprogram, algorithm, matrix, Average_pairwise_distance, scale = "legacy"):

    # creating part of the pipe file, which contains all the non-unique information.
    # each chain will use this file to construct the final pdb_pipe file, to be viewed with FGiJ

    if scale == "legacy":

        scale_block = "!color color_grade0 FFFF96 insufficient data yellow\n!color color_grade1 10C8D1 turquoise variable\n!color color_grade2 8CFFFF\n!color color_grade3 D7FFFF\n!color color_grade4 EAFFFF\n!color color_grade5 FFFFFF\n!color color_grade6 FCEDF4\n!color color_grade7 FAC9DE\n!color color_grade8 F07DAB\n!color color_grade9 A02560 burgundy conserved"

    else:

        scale_block = "!color color_grade0 FFFF96 insufficient data yellow\n!color color_grade1 1b7837 variable\n!color color_grade2 5aae61\n!color color_grade3 a6dba0\n!color color_grade4 d9f0d3\n!color color_grade5 f7f7f7\n!color color_grade6 e7d4e8\n!color color_grade7 c2a5cf\n!color color_grade8 9970ab\n!color color_grade9 762a83 conserved"

    # design the seq3d to be printed out to the pipe file
    seq3d_grades_isd = design_string_for_pipe(seq3d_grades_isd)
    seq3d_grades = design_string_for_pipe(seq3d_grades)

    # creating the frequencies array which corresponds the number of residues in each grade
    [consurf_grade_freqs_isd, consurf_grade_freqs] = freq_array(ref_isd_residue_color, ref_no_isd_residue_color)

    # Taking Care of Strings
    if max_num_homol == "all":

        max_num_homol = "\"all\""

    # write to the pipe file
    try:

        PIPE = open(pipe_file, 'w')

    except:

        exit_on_error('sys_error', "create_part_of_pipe_new : cannot open the file " + pipe_file + " for writing.", 'PANIC')

    PIPE.write("""! consurf_psi_blast_e_value = %s;
! consurf_psi_blast_database = "%s";
! consurf_psi_blast_iterations = %s;
! consurf_max_seqs = %s;
! consurf_apd = %.2f;
! consurf_alignment = "%s";
! consurf_method = "%s";
! consurf_substitution_model =  "%s";
!
! consurf_seqres_length = %s;
! consurf_atom_seq_length = %s;
! consurf_unique_seqs = %s;
! consurf_grade_freqs_isd = %s;
! consurf_grade_freqs = %s;
!
! seq3d_grades_isd =
%s
!
! seq3d_grades =
%s
!
!
!! ====== CONTROL PANEL OPTIONS SECTION ======
!js.init
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! pipe_title_enlarged = false;
! pipe_background_color = "white";
!
!! Specify the custom consurf control panel
!!
! pipe_cp1 = "consurf/consurf.htm";
!
!! If you want the frontispiece to be reset every time you enter this
!! page, use false. If this is a one-page presentation (no contents)
!! and you want to be able to return from QuickViews without resetting
!! the view, use true.
!!
! frontispiece_conditional_on_return = true;
!
!! Open the command input slot/message box to 30%% of window height.
!!
! pipe_show_commands = true;
! pipe_show_commands_pct = 30;
!
!! Don't show the PiPE presentation controls in the lower left frame.
!!
! pipe_hide_controls = true;
!
!! Hide development viewing mode links at the bottom of the control panel.
!!
! pipe_tech_info = false;
!
!! pipe_start_spinning = true; // default is PE's Preference setting.
!! top.nonStopSpin = true; // default: spinning stops after 3 min.
!!
!! ====== COLORS SECTION ======
!!
!color color_carbon C8C8C8
!color color_sulfur FFC832
!
!! Ten ConSurf color grades follow:
!!
%s
!
!
!! ====== SCRIPTS SECTION ======
!!----------------------------------------
!!
!spt #name=select_and_chain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
!!----------------------------------------
!!
!spt #name=view01
! @spt consurf_view_isd
!
!!----------------------------------------
!!
!spt #name=hide_all
! restrict none
! ssbonds off
! hbonds off
! dots off
! list * delete
!
!!----------------------------------------
!! common_spt uses CPK carbon gray (or phosphorus yellow) for backbones.
!!
!spt #name=common_spt
! @spt hide_all
! select all
! color [xC8C8C8] # rasmol/chime carbon gray
! select nucleic
! color [xFFA500] # phosphorus orange
! select hetero
! color cpk
! select not hetero
! backbone 0.4
! javascript top.water=0
!
! ssbonds 0.3
! set ssbonds backbone
! color ssbonds @color_sulfur
!
! select hetero and not water
! spacefill 0.45
! wireframe 0.15
! dots 50
!
! select protein
! center selected
!
!!----------------------------------------
!!
!spt #name=consurf_view_isd
! @spt common_spt
! @for $=0, 9
! @spt select_isd_grade$
! @spt select_and_chain
! color @color_grade$
! spacefill
! @endfor
! zoom 115
!
!!----------------------------------------
""" %(E_score, db, iterations, max_num_homol, round(float(Average_pairwise_distance), 2), MSAprogram, algorithm, matrix, length_of_seqres, length_of_atom, unique_seqs, consurf_grade_freqs_isd, consurf_grade_freqs, seq3d_grades_isd, seq3d_grades, scale_block))

    lineToPrint = ""
    i = 9
    while i > 0:

        PIPE.write("!!\n!spt #name=select_isd_grade%d\n!\n" %i)
        lineToPrint = print_selected(ref_isd_residue_color[i], "yes")
        if re.search(r'select', lineToPrint):

            PIPE.write(lineToPrint + "\n")

        PIPE.write("!\n!\n!!----------------------------------------\n")
        i -= 1

    PIPE.write("!!\n!spt #name=select_isd_grade0\n")
    lineToPrint = print_selected(ref_isd_residue_color[10], "yes")
    if re.search(r'select', lineToPrint):

        PIPE.write(lineToPrint + "\n")

    PIPE.write("!\n!\n!!----------------------------------------\n")

    i = 9
    while i > 0:

        PIPE.write("!!\n!spt #name=select_grade%d\n!\n" %i)
        lineToPrint = print_selected(ref_no_isd_residue_color[i], "yes")
        if re.search(r'select', lineToPrint):

            PIPE.write(lineToPrint + "\n")

        PIPE.write("!\n!\n!!----------------------------------------\n")
        i -= 1

    PIPE.write("!!\n!spt #name=select_grade0\n! select none\n!!\n")
    PIPE.write("!! ====== END OF CONSURF PiPE BLOCK ======\n")
    PIPE.close()



def create_gradesPE(gradesPE, ref_match = "", pdb_file = "", chain = "", prefix = "", pdb_object = "", identical_chains = "", pdb_cif = "", atom_grades = ""):


    # printing the the ConSurf gradesPE file

    if pdb_cif == "pdb": # this is for the pipe file
	
        seq3d_grades_isd = ""
        seq3d_grades = ""
        # arrays showing how to color the residues. The subarrays holds residues of the same color
        no_isd_residue_color = [[],[],[],[],[],[],[],[],[],[],[]] # no insufficient data
        isd_residue_color = [[],[],[],[],[],[],[],[],[],[],[]] # insufficient data

    try:

        PE = open(gradesPE, 'w')

    except:

        exit_on_error('sys_error', "create_gradesPE : can't open '" + gradesPE + "' for writing.")

    if form['DNA_AA'] == "AA":

        unknown_char = "X"
        PE.write("\t Amino Acid Conservation Scores\n")
		
    else:

        unknown_char = "N"
        PE.write("\t Nucleic Acid Conservation Scores\n")

    PE.write("\t=======================================\n\n")
    PE.write("The layers for assigning grades are as follows.\n")
    for i in range(1, len(vars['layers_array'])):
        
        if vars['layers_array'][i - 1] < 0:
        
            left_end = "%.3f" %vars['layers_array'][i - 1]
            
        else:
            
            left_end = " %.3f" %vars['layers_array'][i - 1]
    
            
        if vars['layers_array'][i] < 0:
        
            right_end = "%.3f" %vars['layers_array'][i]
            
        else:
            
            right_end = " %.3f" %vars['layers_array'][i]
            
        PE.write("from %s to %s the grade is %d\n" %(left_end, right_end, 10 - i))
        
    PE.write("\nIf the difference between the colors of the CONFIDENCE INTERVAL COLORS is more than 3 or the msa number (under the column titled MSA) is less than 6, there is insufficient data and an * appears in the COLOR column.\n")


    PE.write("\n- POS: The position of the acid in the sequence.\n")
    PE.write("- SEQ: The acid one letter.\n")
    PE.write("- ATOM: When there's a model, The ATOM derived sequence in three letter code, including the acid's positions as they appear in the PDB file and the chain identifier.\n")
    PE.write("- SCORE: The normalized conservation scores.\n")
    PE.write("- COLOR: The color scale representing the conservation scores (9 - conserved, 1 - variable).\n")
    PE.write("- CONFIDENCE INTERVAL: When using the bayesian method for calculating rates, a confidence interval is assigned to each of the inferred evolutionary conservation scores, next to it are the colors of the lower and upper bounds of the confidence interval\n")
    if vars['B/E']:
	
        PE.write("- B/E: Burried (b) or Exposed (e) residue.\n")
        PE.write("- F/S: functional (f) or structural (s) residue (f - highly conserved and exposed, s - highly conserved and burried).\n")
		
    PE.write("- MSA DATA: The number of aligned sequences having an acid (non-gapped) from the overall number of sequences at each position.\n")
    PE.write("- RESIDUE VARIETY: The residues variety at each position of the multiple sequence alignment.\n\n")



    if form['ALGORITHM'] == "Bayes":
        
        CONFIDENCE_INTERVAL = "CONFIDENCE INTERVAL\t"
        
    else:
        
        CONFIDENCE_INTERVAL = ""
  
    # the size of the POS, ATOM and MSA DATA columns is variable 
	
    pos_number_size = len(str(len(vars['protein_seq_string']))) # size of the of the number of positions
    pos_column_title_size = len("POS") # size of the title of the column POS
    pos_column_size = max(pos_column_title_size, pos_number_size) # size of the column POS
    num_spaces = pos_column_size - pos_column_title_size # number of spaces to add 
    while num_spaces > 0:
	
        PE.write(" ")
        num_spaces -= 1
		
    PE.write("POS\tSEQ\t")
	
    if ref_match != "":

        # consurf. in conseq there is no atom because there is no model
        max_res_details = pdb_object.get_max_res_details()
        atom_column_title_size = len("ATOM") # size of the title of the column ATOM
        atom_column_size = max(atom_column_title_size, max_res_details) # size of the column ATOM
        num_spaces = atom_column_size - atom_column_title_size # number of spaces to add
        while num_spaces > 0:

            PE.write(" ")
            num_spaces -= 1

        PE.write("ATOM\t")
		
    if vars['B/E']:
        
        PE.write(" SCORE\tCOLOR\t%sB/E\tF/S\t" %CONFIDENCE_INTERVAL)

    else:
    
        PE.write(" SCORE\tCOLOR\t%s" %CONFIDENCE_INTERVAL)
                
    msa_size = len(str(vars['final_number_of_homologoues'])) # size of the number of homologs
    msa_column_title_size = len("MSA DATA") # size of the title of the column MSA DATA
    msa_column_size = max(msa_column_title_size, 2 * msa_size + 1) # size of the column MSA DATA
    num_spaces = msa_column_size - msa_column_title_size # number of spaces to add 
    while num_spaces > 0:
	
        PE.write(" ")
        num_spaces -= 1
		
    PE.write("MSA DATA\tRESIDUE VARIETY\n\n")
				
    seq_index = 0 # the index of the position in the query sequence (the rate4site output doesn't contain unknown chars)		
    for elem in vars['gradesPE_Output']:

        pos = elem['POS']
        var = ""
        num_spaces = pos_column_size - len(str(pos)) # number of spaces to add 
        while num_spaces > 0:
	
            PE.write(" ")
            num_spaces -= 1
			
        PE.write("%d\t" %pos)
        PE.write("  %s\t" %elem['SEQ'])
		
        if ref_match != "":

            # consurf, in conseq there is no atom because there is no model
            atom_3L = ref_match[pos]
            num_spaces = atom_column_size - len(atom_3L) # number of spaces to add
            while num_spaces > 0:

                PE.write(" ")
                num_spaces -= 1

            PE.write("%s\t" %atom_3L)

            # save the grade of the residue inorder to write it on the pdb file
            if atom_3L != '-':
			
                residue_number = atom_3L.split(':')[1]
                atom_grades[residue_number] = [str(elem['COLOR']), elem['ISD'], "%6.3f" %elem['GRADE']]

        PE.write("%6.3f\t" %elem['GRADE'])
        if elem['ISD'] == 1:

            PE.write("   %d*\t" %elem['COLOR'])

        else:

            PE.write("   %d \t" %elem['COLOR'])

        if form['ALGORITHM'] == "Bayes":
            
            PE.write("%6.3f, " %elem['INTERVAL_LOW'])
            PE.write("%6.3f  " %elem['INTERVAL_HIGH'])
            PE.write("%d," %ColorScale[elem['INTERVAL_LOW_COLOR']])
            PE.write("%d\t" %ColorScale[elem['INTERVAL_HIGH_COLOR']])
        
        if vars['B/E']:

            PE.write("  %s\t  %s\t" %(elem['B/E'], elem['F/S']))
			
        homologs_in_pos = str(elem['MSA_NUM']) + "/" + elem['MSA_DENUM'] # number of homologs in the position
        num_spaces = msa_column_size - len(homologs_in_pos) # number of spaces to add 
        while num_spaces > 0:
	
            PE.write(" ")
            num_spaces -= 1
			
        PE.write(homologs_in_pos)
		
		# we write the acids percentage in the msa
        while seq_index < len(vars['protein_seq_string']) and vars['protein_seq_string'][seq_index] == unknown_char:
		
            seq_index += 1
			
        for char in vars['percentage_per_pos'][seq_index]:
		
            if vars['percentage_per_pos'][seq_index][char] == 100:
			
                var += char

            else:
			
				# if the precentage is less than one write <1%
                if vars['percentage_per_pos'][seq_index][char] > 1:
				
                    var += "%s %2d%%, " %(char, vars['percentage_per_pos'][seq_index][char])
					
                else:
				
                    var += "%s <1%%, " %char

        """
        if vars['unknown_per_pos'][pos - 1] != 0:
		
			# if the precentage is less than one write <1%
            if vars['unknown_per_pos'][pos - 1] > 1:
			
                var += "%s %2d%%" %(unknown_char, vars['unknown_per_pos'][pos - 1])
				
            else:
			
                var += "%s <1%%" %unknown_char
        """
        if len(vars['percentage_per_pos'][seq_index]) != 1:

            var = var[:-2] # we delete the last comma

        PE.write("\t" + var + "\n")
        seq_index += 1

        # the amino-acid in that position, must be part of the residue variety in this column
        if not re.search(elem['SEQ'], var, re.IGNORECASE):

            PE.close()
            exit_on_error('sys_error', "create_gradesPE : in position %s, the amino-acid %s does not match the residue variety: %s." %(pos, elem['SEQ'], var))

        if pdb_cif != "pdb": # the next part is for the pipe file
		
            continue

        # printing the residue to the rasmol script
        # assigning grades to seq3d strings
        if not '-' in atom_3L:

            atom_3L = re.search(r'(.+):', atom_3L).group(1)
            if form['DNA_AA'] == "Nuc":

                atom_3L = "D" + atom_3L

            color = elem['COLOR']
            no_isd_residue_color[color].append(atom_3L)
            if elem['ISD'] == 1:

                isd_residue_color[10].append(atom_3L)
                seq3d_grades_isd += "0"

            else:

                isd_residue_color[color].append(atom_3L)
                seq3d_grades_isd += str(color)

            seq3d_grades += str(color)

        else:

            seq3d_grades_isd += "."
            seq3d_grades += "."

    PE.write("\n\n*Below the confidence cut-off - The calculations for this site were performed on less than 6 non-gaped homologue sequences,\n")
    PE.write("or the confidence interval for the estimated score is equal to- or larger than- 4 color grades.\n")
    PE.close()

    if pdb_cif != "pdb": # the next part is for the pipe file
	
        return

    if seq3d_grades_isd == "" or seq3d_grades == "":

        exit_on_error('sys_error', "create_gradesPE : there is no data in the returned values seq3d_grades_isd or seq3d_grades from the routine")

    """
    # This will create the pipe file for FGiJ
    pipeFile = prefix + "_consurf_firstglance.pdb"
    pipeFile_CBS = prefix + "_consurf_firstglance_CBS.pdb" # pipe for color blind friendly
    create_pipe_file(pipeFile, pipeFile_CBS, seq3d_grades, seq3d_grades_isd, isd_residue_color, no_isd_residue_color, pdb_file, chain, (prefix).upper(), identical_chains, pdb_object)

    # create RasMol files
    create_rasmol(prefix, chain, no_isd_residue_color, isd_residue_color)
    """
    
def match_pdb_to_seq(ref_fas2pdb, query_seq, pdbseq, pdb_object):

    # matches the position in the seqres/msa sequence to the position in the pdb

    UnKnownChar = ""
    if form['DNA_AA'] == "AA":

        UnKnownChar = "X"

    else:

        UnKnownChar = "N"

    # creating the hash that matches the position in the ATOM fasta to its position
    # in the pdb file and also the fasta ATOM position to the correct residue
    match_ATOM = pdb_object.get_positions()
    """
    # creating the hash that matches the position in the ATOM fasta to its position
    # in the pdb file and also the fasta ATOM position to the correct residue
    try:

        MATCH = open(atom_position, 'r')

    except:

        exit_on_error('sys_error', "match_pdb_to_seq : Could not open the file " + atom_position + " for reading.")

    max_res_details = 0 # longest residue details string. Dictates the size of the ATOM column in the results summary page 
    match_ATOM = {}
    line = MATCH.readline()
    while line != "":

        words = line.split()
        if len(words) == 3:

            res_details = words[0] + ":" + words[2] + ":" + chain
            match_ATOM[int(words[1])] = res_details
            if res_details > max_res_details:
			
                max_res_details = res_details

        line = MATCH.readline()

    MATCH.close()

    length_of_atoms = 0
    for char in pdbseq:

        if char != '-' and char != UnKnownChar:

            length_of_atoms += 1

    length_of_seqres = 0
    for char in query_seq:

        if char != '-' and char != UnKnownChar:

            length_of_seqres += 1
    """
	
    query_pos = 1
    pdb_pos = 1
    for pos in range(len(query_seq)):
        
        if query_seq[pos] != '-' and query_seq[pos] != UnKnownChar:
            
            if pdbseq[pos] == '-' or pdb_pos not in match_ATOM: 

                ref_fas2pdb[query_pos] = '-'

            else:

                ref_fas2pdb[query_pos] = match_ATOM[pdb_pos]
                pdb_pos += 1
                
            query_pos += 1
            
        elif pdbseq[pos] != '-':
               
            pdb_pos += 1
            
    #return length_of_seqres, length_of_atoms, max_res_details
	


def find_pdb_position(ref_fas2pdb, pdb_object):

    # finds the position of the the sequence in the pdb

    LOG.write("find_pdb_position(ref_fas2pdb, pdb_object)\n")

    UnKnownChar = ""
    if form['DNA_AA'] == "AA":

        UnKnownChar = "X"

    else:

        UnKnownChar = "N"
	
    match_ATOM = pdb_object.get_positions()
    # rate4site deletes the unknown chars
    rate4site_pos = 1
    pdb_pos = 1
    for char in vars['ATOM_without_X_seq']:
        
        if char != UnKnownChar:

            ref_fas2pdb[rate4site_pos] = match_ATOM[pdb_pos]                
            rate4site_pos += 1
                           
        pdb_pos += 1
            


def assign_colors_according_to_r4s_layers():

    LOG.write("assign_colors_according_to_r4s_layers(%s, %s)\n" %(vars['gradesPE_Output'], vars['r4s_out']))
	
    vars['insufficient_data'] = False
    """
    if form['DNA_AA'] == "AA":
       
	    #runs the PACC algorithm to calculate burried/exposed
        ref_Solv_Acc_Pred = predict_solvent_accesibility() 
	    # this array connects the position to its index in ref_Solv_Acc_Pred (positions don't include unknown characters, PACC does)
        index_of_pos = get_seq_legal_positions(vars['protein_seq_string'])
		
    else:

        ref_Solv_Acc_Pred = ""
		
    if ref_Solv_Acc_Pred != "":
	
        vars['B/E'] = True
		
    else:
	
        vars['B/E'] = False

    """
	# we extract the data from the rate4site output
    try:

        RATE4SITE = open(vars['r4s_out'], 'r')

    except:

        exit_on_error('sys_error', "assign_colors_according_to_r4s_layers : can't open " + vars['r4s_out'])

    line = RATE4SITE.readline()
    while line != "":

        line.rstrip()

        if form['ALGORITHM'] == "Bayes":

            # baysean
            match1 = re.match(r'^\s*(\d+)\s+(\w)\s+(\S+)\s+\[\s*(\S+),\s*(\S+)\]\s+\S+\s+(\d+)\/(\d+)', line)
            if match1:

                vars['gradesPE_Output'].append({'POS' : int(match1.group(1)), 'SEQ' : match1.group(2), 'GRADE' : float(match1.group(3)), 'INTERVAL_LOW' : float(match1.group(4)), 'INTERVAL_HIGH' : float(match1.group(5)), 'MSA_NUM' : int(match1.group(6)), 'MSA_DENUM' : match1.group(7)})

        else:

            # Maximum likelihood
            match2 = re.match(r'^\s*(\d+)\s+(\w)\s+(\S+)\s+(\d+)\/(\d+)', line)
            if match2:

                vars['gradesPE_Output'].append({'POS' : int(match2.group(1)), 'SEQ' : match2.group(2), 'GRADE' : float(match2.group(3)), 'INTERVAL_LOW' : float(match2.group(3)), 'INTERVAL_HIGH' : float(match2.group(3)), 'MSA_NUM' : int(match2.group(4)), 'MSA_DENUM' : match2.group(5)})

        line = RATE4SITE.readline()

    RATE4SITE.close()

    # we find the maximum and the minimum scores    
    max_cons = vars['gradesPE_Output'][0]['GRADE']
    min_cons = vars['gradesPE_Output'][0]['GRADE']
    for element in vars['gradesPE_Output']:

        if element['GRADE'] < max_cons:

            max_cons = element['GRADE']
            
        if element['GRADE'] > min_cons:

            min_cons = element['GRADE']

    # we divide the interval between min_cons to max_cons to nine intervals
    # 4 intervals on the left side are of length 2 * min_cons / 9
    # 4 intervals on the right side are of length 2 * max_cons / 9
    NoLayers = 10
    LeftLayers = 5
    RightLayers = 5
    ColorLayers = []
    i = 0
    while i < LeftLayers:

        ColorLayers.append(max_cons * ((9 - 2 * i) / 9.0))
        i += 1
        
    i = 0
    while i < RightLayers:

        ColorLayers.append(min_cons * ((1 + 2 * i) / 9.0))
        i += 1

    # each position gets a grade according to the layer its score is in
    for element in vars['gradesPE_Output']:

        i = 0
        while not 'INTERVAL_LOW_COLOR' in element or not 'INTERVAL_HIGH_COLOR' in element or not 'COLOR' in element:

            if not 'INTERVAL_LOW_COLOR' in element:

                if i == NoLayers - 1:

                    element['INTERVAL_LOW_COLOR'] = 8

                elif element['INTERVAL_LOW'] >= ColorLayers[i] and element['INTERVAL_LOW'] < ColorLayers[i + 1]:

                    element['INTERVAL_LOW_COLOR'] = i

                elif element['INTERVAL_LOW'] < ColorLayers[0]:

                    element['INTERVAL_LOW_COLOR'] = 0

            if not 'INTERVAL_HIGH_COLOR' in element:

                if i == NoLayers - 1:

                    element['INTERVAL_HIGH_COLOR'] = 8

                elif element['INTERVAL_HIGH'] >= ColorLayers[i] and element['INTERVAL_HIGH'] < ColorLayers[i + 1]:

                    element['INTERVAL_HIGH_COLOR'] = i

                elif element['INTERVAL_HIGH'] < ColorLayers[0]:

                    element['INTERVAL_HIGH_COLOR'] = 0

            if not 'COLOR' in element:

                if i == NoLayers - 1:

                    element['COLOR'] = ColorScale[i - 1]

                elif element['GRADE'] >= ColorLayers[i] and element['GRADE'] < ColorLayers[i + 1]:

                    element['COLOR'] = ColorScale[i]

            i += 1

		# there is insufficient data if there are more than 3 layers in the confidence interval or the number of homologs in the MSA where the position is not empty is less than 5
        if element['INTERVAL_HIGH_COLOR'] - element['INTERVAL_LOW_COLOR'] > bayesInterval or element['MSA_NUM'] <= 5:

            element['ISD'] = 1
            vars['insufficient_data'] = True

        else:

            element['ISD'] = 0
        """
        if vars['B/E']:
		
            element['B/E'] = ref_Solv_Acc_Pred[index_of_pos[element['POS'] - 1]]
            if element['B/E'] == "e":

                if element['COLOR'] == 9 or element['COLOR'] == 8:

                    element['F/S'] = "f"

                else:

                    element['F/S'] = " "

            elif element['COLOR'] == 9:

                element['F/S'] = "s"

            else:

                element['F/S'] = " "
        """
    vars['layers_array'] = ColorLayers
    LOG.write("assign_colors_according_to_r4s_layers : color layers are %s\n" %str(vars['layers_array']))




def extract_diversity_matrix_info(r4s_log_file):

    # extracting diversity matrix info

    matrix_disINFO = "\"\""
    matrix_lowINFO = "\"\""
    matrix_upINFO = "\"\""

    try:

        RES_LOG = open(r4s_log_file, 'r')

    except:

        exit_on_error('sys_error', "extract_diversity_matrix_info: Can't open '" + r4s_log_file + "' for reading\n")

    line = RES_LOG.readline()
    while line != "":

        line = line.rstrip()
        match1 = re.match(r'\#Average pairwise distance\s*=\s+(.+)', line)
        if match1:

            matrix_disINFO = match1.group(1)

        else:

            match2 = re.match(r'\#lower bound\s*=\s+(.+)', line)
            if match2:

                matrix_lowINFO = match2.group(1)

            else:

                match3 = re.match(r'\#upper bound\s*=\s+(.+)', line)
                if match3:

                    matrix_upINFO = match3.group(1)
                    break

        line = RES_LOG.readline()

    RES_LOG.close()

    vars['Average pairwise distance'] = matrix_disINFO


def add_sequences_removed_by_cd_hit_to_rejected_report(cd_hit_clusters_file, rejected_fragments_file, num_rejected_homologs):

    LOG.write("add_sequences_removed_by_cd_hit_to_rejected_report : running add_sequences_removed_by_cd_hit_to_rejected_report(%s, %s, %d)\n" %(cd_hit_clusters_file, rejected_fragments_file, num_rejected_homologs))

    try:

        REJECTED = open(rejected_fragments_file, 'a')

    except:

        exit_on_error('sys_error', "extract_sequences_removed_by_cd_hit: Can't open '" + rejected_fragments_file + "' for writing.")

    try:

        CDHIT = open(cd_hit_clusters_file, 'r')

    except:

        exit_on_error('sys_error', "extract_sequences_removed_by_cd_hit: Can't open '" + cd_hit_clusters_file + "' for reading.\n")

    REJECTED.write("\n\t Sequences rejected in the clustering stage by CD-HIT\n\n")
    
    cluster_members = {}
    cluster_head = ""

    line = CDHIT.readline()
    while line != "":

        match = re.match(r'^>Cluster', line)
        if match:

            # New Cluster
            for cluster_member in cluster_members.keys():

                REJECTED.write("%d Fragment %s rejected: the sequence shares %s identity with %s (which was preserved)\n" %(num_rejected_homologs, cluster_member, cluster_members[cluster_member], cluster_head))
                num_rejected_homologs += 1

            cluster_members = {}
            cluster_head = ""

        else:

            # Clusters Members
            words = line.split()
            if len(words) > 2:

                x = words[2][1:-3] # delete the symbols > and ... from the beginning and and ending of the sequence name
                if words[3] == "*":

                    cluster_head  = x

                elif len(words) > 3:

                    cluster_members[x] = words[4]

        line = CDHIT.readline()

    vars['num_rejected_homologs'] = num_rejected_homologs



def choose_homologoues_from_search_with_lower_identity_cutoff(searchType, query_seq_length, redundancyRate, frag_overlap, min_length_percent, min_id_percent, min_num_of_homologues, search_output, fasta_output, rejected_seqs, ref_search_hash, Nuc_or_AA):

    # searchType: HMMER, BLAST or MMseqs2
    # query_seq_length: Length of the query sequence
    # redundancyRate: The allowed similarity between the query and the hit
    # frag_overlap: The allowed overlap between the hits
    # min_length_percent: The hit can't be smaller than this percent of the query
    # min_id_percent: The minimum similarity between the query and the hit
    # min_num_of_homologues: Minimum number of homologs
    # search_output: Raw homolog search output
    # fasta_output: Accepted hits
    # rejected_seqs: Rejected hits
    # ref_search_hash: Hash with the evalues of the accepted hits. This is later used when choosing the final hits after cid-hit
    # Nuc_or_AA: Amino or nucleic acid

    LOG.write("choose_homologoues_from_search_with_lower_identity_cutoff(%s, %d, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);\n" %(searchType, query_seq_length, redundancyRate, frag_overlap, min_length_percent, min_id_percent, min_num_of_homologues, search_output, fasta_output, rejected_seqs, ref_search_hash, Nuc_or_AA))

    # Defining the minimum length a homologue should have
    # 60% the query's length
    min_length = query_seq_length * min_length_percent

    # Reading blast/hmmer output and collect the homologues
    # Printing the selected homologues to a file and insert the e-value info to hash

    OUT_REJECT = open(rejected_seqs, 'w')
    OUT = open(fasta_output, 'w')
    RAW_OUT = open(search_output, 'r')

    num_homologoues = 0
    num_rejected = 1
    final_num_homologues = 0
    OUT_REJECT.write("\tSequences rejected for being too short, too similar, or not similar enough to the query sequence.\n\n")
    if searchType == "MMseqs2":

        # we skip the first two lines which have the query
        line = RAW_OUT.readline()
        line = RAW_OUT.readline()
        line = RAW_OUT.readline()
        while line != "":

            if line[0] == ">":

                # this line has the sequence details
                words = line.split()

                # new seq found

                num_homologoues += 1
                seq_name = words[0][1:]

                seq_eval = float(words[3])
                seq_beg = int(words[7])
                seq_end = int(words[8])
                seq_ident = float(words[2]) * 100

            elif line[0] != "\x00":

                # this line has the sequence. We take it and save it with the details of the previous line
                seq = line.strip() # the seq in fasta with gaps
                seq = re.sub(r'-', "", seq) # delete gaps
                seq_frag_name = "%s|%d_%d|%s" %(seq_name, seq_beg, seq_end, seq_eval)

                # deciding if we take the fragment
                ans = check_if_seq_valid_with_min_id(redundancyRate, min_length, min_id_percent, seq_ident, seq, seq_name, Nuc_or_AA)

                if seq_eval > form['E_VALUE']:
                    
                    ans = "The E-value %.16g is over the limit %.16g." %(seq_eval, form['E_VALUE'])
                    
                # after taking the info, check if the currecnt sequence is valid. If so - insert it to the hash
                if ans == "yes":

                    final_num_homologues += 1
                    OUT.write(">%s\n%s\n" %(seq_frag_name, seq))
                    ref_search_hash[seq_frag_name] = seq_eval

                else:

                    OUT_REJECT.write("%d Fragment %s rejected: %s\n" %(num_rejected, seq_frag_name, ans))
                    num_rejected += 1

            line = RAW_OUT.readline()

    OUT_REJECT.close()
    OUT.close()
    # Checking that the number of homologues found is legal

    if final_num_homologues < min_num_of_homologues:

        message = "Only %d unique sequences were chosen. The minimal number of sequences required for the calculation is %d. You can try to rerun with a multiple sequence alignment file of your own, increase the E value or decrease the minimal %%ID for homologs" %(final_num_homologues, min_num_of_homologues)
        exit_on_error("", message)

    vars['number_of_homologoues'] = num_homologoues
    vars['number_of_homologoues_before_cd-hit'] = final_num_homologues
    vars['num_rejected_homologs'] = num_rejected



def change_name(s_name):
    
    if s_name[:2] == "sp" or s_name[:2] == "tr":
        
        words = s_name.split("|")
        return  "up|" + words[1]
    
    elif s_name[:2] == "gi":
    
        words = s_name.split("|")
        return  "gi|" + words[1]
    
    elif s_name[:8] == "UniRef90":
        
        words = s_name.split("_")
        return "ur|" + words[1]
    
    elif "|" in s_name:
        
        words = s_name.split("|")
        return "up|" + words[1]
    
    else:
        
        return "gi|" + s_name    
    



def check_if_seq_valid_with_min_id(redundancyRate, min_length, min_id, ident_percent, aaSeq, seqName, Nuc_or_AA):

    seq_length = len(aaSeq)

    if ident_percent >= redundancyRate:

        # the sequence identity is not too high
        return "identity percent %.2f is too big" %ident_percent

    if ident_percent < min_id:

        # the sequence identity is higher than the minium idnentity percent that was defined for homologus
        return "identity percent %.2f is too low (below %d)" %(ident_percent, min_id)

    elif seq_length < min_length:

        # the sequnece length is greater than the minimum sequence length
        return "the sequence length %d is too short. The minimum is %d" %(seq_length, min_length)

    return check_illegal_character(aaSeq, seqName, Nuc_or_AA)


def check_illegal_character(aaSeq, seqName, Nuc_or_AA):
    
    # the sequnece letters should be legal to rate4site

    if Nuc_or_AA == "AA":

        # AA seq
        if not re.match(r'^[ACDEFGHIKLMNPQRSTVWYBZXacdefghiklmnpqrstvwybzx]+$', aaSeq):

            return "illegal character was found in sequence: " + seqName

    else:

        # Nuc seq
        if not re.match(r'^[ACGTUINacgtuin]+$', aaSeq):

            return "illegal character was found in sequence: " + seqName

    return "yes" 

def check_if_no_overlap(max_overlap, ref_seq_details, s_bgn, s_end):

    ans = "check_if_no_overlap : no ans was picked"

    i = 0
    while i < len(ref_seq_details):

        fragment_beg = ref_seq_details[i][0]
        fragment_end = ref_seq_details[i][1]
        fragment_length = int(fragment_end) - int(fragment_beg) + 1

        if s_bgn <= fragment_beg and s_end >= fragment_end:

            # fragment is inside subjct
            return "previous fragment found %s_%s is fully inside new fragment" %(fragment_beg, fragment_end)

        elif s_bgn >= fragment_beg and s_end <= fragment_end:

            # subjct is inside fragment
            return "new fragment is fully inside previous fragment found " + str(fragment_beg + fragment_end)

        elif fragment_end < s_end and fragment_end > s_bgn:

            # fragment begins before subjct
            overlap_length = fragment_end - s_bgn + 1
            if overlap_length > fragment_length * max_overlap:

                return "overlap length of fragment is %d which is greater than maximum overlap: %d" %(overlap_length, fragment_length * max_overlap)

            else:

                # when the fragment might be a good match, we can only insert it if it did not match to all the fragments
                if i == len(ref_seq_details) - 1:

                    ans = "yes"

        elif fragment_beg > s_bgn and  fragment_beg < s_end:

            # fragment begins after subjct
            overlap_length = s_end - fragment_beg + 1
            if overlap_length > fragment_length * max_overlap:

                return "overlap length of fragment is %d which is greater than maximum overlap: %d" %(overlap_length, fragment_length * max_overlap)

            else:

                # when the fragment might be a good match, we can only insert it if it did not match to all the fragments
                if i == len(ref_seq_details) - 1:

                    ans = "yes"

        elif fragment_beg >= s_end or fragment_end <= s_bgn:

            # no overlap
            if i == len(ref_seq_details) - 1:

                ans = "yes"

        i += 1

    return ans


def count_letters(s):

    l = 0
    for c in s:

        if c.isalpha():

            l += 1

    return l


def write_alignment(first_seq, first_seq_name, second_seq, second_seq_name, middle_line, clustalw_aln):
    
    # we want all the lines to begin at the same point
    while len(first_seq_name) < 15:
        
        first_seq_name += " "
        
    while len(second_seq_name) < 15:
        
        second_seq_name += " "
        
    middle_line_beginning = ""
    while len(middle_line_beginning) < 15:
        
        middle_line_beginning += " "
        
    try:

        CLUSTALW_ALN = open(clustalw_aln, 'w')

    except:

        exit_on_error('sys_error', "write_alignment : could not open " + clustalw_aln + " for writing.")
    
    seq_too_long_for_page = True
    while seq_too_long_for_page:
        
        # we show only 60 chars of the seq in each line
        if len(first_seq) <= 60:
            
            seq_too_long_for_page = False
            
        CLUSTALW_ALN.write(first_seq_name + first_seq[:60] + "\n")
        CLUSTALW_ALN.write(middle_line_beginning + middle_line[:60] + "\n")
        CLUSTALW_ALN.write(second_seq_name + second_seq[:60] + "\n\n")

        first_seq = first_seq[60:]
        middle_line = middle_line[60:]
        second_seq = second_seq[60:]
        
    CLUSTALW_ALN.close()
    
def pairwise_alignment(first_seq, second_seq, clustalw_aln = "", seq_type = ""):
	
	# for new Bio
    aligner = Align.PairwiseAligner()
	
    # Pairwise Alignment Paramaters
	
    aligner.mode = 'global' #Can be either global or local, if undetermined, biopython will choose optimal algorithem

    if form['DNA_AA'] == "AA":
        
        aligner.substitution_matrix = substitution_matrices.load(vars['git_dir'] + "matrix.txt")
        
    else:
      
        aligner.substitution_matrix = substitution_matrices.load(vars['git_dir'] + "matrix-nuc.txt")

    #Default Gap extension and opening penalties for ClustalW are 0.2 and 10.0.
    aligner.open_gap_score = -5.0 #-10.0
    aligner.extend_gap_score = -0.20
    #aligner.target_end_gap_score = 0.0
    #aligner.query_end_gap_score = 0.0
	
    alignments = aligner.align(first_seq, second_seq)
    #[first_seq_with_gaps, middle_line, second_seq_with_gaps] = (str(alignments[0])).split() # old Bio
    first_seq_with_gaps = ""
    second_seq_with_gaps = ""
    alignment_string = str(alignments[0])
    lines = alignment_string.split('\n')
    for line in lines:
        
        words = line.split()
        if len(words) > 2:
            
            if words[0] == "target":
                
                first_seq_with_gaps += words[2]
                
            elif words[0] == "query":
                
                second_seq_with_gaps += words[2]              
	
    matches = 0
    length_without_gaps = 0
    for i in range(len(first_seq_with_gaps)):
	
        if first_seq_with_gaps[i] != '-' and second_seq_with_gaps[i] != '-':
		
            length_without_gaps += 1
            if first_seq_with_gaps[i] == second_seq_with_gaps[i]:
		
                matches += 1

    identity = (matches * 100.0) / length_without_gaps
	
    if clustalw_aln == "":
	
        # we don't write the alignment
        return identity
		
    try:

        CLUSTALW_ALN = open(clustalw_aln, 'w')

    except:

        exit_on_error('sys_error', "write_alignment : could not open " + clustalw_aln + " for writing.")

    if seq_type == "SEQRES":

        CLUSTALW_ALN.write("target - Seqres sequence\nguery - Atom sequence\n\n" + alignment_string)

    else:

        CLUSTALW_ALN.write("target - MSA sequence\nguery - Atom sequence\n\n" + alignment_string)

    CLUSTALW_ALN.close()
    #write_alignment(first_seq_with_gaps, seq_type + "_SEQ", second_seq_with_gaps, "ATOM_SEQ", middle_line, clustalw_aln) # old Bio
    return(first_seq_with_gaps, second_seq_with_gaps, identity)


def pairwise_alignment_old(first_seq, second_seq, clustalw_aln = "", seq_type = ""):
	
	# for old Bio
    aligner = Align.PairwiseAligner()
	
    # Pairwise Alignment Paramaters
	
    aligner.mode = 'global' #Can be either global or local, if undetermined, biopython will choose optimal algorithem

    if form['DNA_AA'] == "AA":
        
        aligner.substitution_matrix = substitution_matrices.load(vars['git_dir'] + "matrix.txt")
        
    else:
      
        aligner.substitution_matrix = substitution_matrices.load(vars['git_dir'] + "matrix-nuc.txt")

    #Default Gap extension and opening penalties for ClustalW are 0.2 and 10.0.
    aligner.open_gap_score = -5.0 #-10.0
    aligner.extend_gap_score = -0.20
    #aligner.target_end_gap_score = 0.0
    #aligner.query_end_gap_score = 0.0
	
    alignments = aligner.align(first_seq, second_seq)
    #[first_seq_with_gaps, middle_line, second_seq_with_gaps] = (str(alignments[0])).split() # old Bio
    alignment_string = str(alignments[0])
    lines = alignment_string.split('\n')
    first_seq_with_gaps = lines[0]
    middle_line = lines[1]
    second_seq_with_gaps = lines[2]  
	
    matches = 0
    length_without_gaps = 0
    for i in range(len(first_seq_with_gaps)):
	
        if first_seq_with_gaps[i] != '-' and second_seq_with_gaps[i] != '-':
		
            length_without_gaps += 1
            if first_seq_with_gaps[i] == second_seq_with_gaps[i]:
		
                matches += 1

    identity = (matches * 100.0) / length_without_gaps
	
    if clustalw_aln == "":
	
        # we don't write the alignment
        return identity
		
    write_alignment(first_seq_with_gaps, seq_type + "_SEQ", second_seq_with_gaps, "ATOM_SEQ", middle_line, clustalw_aln) # old Bio
    return(first_seq_with_gaps, second_seq_with_gaps, identity)


    

class PDF(fpdf.FPDF):
    
    def __init__(self):
        
        super().__init__()
        self.lasth = 0 
        self.cbs = False
        
    def Cell(self, w, h = 0, txt = '', border = 0, ln = 0, align = '', fill = False):
        
        self.cell(w, h, txt, border, ln, align, fill)
        self.lasth = h

    def Print_4_Lines_Element(self, Rows_Pos, Score, AA, Pos, Solv_Acc, font_size, Insufficient_Data, Funct_Struct = ""):
    

        if Pos != "":
            
            x = self.get_x()
            if Pos < 9:
                
                self.set_xy(self.get_x() + 0.5, self.get_y())

            elif Pos < 99:
                
                self.set_xy(self.get_x() + 1, self.get_y())
                
            elif Pos < 9999:

                self.set_xy(self.get_x() + 1.5, self.get_y())
                
            elif Pos < 99999:
                
                self.set_xy(self.get_x() + 2, self.get_y())

            self.Print_BackgroundColor(str(Pos + 1), "", font_size, 5)
            self.set_xy(x, self.get_y())
            
        self.set_xy(self.get_x(), self.get_y() + self.lasth)
            
        self.Print_BackgroundColor(AA, 'B', font_size, Score, 3, Insufficient_Data)
        Col_Pos = self.get_x() # position on the line after printing
        self.set_xy(self.get_x() - 2.5, self.get_y() + self.lasth + 0.2)
        
        if Solv_Acc == "e":
            
            self.Print_ForegroundColor(Solv_Acc, 'B', font_size, 1)
            
        elif Solv_Acc == "b":
            
            self.Print_ForegroundColor(Solv_Acc, 'B', font_size, 2)
            
        self.set_xy(self.get_x() - 2, self.get_y() + self.lasth - 1.2)
        
        if Funct_Struct == "f":
            
            self.Print_ForegroundColor(Funct_Struct, 'B', font_size, 3)
            
        elif Funct_Struct == "s":
            
            self.Print_ForegroundColor(Funct_Struct, 'B', font_size, 4)
            
        self.set_xy(Col_Pos, Rows_Pos)


        
    def Print_BackgroundColor(self, txt, print_style, font_size, Color_Num, spacer = 2, isd = False):
    
        cbs = self.cbs
        
        if Color_Num == 0:
        
            self.set_text_color(0, 0, 0)
            self.set_fill_color(255, 255, 150)

        elif Color_Num == 1:
            
            if cbs:
                
                self.set_fill_color(10, 125, 130)
                
            else:
                
                self.set_fill_color(15, 90, 35)
                
            self.set_text_color(255, 255, 255)

        elif Color_Num == 2:
            
            if cbs:
                
                self.set_fill_color(75, 175, 190)
                
            else:
                
                self.set_fill_color(90, 175, 95)
        
        elif Color_Num == 3:
            
            if cbs:
                
                self.set_fill_color(165, 220, 230)
                
            else:
                
                self.set_fill_color(165, 220, 160)
    
        elif Color_Num == 4:
            
            if cbs:
                
                self.set_fill_color(215, 240, 240)
                
            else:
                
                self.set_fill_color(215, 240, 210)

        elif Color_Num == 5:
            
            self.set_fill_color(255, 255, 255)
                
        elif Color_Num == 6:
            
            if cbs:
                
                self.set_fill_color(250, 235, 245)
                
            else:
                
                self.set_fill_color(230, 210, 230)
                
        elif Color_Num == 7:
            
            if cbs:
                
                self.set_fill_color(250, 200, 220)
                
            else:
                
                self.set_fill_color(195, 165, 205)
                
        elif Color_Num == 8:
            
            if cbs:
                
                self.set_fill_color(240, 125, 170)
                
            else:
                
                self.set_fill_color(155, 110, 170)
                
        elif Color_Num == 9:
            
            if cbs:
                
                self.set_fill_color(160, 40, 95)
                
            else:
                
                self.set_fill_color(120, 40, 130)
                
            self.set_text_color(255, 255, 255)
            
        if isd:
            
            self.set_fill_color(255, 255, 150)
            
        self.set_font("Courier", print_style, font_size)
        width = len(txt) - 1 + spacer
        high = font_size / 2
        self.Cell(width, high, txt, 0, 0, "C", True)
        self.set_fill_color(255, 255, 255) # return to default background color (white)
        self.set_text_color(0, 0, 0) # return to default text color (black)
        
    def Print_NEW_Legend(self, IS_THERE_FUNCT_RES, IS_THERE_STRUCT_RES, IS_THERE_INSUFFICIENT_DATA, B_E_METHOD):
        
        self.set_font("", 'B', 12)
        font_size = 12
        self.ln()
        self.Cell(40, 10, "The conservation scale:", 0, 1)
        self.set_xy(18, self.get_y())
        #self.Print_BackgroundColor('?', "", 12, 0, 4, 1)
        self.Print_BackgroundColor('1', "", 12, 1, 4)
        self.Print_BackgroundColor('2', "", 12, 2, 4)
        self.Print_BackgroundColor('3', "", 12, 3, 4)
        #Average_X = self.get_x()
        self.Print_BackgroundColor('4', "", 12, 4, 4)
        self.Print_BackgroundColor('5', "", 12, 5, 4)
        self.Print_BackgroundColor('6', "", 12, 6, 4)
        self.Print_BackgroundColor('7', "", 12, 7, 4)
        #Conserved_X = self.get_x()
        self.Print_BackgroundColor('8', '', 12, 8, 4)
        self.Print_BackgroundColor('9', '', 12, 9, 4)
        self.ln()
        self.set_font("Times", 'B', 9.5)
        self.Cell(10, 6, "Variable", 0, 0, 'R', False)
        self.set_xy(28, self.get_y())
        self.Cell(15, 6, "Average", 0, 0, 'R', False)
        self.set_xy(53, self.get_y())
        self.Cell(15, 6, "Conserved", 0, 0, 'R', False)
        self.ln()
        self.ln()
    
        if B_E_METHOD != "no prediction":
            

            offset_1 = 0 
            offset_2 = 0
            if B_E_METHOD == "neural network algorithm":
                
                offset_1 = 63.5
                offset_2 = 62
                
            elif B_E_METHOD == "NACSES algorithm":
                
                offset_1 = 57.5
                offset_2 = 56


            self.Print_ForegroundColor('e', 'B', font_size, 1)
            self.set_xy(offset_1, self.get_y())
            self.Print_ForegroundColor(" - An exposed residue according to the %s." %B_E_METHOD, 'B', font_size, 9)
            self.ln()
            #self.set_y(self.get_y() + self.lasth + 5)
            #self.set_x(self.left_margin)
            self.Print_ForegroundColor('b', 'B', font_size, 2)
            self.set_xy(offset_2, self.get_y())
            self.Print_ForegroundColor(" - A buried residue according to the %s." %B_E_METHOD, 'B', font_size, 9)
            self.ln()
             
        #self.set_xy(45, self.get_y() + self.lasth + 5)
        #self.y += self.lasth + 5
        #self.x = self.left_margin
        if IS_THERE_FUNCT_RES:
            
            self.Print_ForegroundColor('f', 'B', font_size, 3)
            self.set_xy(64.5, self.get_y())
            self.Print_ForegroundColor(" - A predicted functional residue (highly conserved and exposed).", 'B', font_size, 9)
            self.ln()
            #self.set_y(self.get_y() + self.lasth + 5)
            #self.x = self.left_margin
            
        if IS_THERE_STRUCT_RES:
            
            self.Print_ForegroundColor('s', 'B', font_size, 4)
            self.set_xy(64, self.get_y())
            self.Print_ForegroundColor(" - A predicted structural residue (highly conserved and buried).", 'B', font_size, 9)
            self.ln()
            #self.set_y(self.get_y() + self.lasth + 5)
            #self.x = self.left_margin
            
        if IS_THERE_INSUFFICIENT_DATA:
            
            self.Print_BackgroundColor('x', 'B', font_size, 0, 2, 1)
            self.set_xy(58, self.get_y())
            self.Print_ForegroundColor(" - Insufficient data - the calculation for this site was", 'B', font_size, 9)
            self.ln()
            #self.set_y(self.get_y() + self.lasth + 4)
            #self.x = self.left_margin
            self.set_xy(48, self.get_y())
            self.Print_ForegroundColor("     performed on less than 10% of the sequences.",'B', font_size, 9)
            #self.set_y(self.get_y() + self.lasth + 5)
            #self.x = self.left_margin
            

    def Print_ForegroundColor(self, txt, print_style, font_size, Color, spacer = 2):
    
        if Color == 1: # orange
        
            self.set_text_color(255, 153, 0)
        
        elif Color == 2: # green
        
            self.set_text_color(0, 204, 0)
        
        elif Color == 3: # red

            self.set_text_color(255, 0, 0)
        
        elif Color == 4: # blue
    
            self.set_text_color(0, 0, 153)
        
        else: # black
        
            self.set_text_color(0, 0, 0)
        
        self.set_font("Courier", print_style, font_size)
        width = len(txt) - 1 + spacer
        high = font_size / 2
        self.Cell(width, high, txt, 0, 0, 'C', True)
        self.set_fill_color(255, 255, 255) # return to default background (white)
        self.set_text_color(0, 0, 0) # return to default text color (black)



    
def create_pdf_regular_or_cbs(cbs, name):
    
    #prediction_method = prediction_method.replace('-', ' ')
    pdf = PDF()
    pdf.add_page()
    #pdf.set_font("Times", "B", 20)
    pdf.add_font('DejaVu', '', '/content/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 20)
    pdf.Cell(0, 0, "ConSurf Results. Date:" + vars['date'], 0, 0, 'C')
    pdf.set_font("Times", "B", 20)
    pdf.set_y(pdf.get_y() + 10)        
    pdf.cbs = cbs
    Rows_Pos = 0
	
    """
    ConSurf_Grades = []
    Protein_Length = 0
    IS_THERE_INSUFFICIENT_DATA = False
    IS_THERE_FUNCT_RES = False
    IS_THERE_STRUCT_RES = False
    try:
        
        GRADES = open(vars['gradesPE'], 'r')
        
    except:
        
        exit_on_error("create_pdf: unable to open the file " + vars['gradesPE'] + " for reading.")
    
    line = GRADES.readline()
    while line != "":
        
        words = line.split()
        if len(words) > 5 and words[0].isnumeric():
            
            Protein_Length += 1
            details = {}
            
            details["AA"] = words[1]
            
            if '*' in words[color_column]:
                
                details["COLOR"] = 0
                IS_THERE_INSUFFICIENT_DATA = True
                
            else:
                
                details["COLOR"] = int(words[color_column])
                
            if B_E_column is not None:
                
                F_S_column = B_E_column + 1
                if words[B_E_column] == 'b' or words[B_E_column] == 'e':
                    
                    details["B_E"] = words[B_E_column]
                    IS_THERE_STRUCT_RES = True
                    if words[F_S_column] == 'f' or words[F_S_column] == 's':

    
                        details["F_S"] = words[F_S_column]
                        IS_THERE_FUNCT_RES = True
                    
                    else:
                    
                       details["F_S"] = ""
                    
                else:
                
                    details["B_E"] = ""
                    details["F_S"] = ""
                
            else:
    
                details["B_E"] = ""
                details["F_S"] = ""
                
            ConSurf_Grades.append(details)
            
        line = GRADES.readline()
        
    GRADES.close()
    """
    maxPosPerPage = 600
    for elem in vars['gradesPE_Output']:

        Pos = elem['POS'] - 1
        if vars['B/E']:
		
            prediction_method = "neural network algorithm"
            B_E = elem['B/E']
            F_S = elem['F/S'].strip()
			
        else:
		
            prediction_method = "no prediction"
            B_E = ""
            F_S = ""
			
        if Pos % maxPosPerPage == 0 and Pos != 0:
            
            pdf.add_page()
            
        if Pos % 50 == 0:
            
            pdf.ln()
            pdf.ln()
            pdf.ln()
            pdf.ln()
            
            Rows_Pos = pdf.get_y()

        elif Pos % 10 == 0:
            
            pdf.Print_ForegroundColor("", 'B', 10, 0, 4)
            
        if Pos % 10 == 0:
            
            pdf.Print_4_Lines_Element(Rows_Pos, elem['COLOR'], elem['SEQ'], Pos, B_E, 10, elem['ISD'], F_S)

        else:
            
            pdf.Print_4_Lines_Element(Rows_Pos, elem['COLOR'], elem['SEQ'], "", B_E, 10, elem['ISD'], F_S)
            
    pdf.ln()
    pdf.ln()
    pdf.ln()
    pdf.ln()
    pdf.Print_NEW_Legend(vars['B/E'], vars['B/E'], vars['insufficient_data'],  prediction_method)
    
    pdf.output(name)



def conseq_create_output():

    create_gradesPE(vars['gradesPE'])
    create_pdf()
    no_model_view()


def consurf_create_output():

																																																								
    r4s2pdb = {} # key: poistion in SEQRES/MSA, value: residue name with position in atom (i.e: ALA:22:A)

    if vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_pdb_msa_tree" or vars['SEQRES_seq'] != "":

        match_pdb_to_seq(r4s2pdb, vars['seqres_or_msa_seq_with_gaps'], vars['ATOM_seq_with_gaps'], vars['pdb_object'])

    else: # no seqres and no msa

        find_pdb_position(r4s2pdb, vars['pdb_object']) 
							

    identical_chains = find_identical_chains_in_PDB_file(vars['pdb_object'], form['PDB_chain'])

    atom_grades = {}
    create_gradesPE(vars['gradesPE'], r4s2pdb, vars['pdb_file_name'], form['PDB_chain'], vars['Used_PDB_Name'], vars['pdb_object'], identical_chains, vars['cif_or_pdb'], atom_grades)
    replace_TmpFactor_Consurf_Scores(atom_grades, form['PDB_chain'], vars['pdb_file_name'], vars['Used_PDB_Name']) # Create ATOMS section and replace the TempFactor Column with the ConSurf Grades (will create also isd file if relevant)

    create_pdf()
    


def extract_data_from_MSA():
    
    #vars['query_string'] = form['msa_SEQNAME']
    #vars['protein_seq_string'] = vars['MSA_query_seq']
        
    ## mode :  include msa and pdb

    if vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_pdb_msa_tree":

        compare_atom_seqres_or_msa("MSA")

def no_MSA():

    # if there is pdb : we compare the atom and seqres
    if vars['running_mode'] ==  "_mode_pdb_no_msa" and ('SEQRES_seq' in vars and len(vars['SEQRES_seq']) > 0):

        # align seqres and pdb sequences
        compare_atom_seqres_or_msa("SEQRES")

    vars['max_homologues_to_display'] = 500

    blast_hash = {}

    try:

        call_mmseqs2()

    except:

        os.chdir(vars['working_dir'])
        exit_on_error("calling mmseqs2 failed.")

    # choosing homologs, create fasta file for all legal homologs
    cd_hit_hash = {}
    #vars['hit_redundancy'] = float(form['MAX_REDUNDANCY']) # Now taken as argument from user #OLD: #CONSURF_CONSTANTS.FRAGMENT_REDUNDANCY_RATE
    vars['hit_overlap'] = 0.1
    #vars['min_num_of_hits'] = GENERAL_CONSTANTS.MINIMUM_FRAGMENTS_FOR_MSA
    vars['low_num_of_hits'] = 10
    vars['HITS_fasta_file'] = "query_homolougs.txt"
    vars['HITS_rejected_file'] = vars['job_name'] + "_rejected_homolougs.txt"

    choose_homologoues_from_search_with_lower_identity_cutoff(form['Homolog_search_algorithm'], len(vars['protein_seq_string']), vars['hit_redundancy'], vars['hit_overlap'], vars['hit_min_length'], float(form['MIN_IDENTITY']), vars['min_num_of_hits'], vars['BLAST_out_file'], vars['HITS_fasta_file'], vars['HITS_rejected_file'], blast_hash, form['DNA_AA'])

    vars['cd_hit_out_file'] = "query_cdhit.out"
    vars['unique_seqs'] = cluster_homologoues(cd_hit_hash)
    LOG.write("num_of_unique_seq: %d\n" %vars['unique_seqs'])
    add_sequences_removed_by_cd_hit_to_rejected_report(vars['cd_hit_out_file'] + ".clstr", vars['HITS_rejected_file'], vars['num_rejected_homologs'])
    choose_final_homologoues(blast_hash, cd_hit_hash, float(form['MAX_NUM_HOMOL']) -1, form['best_uniform_sequences'], vars['FINAL_sequences'], vars['HITS_rejected_file'], vars['num_rejected_homologs'])
    vars['zip_list'].append(vars['HITS_rejected_file'])


    if form['DNA_AA'] == "Nuc":

        # convert rna to dna

        LOG.write("convert_rna_to_dna(%s, %s)\n" %(vars['FINAL_sequences'], vars['FINAL_sequences'] + ".dna"))
        ans = convert_rna_to_dna(vars['FINAL_sequences'], vars['FINAL_sequences'] + ".dna")
        if ans[0] == "OK":

            vars['FINAL_sequences'] += ".dna"
            LOG.write("Seqs with u or U: " + str(ans[1]))
            for seq in ans[1]:

                print("Warnning: The seqeunce '" + seq + "' contains a 'U' replaced by 'T'")

        else:

            exit_on_error('sys_error', ans)

    LOG.write("make_sequences_file_HTML(%s, %s)\n" %(vars['FINAL_sequences'], vars['FINAL_sequences_html']))
    make_sequences_file_HTML(vars['FINAL_sequences'], vars['FINAL_sequences_html'])

    # we save to copies of the msa, one in fasta format and another in clustal format.
    #vars['msa_fasta'] = "msa_fasta.aln"
    #vars['msa_clustal'] = "msa_clustal.aln"
    create_MSA()
    vars['msa_SEQNAME'] = vars['query_string']
    
    print("%d homologues were collected." %vars['number_of_homologoues'])
    create_download_link(vars['FINAL_sequences_html'], "These %d are sequences used for creating the MSA." %vars['final_number_of_homologoues'])
    create_download_link(vars['HITS_rejected_file'], "Download the list of rejected homologues")

def call_mmseqs2():

    os.chdir(vars['root_dir'])

    msa_mode = "mmseqs2_uniref"
    pair_mode = "unpaired_paired"
    pairing_strategy = "greedy"
    result_dir = vars['job_name']

    csv_file = "/content/%s/%s.csv" %(vars['job_name'], vars['job_name'])

    CSV = open(csv_file, 'w')

    CSV.write("id,sequence\n%s,%s" %(vars['job_name'], vars['protein_seq_string']))
    CSV.close()

    result_dir = Path(result_dir)
    from colabfold.batch import get_msa_and_templates
    get_msa_and_templates(vars['job_name'], vars['protein_seq_string'], None, result_dir, msa_mode, False, None, pair_mode, pairing_strategy, 'https://api.colabfold.com', 'colabfold/google-colab-main')

    os.chdir(vars['working_dir'])


def extract_data_from_model():



    if vars['cif_or_pdb'] == "pdb":

        vars['pdb_object'] = pdbParser()

    else:

        vars['pdb_object'] = cifParser()

    vars['pdb_object'].read(vars['pdb_file_name'], form['PDB_chain'], form['DNA_AA'])


    #[vars['SEQRES_seq'], vars['ATOM_seq'], vars['ATOM_without_X_seq']] = get_seqres_atom_seq(vars['pdb_object'], form['PDB_chain'], vars['pdb_file_name'])
    vars['SEQRES_seq'] = vars['pdb_object'].get_SEQRES()
    All_atoms = vars['pdb_object'].get_ATOM_withoutX()
    if form['PDB_chain'] in All_atoms:
        
        vars['ATOM_without_X_seq'] = All_atoms[form['PDB_chain']]
        
    else:
        
        exit_on_error('user_error', "The chain is not in the PDB. Select the PDB chain using the flag --chain")
        
    analyse_seqres_atom()

    try:

        FAS = open(vars['protein_seq'], 'w')

    except:

        exit_on_error('sys_error',"cannot open the file " + vars['protein_seq'] + " for writing!")

    # we write the sequence to a fasta file for the homologues search
    # we save the name of the quey string for rate4site
    if vars['SEQRES_seq'] == "":

        vars['query_string'] = "Input_seq_ATOM_" + form['PDB_chain']
        vars['protein_seq_string'] = vars['ATOM_without_X_seq']
        FAS.write(">" + vars['query_string'] + "\n" + vars['ATOM_without_X_seq'])

    else:

        vars['query_string'] = "Input_seq_SEQRES_" + form['PDB_chain']
        vars['protein_seq_string'] = vars['SEQRES_seq']
        FAS.write(">" + vars['query_string'] + "\n" + vars['SEQRES_seq'])

    FAS.close()


def create_cd_hit_output(input_file, output_file, cutoff, ref_cd_hit_hash, type):


    seq = ""
    seq_name = ""
    cmd = "" 
    n = 0

    # running cd-hit

    if type == "AA":

        cmd += "%scd-hit -i %s -o %s " %(vars['cd_hit_dir'], input_file, output_file)
        if cutoff > 0.7 and cutoff < 1:

            n = 5

        elif cutoff > 0.6 and cutoff <= 0.7:

            n = 4

        elif cutoff > 0.5 and cutoff <= 0.6:

            n = 3

        elif cutoff > 0.4 and cutoff <= 0.5:

            n = 2
            
    else:

        # DNA
        cmd += "cd-hit-est -i %s -o %s " %(input_file, output_file)
        if cutoff > 0.9 and cutoff < 1:

            n = 8

        elif cutoff > 0.88 and cutoff <= 0.9:

            n = 7

        elif cutoff > 0.85 and cutoff <= 0.88:

            n = 6

        elif cutoff > 0.8 and cutoff <= 0.85:

            n = 5

        elif cutoff > 0.75 and cutoff <= 0.8:

            n = 4
            
    cmd += "-c %f -n %d -d 0" %(cutoff, n)

    submit_job_to_Q("CD-HIT", cmd)

    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:

        exit_on_error("sys_error", "create_cd_hit_output : " + str(cmd) + ": CD-HIT produced no output!\n")

    num_cd_hits = 0

    try:
	
        CDHIT_OUTPUT = open(output_file, 'r')
		
    except:
	
        exit_on_error('sys_error', "create_cd_hit_output : could not open the file " + output_file + " for writing.")
        
    # inserting chosen homologues to a hash
    line = CDHIT_OUTPUT.readline()
    seq_name = ""
    seq = ""
    while line != "":

        line = line.strip()
        if line[0] == ">":
		
            seq_name = line[1:]
			
        else:
		
            seq = line
            if not seq_name in ref_cd_hit_hash:

                num_cd_hits += 1
                ref_cd_hit_hash[seq_name] =seq
			
        line = CDHIT_OUTPUT.readline()

    CDHIT_OUTPUT.close()	   
    return num_cd_hits

	

def make_sequences_file_HTML(plain_txt_sequences, HTML_sequences):

    try:

        HTML_SEQUENCES = open(HTML_sequences, 'w')

    except:

        exit_on_error('sys_error', "make_sequences_file_HTML : cannot open the file " + HTML_sequences + " for writing.")

    try:

        TXT_SEQUENCES = open(plain_txt_sequences, 'r')

    except:

        exit_on_error('sys_error', "make_sequences_file_HTML : cannot open the file " + plain_txt_sequences + " for reading.")
	
    counter = 1	
    line = TXT_SEQUENCES.readline()
    while line != "":

        line = line.strip()
        if line == "":
            
            line = TXT_SEQUENCES.readline()
            continue
        
        if line[0] != ">":
            
            counter += 1		
            HTML_SEQUENCES.write("<FONT FACE=\"courier new\" SIZE=3>" + line + "</FONT><BR>\n")
            
        else:
            
            line = line[1:]
            if line[:9] == "Input_seq":
                    
                HTML_SEQUENCES.write("<FONT FACE=\"courier new\" SIZE=3>>%d_%s</FONT><BR>\n" %(counter, line))
                    
            else:
                
                name = line.split("|")[0]            
                HTML_SEQUENCES.write("<FONT FACE=\"courier new\" SIZE=3><A HREF=\"https://www.uniprot.org/uniref/%s\">>%d_%s</A></FONT><BR>\n" %(name, counter, line))

        line = TXT_SEQUENCES.readline()
        
    HTML_SEQUENCES.close()
    TXT_SEQUENCES.close()


def convert_rna_to_dna(Seqs, Seqs_dna):

    # replace the u with t and return the sequences names replaced

    try:

        OUT = open(Seqs_dna, 'w')

    except:

        return("convert_rna_to_dna: Can't open file " + Seqs_dna + " for writing.")

    try:

        SEQS = open(Seqs, 'r')

    except:

        return("convert_rna_to_dna: Can't open file " + Seqs + " for reading.")

    Seqs_Names = []
    seq_name = ""

    line = SEQS.readline()
    while line != "":

        line = line.rstrip()
        match1 = re.match(r'^>(.*)', line)
        if match1:

            seq_name = match1.group(1)


        elif 'u' in line or 'U' in line:

            Seqs_Names.append(seq_name)
            line = line.replace('u', 't')
            line = line.replace('U', 'T')

        OUT.write(line + "\n")
        line = SEQS.readline()

    OUT.close()
    SEQS.close()

    return("OK", Seqs_Names)


def create_pdf():

    create_pdf_regular_or_cbs(True, vars['Colored_Seq_PDF'])
    create_pdf_regular_or_cbs(False, vars['Colored_Seq_CBS_PDF'])


def create_pymol(input, prefix):

    cmd = "pymol -qc " + input + " -d \"run " + vars['pymol_color_script_isd'] + "\"\n"
    cmd += "pymol -qc " + input + " -d \"run " + vars['pymol_color_script_CBS_isd'] + "\"\n"

    LOG.write("create_pymol : %s\n" %cmd)
    submit_job_to_Q("PYMOL", cmd)

    pymol_session = "consurf_pymol_session.pse"
    pymol_session_CBS = "consurf_CBS_pymol_session.pse"

    if os.path.exists(pymol_session) and os.path.getsize(pymol_session) != 0:

        os.chmod(pymol_session, 0o664)
        os.rename(pymol_session, prefix + pymol_session)
        vars['zip_list'].append(prefix + pymol_session)

    if os.path.exists(pymol_session_CBS) and os.path.getsize(pymol_session_CBS) != 0:

        os.chmod(pymol_session_CBS, 0o664)
        os.rename(pymol_session_CBS, prefix + pymol_session_CBS)
        vars['zip_list'].append(prefix + pymol_session_CBS)


def create_chimera(input, prefix):
 
    run_chimera(input, prefix + "consurf_chimerax_session.cxs", vars['chimera_color_script'])
    run_chimera(input, prefix + "consurf_CBS_chimerax_session.cxs", vars['chimera_color_script_CBS'])

def run_chimera(input, output, script):

    cmd = "chimerax --nogui --script '%s %s %s' --exit\n" %(script, input, output)
    LOG.write("create_chimera : %s\n" %cmd)
    submit_job_to_Q("CHIMERA", cmd)

    vars['zip_list'].append(output)

"""
def check_msa_tree_match(ref_msa_seqs):

    ref_tree_nodes = []
    check_validity_tree_file(ref_tree_nodes)
    LOG.write("check_msa_tree_match : check if all the nodes in the tree are also in the MSA\n")

    for node in ref_tree_nodes:

        if not node in ref_msa_seqs:

            exit_on_error('user_error', "The uploaded tree file is inconsistant with the uploaded MSA file. The node '" + node + "' is found in the tree file, but there is no sequence in the MSA file with that exact name. Note that the search is case-sensitive!")

    LOG.write("check_msa_tree_match : check if all the sequences in the MSA are also in the tree\n")

    for seq_name in ref_msa_seqs: #check that all the msa nodes are in the tree

        if not seq_name in ref_tree_nodes:

            exit_on_error('user_error', "The uploaded MSA file is inconsistant with the uploaded tree file. The Sequence name '" + seq_name + "' is found in the MSA file, but there is no node with that exact name in the tree file. Note that the search is case-sensitive!")

    vars['unique_seqs'] = len(ref_msa_seqs)
    LOG.write("There are " + str(vars['unique_seqs']) + " in the MSA.\n")
"""
    
def get_info_from_msa(seq_names):
    
    # returns the name of the query sequence and fills the input array with the names of the sequences
    query_seq = ""
    msa_format = check_msa_format(vars['user_msa_file_name'])
    try:
        
        alignment = AlignIO.read(vars['user_msa_file_name'], msa_format)
        
    except:
        
        exit_on_error('sys_error', "get_info_from_msa : can't open the file " + vars['user_msa_file_name'] + " for reading.")
        
    try:
                
        MSA = open(vars['working_dir'] + vars['msa_fasta'], 'w')
                
    except:
                
        exit_on_error('sys_error', "get_info_from_msa : can't open the file " + vars['msa_fasta'] + " for writing.")
        
    for record in alignment:
        
        new_seq_name = str(record.id)
        if new_seq_name in seq_names:
			
            exit_on_error('user_error', "The sequence %s appears more than once in the MSA." %new_seq_name)
            
            
        seq_names.append(new_seq_name)
                
        seq = str(record.seq)
        if form['msa_SEQNAME'] == new_seq_name:
            
            query_seq = seq

                
        if form['DNA_AA'] == "Nuc" and ('u' in seq or 'U' in seq):
                
            seq = seq.replace('u', 't')
            seq = seq.replace('U', 'T')
            print("Warnning: The seqeunce '" + new_seq_name + "' contains a 'U' replaced by 'T'")
            
        MSA.write(">%s\n%s\n" %(new_seq_name, seq))
                
    MSA.close()
    
       
    num_of_seq = len(seq_names)
    vars['unique_seqs'] = num_of_seq
    vars['final_number_of_homologoues'] = num_of_seq
    LOG.write("MSA contains " + str(num_of_seq) + " sequences\n")
    if num_of_seq < 5:

        exit_on_error('user_error',"The MSA file contains only " + str(num_of_seq) + " sequences. The minimal number of homologues required for the calculation is 5.")
 
    query_seq = query_seq.replace("-", "")
    query_seq = query_seq.upper()
    
    if query_seq == "":
        
        exit_on_error('user_error', "The query sequence is not in the msa. Please choose the name of the query sequence by adding the flag --query")
        
    vars['msa_SEQNAME'] = form['msa_SEQNAME']
    #vars['query_string'] = form['msa_SEQNAME']
    vars['MSA_query_seq'] = query_seq
    """
    # there is no input seq, use msa seq instead
    vars['protein_seq_string'] = vars['MSA_query_seq']
    try:

        QUERY_FROM_MSA = open(vars['working_dir'] + vars['protein_seq'], 'w')

    except:

        exit_on_error('sys_error', "get_info_from_msa : Could not open %s for writing." %vars['protein_seq'])

    QUERY_FROM_MSA.write(">" + form['msa_SEQNAME'] + "\n")
    QUERY_FROM_MSA.write(vars['MSA_query_seq'] + "\n")
    QUERY_FROM_MSA.close()
    """
    

def create_pipe_file(pipeFile, pipeFile_CBS, seq3d_grades, seq3d_grades_isd, isd_residue_color_ArrRef, no_isd_residue_color_ArrRef, pdb_file_name, user_chain, IN_pdb_id_capital, identical_chains, pdb_object):

    # CREATE PART of PIPE
    partOfPipe = "partOfPipe"
    partOfPipe_CBS = "partOfPipe_CBS"
	
    length_of_seqres = pdb_object.get_num_known_seqs()
    length_of_atom = pdb_object.get_num_known_atoms()

    LOG.write("create_part_of_pipe_new(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n" %(partOfPipe, vars['unique_seqs'], "user_DB", "seq3d_grades_isd", "seq3d_grades", length_of_seqres, length_of_atom, "isd_residue_color_ArrRef", "no_isd_residue_color_ArrRef", form['E_VALUE'], form['ITERATIONS'], form['MAX_NUM_HOMOL'], form['MSAprogram'], form['ALGORITHM'], form['SUB_MATRIX'], "legacy"))
    create_part_of_pipe_new(partOfPipe, vars['unique_seqs'], "user_DB", seq3d_grades_isd, seq3d_grades, length_of_seqres, length_of_atom, isd_residue_color_ArrRef, no_isd_residue_color_ArrRef, form['E_VALUE'], form['ITERATIONS'], form['MAX_NUM_HOMOL'], form['MSAprogram'], form['ALGORITHM'], form['SUB_MATRIX'], vars['Average pairwise distance'], "legacy")
					  																									 																															

    # create the color blind friendly version
    create_part_of_pipe_new(partOfPipe_CBS, vars['unique_seqs'], "user_DB", seq3d_grades_isd, seq3d_grades, length_of_seqres, length_of_atom, isd_residue_color_ArrRef, no_isd_residue_color_ArrRef, form['E_VALUE'], form['ITERATIONS'], form['MAX_NUM_HOMOL'], form['MSAprogram'], form['ALGORITHM'], form['SUB_MATRIX'], vars['Average pairwise distance'], "cb")
																																

    LOG.write("extract_data_from_pdb(%s)\n" %pdb_file_name)
    header_pipe = extract_data_from_pdb(pdb_file_name)
										   

    # GET THE FILE NAMES
    msa_filename = ""
    msa_query_seq_name = ""
    if vars['user_msa_file_name'] is not None:

        msa_filename = vars['user_msa_file_name']
        msa_query_seq_name = form['msa_SEQNAME']

    tree_filename = ""
    if form['tree_name'] is not None:

        tree_filename = vars['tree_file']

    # GET THE CURRENT TIME
    completion_time = str(datetime.now().time())
    run_date = str(datetime.now().date())

    # USE THE CREATED PART of PIPE to CREATE ALL THE PIPE TILL THE PDB ATOMS (DELETE THE PART PIPE)
    LOG.write("create_consurf_pipe_new(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n" %(vars['working_dir'], IN_pdb_id_capital, user_chain, "header_pipe", pipeFile, identical_chains, partOfPipe, vars['working_dir'], form['Run_Number'], msa_filename, msa_query_seq_name, tree_filename, vars['submission_time'], completion_time, run_date))
    create_consurf_pipe_new(vars['working_dir'], IN_pdb_id_capital, user_chain, header_pipe, pipeFile, identical_chains, partOfPipe, vars['working_dir'], form['Run_Number'], msa_filename, msa_query_seq_name, tree_filename, vars['submission_time'], completion_time, run_date)
				   																								
    # USE THE CREATED PART of PIPE to CREATE ALL THE PIPE TILL THE PDB ATOMS (DELETE THE PART PIPE) - Color friendly version
    LOG.write("create_consurf_pipe_new(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n" %(vars['working_dir'], IN_pdb_id_capital, user_chain, "header_pipe", pipeFile_CBS, identical_chains, partOfPipe_CBS, vars['working_dir'], form['Run_Number'], msa_filename, msa_query_seq_name, tree_filename, vars['submission_time'], completion_time, run_date))
    create_consurf_pipe_new(vars['working_dir'], IN_pdb_id_capital, user_chain, header_pipe, pipeFile_CBS, identical_chains, partOfPipe_CBS, vars['working_dir'], form['Run_Number'], msa_filename, msa_query_seq_name, tree_filename, vars['submission_time'], completion_time, run_date)
				   

																										

    # Add the PDB data to the pipe
    LOG.write("add_pdb_data_to_pipe(%s, %s)\n" %(pdb_file_name, pipeFile))
    add_pdb_data_to_pipe(pdb_file_name, pipeFile)
				   																								 																																		   

    # Add the PDB data to the pipe - color blind version
    LOG.write("add_pdb_data_to_pipe(%s, %s)\n" %(pdb_file_name, pipeFile_CBS))
    add_pdb_data_to_pipe(pdb_file_name, pipeFile_CBS)
				   																							 	

def compare_atom_to_query(Query_seq, ATOM_seq, pairwise_aln, PDB_Name):

    # in case there are both seqres and atom fields, checks the similarity between the 2 sequences.

    [first_seq, second_seq, score] = pairwise_alignment(Query_seq, ATOM_seq, pairwise_aln, "QUERY")
    return(first_seq, second_seq)


def upload_protein_sequence():


    if os.path.exists(vars['uploaded_Seq']) and os.path.getsize(vars['uploaded_Seq']) != 0: # file fasta uploaded

        try:

            UPLOADED = open(vars['uploaded_Seq'], 'r')

        except:

            exit_on_error('sys_error', "upload_protein_sequence : Cannot open the file " + vars['protein_seq'] + "for writing!")

        protein_seq_string = UPLOADED.read()
        UPLOADED.close()

        if protein_seq_string.count('>') > 1:

            exit_on_error('user_error', "The protein input <a href = \"%s\">sequence</a> contains more than one FASTA sequence. If you wish to upload MSA, please upload it as a file." %protein_seq_string)

        # delete sequence name and white spaces
        protein_seq_string = re.sub(r'>.*\n', "", protein_seq_string)
        protein_seq_string = re.sub(r'\s', "", protein_seq_string)

    else:

        exit_on_error('sys_error', 'upload_protein_sequence : no user sequence.')

    # we write the sequence to a file for the homologues search
    try:

        UPLOADED = open(vars['working_dir'] + vars['protein_seq'], 'w')

    except:			

        exit_on_error('sys_error', "upload_protein_sequence : Cannot open the file " + vars['protein_seq'] + "for writing!")

    UPLOADED.write(">Input_seq\n" + protein_seq_string)
    UPLOADED.close()

    amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "X"]
    nucleic_acids = ["A", "C", "G", "T", "U", "N"]
    if form['DNA_AA'] == "AA":

        amino = False
        for char in protein_seq_string:
		
            if not char.upper() in amino_acids:
			
                exit_on_error('user_error', "The input sequence contains the illegal character %s." %char)
				
            elif not char.upper() in nucleic_acids:
			
                amino = True
				
        if not amino:

            exit_on_error('user_error',"It seems that the protein input is only composed of Nucleotides (i.e. :A,T,C,G). Please note that you chose to run the server based on amino acids sequnce and not DNA / RNA sequence.<br />You may translate your sequence to amino acids and resubmit your query, or alternatively choose to analyze nucleotides.<br />")

    else:


        for char in protein_seq_string:
		
            if not char.upper() in amino_acids:
			
                exit_on_error('user_error', "The input sequence contains the illegal character %s." %char)
				
            elif not char.upper() in nucleic_acids:
			
                exit_on_error('user_error',"It seems that the input sequence contains the Amino Acid %s. Please note that you chose to run the server based on nucleotides sequnce and not protein sequence.<br />You may resubmit your query and choose to analyze Amino Acids.<br />" %char)

    vars['protein_seq_string'] = protein_seq_string
    vars['query_string'] = "Input_seq" # name of the sequence is saved for rate4site


def array_to_string(array):

    string = ""
    for word in array:
	
        string += " " +word
		
    return string


def zip_all_outputs():

    zipObj = ZipFile(vars['All_Outputs_Zip'], 'w')
	
    for file in vars['zip_list']:
	    
        if os.path.exists(file):
		
            zipObj.write(file)	

    zipObj.close()
    create_download_link(vars['All_Outputs_Zip'], "Download zip with all the files.")

def find_identical_chains_in_PDB_file(pdb_Object, query_chain):

    # Looking in the PDB for chains identical to the original chain
    ATOMS = pdb_Object.get_ATOM_withoutX()

    # string with identical chains
    identicalChains = query_chain

    # looking for chains identical to the original chain
    for chain in ATOMS:

        if query_chain != chain:

            other_seq = ATOMS[chain]
            query_seq = ATOMS[query_chain]
            chain_length = len(other_seq)
            OrgChain_length = len(query_seq)

            # if length not similar, skip
            if min(OrgChain_length, chain_length)/max(OrgChain_length, chain_length) <= 0.9:

                continue

            # compare the two chains 
            try:
			
                if pairwise_alignment(other_seq, query_seq) > 0.95:
			
                    identicalChains += " " + chain
					
            except:
			
                LOG.write("find_identical_chains_in_PDB_file: Error comparing the chains %s and %s\n" %(query_chain, chain))

    return identicalChains



def replace_TmpFactor_Rate4Site_Scores(chain, pdb_file, gradesPE, pdb_file_with_score_at_TempFactor):

    # This will create a PDB file that contains the Rate4Site scores instead of the TempFactor Column

    Rate4Site_Grades = {}
    LOG.write("read_Rate4Site_gradesPE(%s, %s)\n" %(gradesPE, str(Rate4Site_Grades)))
    read_Rate4Site_gradesPE(gradesPE, Rate4Site_Grades)
										   
    if vars['cif_or_pdb'] == "pdb":

        LOG.write("replace_TmpFactor_Rate4Site_Scores_PDB(%s, %s, %s, %s)\n" %(pdb_file, chain, "Rate4Site_Grades", pdb_file_with_score_at_TempFactor))
        replace_TmpFactor_Rate4Site_Scores_PDB(pdb_file, chain, Rate4Site_Grades, pdb_file_with_score_at_TempFactor)

    else:
																														 
        LOG.write("replace_TmpFactor_Rate4Site_Scores_CIF(%s, %s, %s, %s)\n" %(pdb_file, chain, "Rate4Site_Grades", pdb_file_with_score_at_TempFactor))
        replace_TmpFactor_Rate4Site_Scores_CIF(pdb_file, chain, Rate4Site_Grades, pdb_file_with_score_at_TempFactor)



def replace_TmpFactor_Consurf_Scores(atom_grades, chain, pdb_file, prefix):

    # This Will create a File containing the ATOMS records with the ConSurf grades instead of the TempFactor column

    if vars['cif_or_pdb'] == "pdb":

        LOG.write("replace_TmpFactor_Consurf_Scores_PDB(atom_grades, %s, %s, %s);\n" %(chain, pdb_file, prefix))
        replace_TmpFactor_Consurf_Scores_PDB(atom_grades, chain, pdb_file, prefix)
		
    else:																																												 
        LOG.write("replace_TmpFactor_Consurf_Scores_CIF(atom_grades, %s, %s, %s);\n" %(chain, pdb_file, prefix))
        replace_TmpFactor_Consurf_Scores_CIF(atom_grades, chain, pdb_file, prefix)
								   
	
def install_rate4site(rate4site_dir, rate4site_slow_dir):

    LOG.write("Installing rate4site.\n")
	
    # create directory for rate4site
    submit_job_to_Q("download_rate4site", "git clone https://github.com/barakav/r4s_for_collab.git")

    # create directory for rate4site slow
    shutil.copytree(rate4site_dir, rate4site_slow_dir)

    # make rate4site
    submit_job_to_Q("install_rate4site", "cd %s\nmake\nchmod 755 rate4site" %rate4site_dir)
    
    # change the make file 
    os.remove(rate4site_slow_dir + "Makefile") 
    os.rename(rate4site_slow_dir + "Makefile_slow", rate4site_slow_dir + "Makefile") 

    # make rate4site
    submit_job_to_Q("install_rate4site", "cd %s\nmake\nchmod 755 rate4site" %rate4site_slow_dir)

def run_rate4site():

    rate4site_dir = "/content/r4s_for_collab/" 
    rate4site_slow_dir = "/content/r4s_for_collab_slow/"
    vars['r4s_log'] = "r4s.log" # log file
    vars['r4s_out'] = "r4s.res" # output file

    #install_rate4site(rate4site_dir, rate4site_slow_dir)

    MatrixHash = {'JTT' : '-Mj', 'MTREV' : '-Mr', 'CPREV' : '-Mc', 'WAG' : '-Mw', 'DAYHOFF' : '-Md', 'T92' : '-Mt', 'HKY' : '-Mh', 'GTR' : '-Mg', 'JC_NUC' : '-Mn', 'JC_AA' : '-Ma', 'LG' : '-Ml'}

    params = "rate4site -a '%s' -s %s -zn %s -bn -o %s -v 9" %(vars['query_string'], vars['msa_fasta'], MatrixHash[(form['SUB_MATRIX']).upper()], vars['r4s_out'])
    if vars['running_mode'] == "_mode_pdb_msa_tree" or vars['running_mode'] == "_mode_msa_tree":

        params += " -t %s" %vars['tree_file']

    if form['ALGORITHM'] == "Bayes":

        params +=  " -ib"

    else:

        params +=  " -im"

    r4s_comm = rate4site_dir + params + " -l " + vars['r4s_log']

    LOG.write("run_rate4site : running command: %s\n" %r4s_comm)
    print("The conservation scores are being calculated. Please wait.")
    submit_job_to_Q("rate4site", r4s_comm)

    # if the run failed - we rerun using the slow verion
    if check_if_rate4site_failed(vars['r4s_log']):

        LOG.write("run_rate4site : The run of rate4site failed. Sending warning message to output.\nThe same run will be done using the SLOW version of rate4site.\n")
        #print("Warning: The given MSA is very large, therefore it will take longer for ConSurf calculation to finish. The results will be sent to the e-mail address provided.<br>The calculation continues nevertheless.")
        vars['r4s_log'] = "r4s_slow.log"
        r4s_comm = rate4site_slow_dir + params + " -l " + vars['r4s_log']
        LOG.write("run_rate4site : running command: %s\n" %str(r4s_comm))
        submit_job_to_Q("rate4siteSlow", r4s_comm)

        if check_if_rate4site_failed(vars['r4s_log']):

            exit_on_error('sys_error', "Both rate4site and rate4site slow failed.")

    extract_diversity_matrix_info(vars['r4s_log'])
    
    if not os.path.exists(vars['tree_file']):
        
        os.rename("TheTree.txt", vars['tree_file'])


def run_rate4site_old():
    
    rate4s = vars['script_dir'] + "/rate4site_bioseq/rate4site" 
    rate4s_ML = vars['script_dir'] + "/rate4site_bioseq/rate4site.24Mar2010"
    rate4s_slow = vars['script_dir'] + "/rate4site_bioseq/rate4site.doubleRep"

    vars['r4s_log'] = "r4s.log" # log file
    vars['r4s_out'] = "r4s.res" # output file
    MatrixHash = {'JTT' : '-Mj', 'MTREV' : '-Mr', 'CPREV' : '-Mc', 'WAG' : '-Mw', 'DAYHOFF' : '-Md', 'T92' : '-Mt', 'HKY' : '-Mh', 'GTR' : '-Mg', 'JC_NUC' : '-Mn', 'JC_AA' : '-Ma', 'LG' : '-Ml'}

    params = " -a '%s' -s %s -zn %s -bn -o %s" %(vars['query_string'], vars['msa_fasta'], MatrixHash[(form['SUB_MATRIX']).upper()], vars['r4s_out'])
    if vars['running_mode'] == "_mode_pdb_msa_tree" or vars['running_mode'] == "_mode_msa_tree":

        params += " -t %s" %vars['tree_file']
		
    if form['ALGORITHM'] == "Bayes":

        params += " -ib -n 32 -v 9" 
        r4s_comm = rate4s + params

    else:

        params += " -im -v 9"
        r4s_comm = rate4s_ML + params 	

    r4s_comm += " -l " + vars['r4s_log']
    LOG.write("run_rate4site : running command: %s\n" %r4s_comm)
    submit_job_to_Q("rate4site", r4s_comm)

    # if the run failed - we rerun using the slow verion
    if check_if_rate4site_failed(vars['r4s_log']):

        vars['r4s_log'] = "r4s_slow.log"
        LOG.write("run_rate4site : The run of rate4site failed. Sending warning message to output.\nThe same run will be done using the SLOW version of rate4site.\n")
        print("Warning: The given MSA is very large, therefore it will take longer for ConSurf calculation to finish. The results will be sent to the e-mail address provided.<br>The calculation continues nevertheless.")
        r4s_comm = rate4s_slow + params + " -l " + vars['r4s_log']
        LOG.write("run_rate4site : running command: %s\n" %r4s_comm)
        submit_job_to_Q("rate4siteSlow", r4s_comm)

        if check_if_rate4site_failed(vars['r4s_log']):

            exit_on_error('sys_error', "Both rate4site and rate4site slow failed.")

    extract_diversity_matrix_info(vars['r4s_log'])

def find_best_substitution_model():

    try:
	
        # convert fasta to phylip
        msa_phy_filepath = "input_msa.phy"
        #convert_msa_format(vars['msa_fasta'], "fasta", msa_phy_filepath, "phylip-relaxed")
        AlignIO.convert(vars['msa_fasta'], "fasta", msa_phy_filepath, "phylip-relaxed")
        os.chmod(vars['msa_fasta'], 0o644)
        os.chmod(msa_phy_filepath, 0o644)

        if form['DNA_AA'] == "Nuc":

            run_jmt(msa_phy_filepath)

        else:

            run_prottest(msa_phy_filepath)
				
    except:
	 
        vars['best_fit'] = "model_search_failed"
        form['SUB_MATRIX'] = "JTT"
        print("The evolutionary model search has failed. The JTT model is chosen by default.")


def run_prottest(msa_file_path):

    output_file_path = vars['job_name'] + "_model_selection.txt"
    cmd = "java -jar %s -log disabled -i %s -AICC -o %s -S 1 -JTT -LG -MtREV -Dayhoff -WAG -CpREV -threads 1" %(vars['prottest'], msa_file_path, output_file_path)
    submit_job_to_Q("protest", cmd)
    LOG.write("run_protest: %s\n" %cmd)

    f = open(output_file_path, 'r')

    match = re.search(r"(?<=Best model according to AICc: ).*", f.read())
    f.close()
    if match:

        vars['best_fit'] = "model_found"
        model = match.group()
        model = model.strip('()')
        print("The best evolutionary model was selected to be: " + model)
        create_download_link(output_file_path, "See details")
        vars['zip_list'].append(output_file_path)

    else:
	
        vars['best_fit'] = "model_search_failed"
        model = "JTT"
        print("The evolutionary model search has failed. The JTT model is chosen by default.")

    form['SUB_MATRIX'] = model

def run_jmt(msa_file_path):

    JMT_JAR_FILE = GENERAL_CONSTANTS.JMODELTEST2
    output_file_path = "model_selection.txt"
    cmd = "java -Djava.awt.headless=true -jar %s -d %s -t BIONJ -AICc -f -o %s" %(JMT_JAR_FILE, msa_file_path, output_file_path)
    submit_job_to_Q("jmt", cmd)
    LOG.write("run_jmt: %s\n" %cmd)

    f = open(output_file_path, 'r')

    start_reading = False
    JMT_VALID_MODELS = ["JC","HKY","GTR"]

    # extract best model from table
    line = f.readline()
    while line != "":

        if start_reading:

            line = line.strip()
            split_row = line.split()
            model = split_row[0]
            if model in JMT_VALID_MODELS:

                f.close()
                #model = model.strip('()')
                if model == "JC":

                    form['SUB_MATRIX'] = "JC_Nuc"
					
                else:
				
                    form['SUB_MATRIX'] = model
				
                vars['best_fit'] = "model_found"
                print("The best evolutionary model was selected to be: " + model)
                return				

        elif re.search(r'Model             -lnL    K     AICc       delta       weight   cumWeight', line, re.M):

            start_reading = True

        line = f.readline()

    vars['best_fit'] = "model_search_failed"
    form['SUB_MATRIX'] = "JC_Nuc"
    print("The evolutionary model search has failed. The JC model is chosen by default")
    f.close()


def convert_msa_format(infile, infileformat, outfile, outfileformat):

    try:

        AlignIO.convert(infile, infileformat, outfile, outfileformat)

    except:

        exit_on_error('sys_error', "convert_msa_format : exception")

def create_MSA():

    if form['MSAprogram'] == "CLUSTALW":

        cmd = "clustalw -infile=%s -outfile=%s" %(vars['FINAL_sequences'], vars['msa_clustal'])
        LOG.write("create_MSA : run %s\n" %cmd)
        submit_job_to_Q("clustalw", cmd)
        convert_msa_format(vars['msa_clustal'], "clustal", vars['msa_fasta'], "fasta")

    elif form['MSAprogram'] == "MAFFT":

        cmd = "mafft --localpair --maxiterate 1000 --quiet %s > %s" %(vars['FINAL_sequences'], vars['msa_fasta'])
        LOG.write("create_MSA : run %s\n" %cmd)
        submit_job_to_Q("MAFFT", cmd)
        #convert_msa_format(vars['msa_fasta'], "fasta", vars['msa_clustal'], "clustal")

    elif form['MSAprogram'] == "PRANK":

        cmd = "prank -d=%s -o=%s -F" %(vars['FINAL_sequences'], vars['msa_fasta'])
        print("Warning: PRANK is accurate but slow MSA program, please be patient.")
        LOG.write("create_MSA : run %s\n" %cmd)
        submit_job_to_Q("PRANK", cmd)

        if os.path.exists(vars['msa_fasta'] + ".2.fas"):

            vars['msa_fasta'] += ".2.fas"

        elif os.path.exists(vars['msa_fasta'] + ".1.fas"):

            vars['msa_fasta'] += ".1.fas"

        elif os.path.exists(vars['msa_fasta'] + ".best.fas"):

            vars['msa_fasta'] +=  ".best.fas"

        #convert_msa_format(vars['msa_fasta'], "fasta", vars['msa_clustal'], "clustal")

    elif form['MSAprogram'] == "MUSCLE":

        #cmd = "muscle -align %s -output %s" %(vars['FINAL_sequences'], vars['msa_fasta'])
        cmd = "muscle -in %s -out %s -quiet" %(vars['FINAL_sequences'], vars['msa_fasta'])
        LOG.write("create_MSA : run %s\n" %cmd)
        submit_job_to_Q("MUSCLE", cmd)
        #convert_msa_format(vars['msa_clustal'], "clustal", vars['msa_fasta'], "fasta")
        
    else:
        
        exit_on_error('user_error', "Choose one of the programs for creating the msa: clustalw, mafft, prank or muscle.")

    if not os.path.exists(vars['msa_fasta']) or os.path.getsize(vars['msa_fasta']) == 0:
	
        exit_on_error('user_error', "The %s program failed to create the MSA. Choose a different program to create the MSA." %form['MSAprogram'])

def choose_final_homologoues(ref_to_seqs_hash, ref_to_cd_hash, max_num_homologs, witch_unifrom, output_file, rejected_file, num_rejected_homologs):

    LOG.write("sort_sequences_from_eval(%s ,%s , %f, %s, %s, %s, %d)\n" %("ref_to_seqs_hash", "ref_to_cd_hash", max_num_homologs, witch_unifrom, output_file, rejected_file, num_rejected_homologs))
										 

    query_name = ""
    query_AAseq = ""
    counter = 1

    
    try:

        FINAL = open(vars['FINAL_sequences'], 'w')

    except:

        exit_on_error('sys_error',"choose_final_homologoues : cannot open the file %s for writing" %vars['FINAL_sequences'])

    # we write the query sequence to the file of the final homologs
    FINAL.write(">%s\n%s\n" %(vars['query_string'], vars['protein_seq_string']))
    
    final_file_size = os.path.getsize(vars['FINAL_sequences']) # take the size of the file before we add more sequences to it
    
    try:

        REJECTED = open(rejected_file, 'a')

    except:

        exit_on_error('sys_error', "Can't open '" + rejected_file + "' for writing.")

    # write query details
    if query_AAseq != "":

        FINAL.write(">%s\n%s\n" %(query_name, query_AAseq))

    size_cd_hit_hash = len(ref_to_cd_hash)
    uniform = 1
    jump = 1

    if witch_unifrom == "sample":

        uniform = int(size_cd_hit_hash / max_num_homologs)
        if uniform == 0:

            uniform = 1

    final_number_of_homologoues = 1
    REJECTED.write("\n\tSequences rejected because of the requirement to select only %d representative homologs\n\n" %(max_num_homologs + 1))
    # write homologs
    for s_name in sorted(ref_to_seqs_hash.keys(), key = ref_to_seqs_hash.get):

        # write next homolog
        if s_name in ref_to_cd_hash: # and 'SEQ' in ref_to_cd_hash[s_name]:

            if counter != jump or counter > max_num_homologs * uniform:
																																																																			   
																																																								   

                counter += 1
                REJECTED.write("%d %s\n" %(num_rejected_homologs, s_name))
                num_rejected_homologs += 1
                continue

            final_number_of_homologoues += 1  
            FINAL.write(">%s\n%s\n" %(s_name, ref_to_cd_hash[s_name]))
            counter += 1
            jump += uniform

    FINAL.close()
    REJECTED.close()

    vars['final_number_of_homologoues'] = final_number_of_homologoues
    # check that more sequences were added to the file
    if not final_file_size < os.path.getsize(vars['FINAL_sequences']):

        exit_on_error('sys_error', "choose_final_homologoues : the file " + vars['FINAL_sequences'] + " doesn't contain sequences")

def cluster_homologoues(ref_cd_hit_hash):

    msg = ""
    LOG.write("cluster_homologoues : create_cd_hit_output(%s, %s, %f, %s, %s);\n" %(vars['HITS_fasta_file'], vars['cd_hit_out_file'], vars['hit_redundancy']/100, ref_cd_hit_hash, form['DNA_AA']))
    total_num_of_hits = create_cd_hit_output(vars['HITS_fasta_file'], vars['cd_hit_out_file'], vars['hit_redundancy']/100, ref_cd_hit_hash, form['DNA_AA'])

    if form['MAX_NUM_HOMOL'].upper() == 'ALL':

        form['MAX_NUM_HOMOL'] = total_num_of_hits

    if total_num_of_hits < vars['min_num_of_hits']: # less seqs than the minimum: exit

        if total_num_of_hits <= 1:

            msg = "There is only 1 "

        else:

            msg = "There are only %d " %total_num_of_hits

        msg += "unique hits. The minimal number of sequences required for the calculation is %d. You may try to: " %vars['min_num_of_hits']
        msg += "Re-run the server with a multiple sequence alignment file of your own. Increase the Evalue. Decrease the Minimal %ID For Homologs"

        if int(form['ITERATIONS']) < 5:

            msg += " Increase the number of " + form['Homolog_search_algorithm'] + " iterations."

        msg += "\n"
        exit_on_error('user_error',msg)

    elif total_num_of_hits + 1 < vars['low_num_of_hits']: # less seqs than 10 : output a warning.

        msg = "Warning: There are "

        if total_num_of_hits + 1 < vars['number_of_homologoues_before_cd-hit']: # because we will add the query sequence itself to all the unique sequences.

            msg += "%d hits, only %d of them are" %(vars['number_of_homologoues_before_cd-hit'], total_num_of_hits+1)

        else:

            msg += str(total_num_of_hits + 1)

        msg += " unique sequences. The calculation is performed on the %d unique sequences, but it is recommended to run the program with a multiple sequence alignment file containing at least %s sequences." %(total_num_of_hits + 1, vars['low_num_of_hits'])

    else:

        msg = "There are %d %s hits. %d of them are unique, including the query. The calculation is performed on " %(vars['number_of_homologoues_before_cd-hit'], form['Homolog_search_algorithm'], total_num_of_hits + 1)

        if total_num_of_hits <= int(form['MAX_NUM_HOMOL']):

            msg += "%d unique sequences." %(total_num_of_hits + 1)

        elif form['best_uniform_sequences'] == "best":

            msg += "the %s <a href=\"<?=$orig_path?>/%s\" style=\"color: #400080; text-decoration:underline;\">sequences</a> closest to the query (with the lowest E-value)." %(form['MAX_NUM_HOMOL'], vars['FINAL_sequences_html'])

        else:

            msg += "a sample of %s sequences that represent the list of homologues to the query." %form['MAX_NUM_HOMOL']
            #msg += "a sample of %s <a href=\"<?=$orig_path?>/%s\" style=\"color: #400080; text-decoration:underline;\">sequences</a> that represent the list of homologues to the query." %(form['MAX_NUM_HOMOL'], vars['FINAL_sequences_html'])

    print(msg)

    #if os.path.exists(vars['HITS_rejected_file']) and os.path.getsize(vars['HITS_rejected_file']) != 0:

        #print_message_to_output("Here is the <a href=\"<?=$orig_path?>/" + vars['HITS_rejected_file'] + "\" TARGET=Rejected_Seqs style=\"color: #400080; text-decoration:underline;\">list of sequences</a> that produced significant alignments, but were not chosen as hits.")
        #print_message_to_output("<a href=\"<?=$orig_path?>/" + vars['HITS_rejected_file'] + "\" TARGET=Rejected_Seqs style=\"color: #400080; text-decoration:underline;\">Click here</a> if you wish to view the list of sequences which produced significant alignments, but were not chosen as hits.")

    return(total_num_of_hits + 1)



def submit_job_to_Q(job_name_prefix, cmd):

    os.system("cd %s\n%s" %(vars['working_dir'], cmd))
    os.system("dmesg | tail -40")

    
def compare_atom_seqres_or_msa(what_to_compare):

    # in case there is a msa and pdb, we check the similarity between the atom and the msa sequences
    # in case there is no msa and there are both atom and seqres sequences, we check the similarity between them

    pairwise_aln = "PDB_" + what_to_compare + ".aln"
    atom_length = len(vars['ATOM_without_X_seq'])
    alignment_score = 0
    other_query_length = len(vars['protein_seq_string'])
    query_line = {}
    atom_line = "sequence extracted from the ATOM field of the PDB file"
    query_line['SEQRES'] = "sequence extracted from the SEQRES field of the PDB file"
    query_line['MSA'] = "sequence extracted from the MSA file"

    # compare the length of sequences. output a message accordingly
    if other_query_length != 0 and other_query_length < atom_length:

        print("The %s is shorter than the %s. The %s sequence has %d residues and the ATOM sequence has %d residues. The calculation continues nevertheless." %(query_line[what_to_compare],atom_line ,what_to_compare, other_query_length, atom_length))

    if atom_length < other_query_length:

        if atom_length < other_query_length * 0.2:

            print("Warning: The %s is significantly shorter than the %s. The %s sequence has %d residues and the ATOM sequence has only %d residues. The calculation continues nevertheless." %(atom_line, query_line[what_to_compare], what_to_compare, other_query_length, atom_length))

        else:

            print("The %s is shorter than the %s. The %s sequence has %d residues and the ATOM sequence has %d residues. The calculation continues nevertheless." %(atom_line, query_line[what_to_compare], what_to_compare, other_query_length, atom_length))

    # match the sequences 
    LOG.write("compare_atom_seqres_or_msa : Align ATOM and " + what_to_compare + " sequences\n")
    [vars['seqres_or_msa_seq_with_gaps'], vars['ATOM_seq_with_gaps'], alignment_score] = pairwise_alignment(vars['protein_seq_string'], vars['ATOM_without_X_seq'], pairwise_aln, what_to_compare)

    if alignment_score < 100:

        if alignment_score < 30:

            exit_on_error('user_error',"The Score of the alignment between the %s and the %s is ONLY %d%% identity.<br>See <a href=\"<?=$orig_path?>/%s\" style=\"color: #400080; text-decoration:underline;\" TARGET=PairWise_Align>pairwise alignment</a>." %(query_line[what_to_compare], atom_line, alignment_score, pairwise_aln))

        else:

            print("The Score of the alignment between the %s and the %s is %d%% identity. The calculation continues nevertheless." %(query_line[what_to_compare], atom_line, alignment_score))



def exit_on_error(which_error, error_msg):

    complete_msg = "\n\nEXIT on error:\n\n" + error_msg + "\n"
    LOG.write(complete_msg)
    print(complete_msg)
    raise Exception("The error is not an exception")


def analyse_seqres_atom():

    # there is no ATOM field in the PDB

    if vars['ATOM_without_X_seq'] == "":

        exit_on_error('user_error', "There is no ATOM derived information in the PDB file.<br>Please refer to the OVERVIEW for detailed information about the PDB format.")

    # there is no SEQRES field in the PDB

    if vars['SEQRES_seq'] == "":

        msg = "Warning: There is no SEQRES derived information in the PDB file. The calculation will be based on the ATOM derived sequence. "

        if vars['running_mode'] == "_mode_pdb_no_msa":

            msg += "If this sequence is incomplete, we recommend to re-run the server using an external multiple sequence alignment file, which is based on the complete protein sequence."

        LOG.write("analyse_seqres_atom : There is no SEQRES derived information in the PDB file.\n")
        print(msg)

    if form['DNA_AA'] == "AA":

        # check if seqres contains nucleic acid
        if vars['pdb_object'].get_type() == "Nuc":

            exit_on_error('user_error', "The selected chain: " + form['PDB_chain'] + " contains nucleic acid, and you have selected amino acid")

    else:

        # check if seqres contains amino acid
        #type_SEQRES = vars['pdb_object'].get_type_SEQRES()
        #if form['PDB_chain'] in type_SEQRES and type_SEQRES[form['PDB_chain']] == "AA":
        if vars['pdb_object'].get_type() == "AA":

            exit_on_error('user_error', "The selected chain: " + form['PDB_chain'] + " contains amino acid, and you have selected nucleic acid")

    # if modified residues exists, print them to the screen

    MAXIMUM_MODIFIED_PERCENT = 0.15
    MODIFIED_COUNT = vars['pdb_object'].get_MODIFIED_COUNT()
    if MODIFIED_COUNT > 0:
        
        if form['DNA_AA'] == "AA":

            if len(vars['SEQRES_seq']) > 0 and MODIFIED_COUNT / len(vars['SEQRES_seq']) > MAXIMUM_MODIFIED_PERCENT:

                LOG.write("MODIFIED_COUNT %d\nSEQRES_seq %s\n" %(MODIFIED_COUNT, vars['SEQRES_seq']))
                exit_on_error('user_error', "Too many modified residues were found in SEQRES field; %0.3f%% of the residues are modified, the maximum is %0.3f%%." %(MODIFIED_COUNT / len(vars['SEQRES_seq']) ,MAXIMUM_MODIFIED_PERCENT))

            LOG.write("analyse_seqres_atom : modified residues found\n")
            print("Please note: Before the analysis took place, modified residues read from SEQRES field were converted back to the original residues:\n" + vars['pdb_object'].get_MODIFIED_LIST() + ".")

        else:

            LOG.write("analyse_seqres_atom : modified residues found\n")
            print("Please note: Before the analysis took place, modified nucleotides read from SEQRES field were converted back to the original nucleotides:\n" + vars['pdb_object'].get_MODIFIED_LIST() + ".")




def get_seqres_atom_seq(PDB_Obj, query_chain, pdb_file_name, model = False):

    # extract the sequences from the pdb

    seqres = PDB_Obj.get_SEQRES() # seqres sequence
    atom = PDB_Obj.get_ATOM() # atom sequence with gaps filled with X for amino acids and N for nucleic acids. This is only used to find the residues' place in the pdb
    atom_without_X = PDB_Obj.get_ATOM_withoutX()[query_chain] # atom sequence with gaps not filled 
									 
    if seqres == "" and atom == "":

        if model:

            return('user_error', "The protein sequence for chain '%s' was not found in SEQRES nor ATOM fields in the <a href=\"%s\">PDB file</a>." %(query_chain, pdb_file_name), "1")

        else:

            exit_on_error('user_error', "The protein sequence for chain '%s' was not found in SEQRES nor ATOM fields in the PDB file." %query_chain)

    return [seqres, atom, atom_without_X]


def run_consurf(root_dir, git_dir, job_name, mode, substitution_model, rate4site_algorithm):
    
    # Use HTML and CSS to style the input box
    display(HTML("""
    <style>
        .container { width: 100% !important; }
        .CodeMirror, .output_subarea, .output_area { max-width: 100% !important; }
        .input { width: 100% !important; }
    </style>
    """))
    
    global vars
    global form
    global LOG

    vars['root_dir'] = root_dir
    vars['git_dir'] = git_dir
    #os.chdir(vars['root_dir'])

    # check if directory already exists
    i = 1
    temp_job_name = job_name
    while os.path.isdir(temp_job_name):

        temp_job_name = job_name + "_" + str(i)
        i += 1

    vars['job_name'] = temp_job_name
    vars['working_dir'] = vars['root_dir'] + "/" + vars['job_name'] + "/"

    # create job directory
    os.mkdir(vars['job_name'])

    # directories for programs installed
    vars['prottest'] = vars['root_dir'] + "/prottest-3.4.2/prottest-3.4.2.jar"
    vars['cd_hit_dir'] = vars['root_dir'] + "/cdhit/"
    vars['rate4site_dir'] = vars['root_dir'] + "/r4s_for_collab/"
    vars['rate4site_slow_dir'] = vars['root_dir'] + "/r4s_for_collab_slow/"

    vars['pymol'] = "pymol.py"
    vars['pymol_CBS'] = "pymol_CBS.py"
    
    install()
    os.chdir(vars['job_name'])
    print("\n")

    shutil.copy(vars['git_dir'] + vars['pymol'], vars['pymol'])
    shutil.copy(vars['git_dir'] + vars['pymol_CBS'], vars['pymol_CBS'])

    choose_mode(mode)
    
    vars['tree_file'] = vars['job_name'] + "_Tree.txt"
    vars['msa_fasta'] = vars['job_name'] + "_msa_fasta.aln" # msa copy in fasta format
    
    if substitution_model == "choose the best substitution model using prottest":

        form['SUB_MATRIX'] = "BEST"

    else:

        form['SUB_MATRIX'] = substitution_model
        
    if rate4site_algorithm == "bayesian":

        form['ALGORITHM'] = "Bayes"

    else:

        form['ALGORITHM'] = "Maximum"
        
    vars['run_log'] = "log.txt"
    form['Homolog_search_algorithm'] = "MMseqs2"
    form['DNA_AA'] = "AA"


    LOG = open(vars['run_log'], 'w')

    vars['BLAST_out_file'] = "%s/%s_all/uniref.a3m" %(vars['working_dir'], vars['job_name'])



    vars['Msa_percentageFILE'] = vars['job_name'] + "_msa_aa_variety_percentage.csv"


    vars['All_Outputs_Zip'] = "Consurf_Outputs_" + vars['job_name'] + ".zip"


    vars['hit_min_length'] = 0.60 # minimum length of homologs
    vars['min_num_of_hits'] = 5 # minimum number of homologs
    vars['FINAL_sequences'] = "query_final_homolougs.fasta" # finial homologs for creating the MSA
    vars['FINAL_sequences_html'] = vars['job_name'] + "_final_homolougs.html" # html files showing the finial homologs to the user
    vars['r4s_log'] = "r4s.log" # rate4site log
    vars['r4s_out'] = "r4s.res" # rate4site output
    vars['r4s_slow_log'] = "r4s_slow.log" # rate4site slow log
    vars['gradesPE'] = vars['job_name'] + "_consurf_grades.txt" # file with consurf output
    vars['zip_list'] = []
    vars['date'] = date.today().strftime("%d/%m/%Y")

    vars['Colored_Seq_PDF'] = vars['job_name'] + "_consurf_colored_seq.pdf"
    vars['Colored_Seq_CBS_PDF'] = vars['job_name'] + "_consurf_colored_seq_CBS.pdf"
    vars['color_array'] = {1 : "#0A7D82", 2 : "#4BAFBE", 3 : "#A5DCE6", 4 : "#D7F0F0", 5 : "#FFFFFF", 6 : "#FAEBF5", 7 : "#FAC8DC", 8 : "#F07DAA", 9 : "#A0285F", 'ISD' : "#FFFF96"}
    vars['color_array_CBS'] = {1 : "#0F5A23", 2 : "#5AAF5F", 3 : "#A5DCA0", 4 : "#D7F0D2", 5 : "#FFFFFF", 6 : "#E6D2E6", 7 : "#C3A5CD", 8 : "#9B6EAA", 9 : "#782882", 'ISD' : "#FFFF96"}

    vars['msa_clustal'] = "msa_clustal.aln" # if the file is not in clustal format, we create a clustal copy of it

    vars['protein_seq'] = "protein_seq.fas" # a fasta file with the protein sequence from PDB or from protein seq input

    vars['gradesPE_Output'] = [] # an array to hold all the information that should be printed to gradesPE
    # in each array's cell there is a hash for each line from r4s.res.
    # POS: position of that aa in the sequence ; SEQ : aa in one letter ;
    # GRADE : the given grade from r4s output ; COLOR : grade according to consurf's scale




    vars['zip_list'].append(vars['tree_file'])
    vars['zip_list'].append(vars['gradesPE'])
    vars['zip_list'].append(vars['Msa_percentageFILE'])
    vars['zip_list'].append(vars['Colored_Seq_PDF'])
    vars['zip_list'].append(vars['Colored_Seq_CBS_PDF'])
    vars['zip_list'].append(vars['msa_fasta'])
    vars['zip_list'].append(vars['pymol'])
    vars['zip_list'].append(vars['pymol_CBS'])
    vars['zip_list'].append(vars['FINAL_sequences_html'])


    ## mode : include pdb

    # create a pdbParser, to get various info from the pdb file
    if vars['running_mode'] == "_mode_pdb_no_msa" or vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_pdb_msa_tree":

        upload_PDB()
        extract_data_from_model()

    
    ## mode : only protein sequence

    # if there is only protein sequence: we upload it.
    elif vars['running_mode'] == "_mode_no_pdb_no_msa":

        upload_sequence()
        #upload_protein_sequence()
    
    ## mode : no msa - with PDB or without PDB

    if vars['running_mode'] == "_mode_pdb_no_msa" or vars['running_mode'] == "_mode_no_pdb_no_msa":

        create_MSA_parameters()
        no_MSA()

    ## mode : include msa

    elif vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_msa" or vars['running_mode'] == "_mode_pdb_msa_tree" or vars['running_mode'] == "_mode_msa_tree":
        
        upload_MSA()
        extract_data_from_MSA()

    if form['SUB_MATRIX'] == "BEST":

        #vars['best_fit'] = True
        find_best_substitution_model()

    else:

        vars['best_fit'] = "model_chosen"

    run_rate4site()
    assign_colors_according_to_r4s_layers()
    write_MSA_percentage_file()

    ## mode : include pdb

    if vars['running_mode'] == "_mode_pdb_no_msa" or vars['running_mode'] == "_mode_pdb_msa" or vars['running_mode'] == "_mode_pdb_msa_tree":

        consurf_create_output()
	
    ## mode : ConSeq - NO PDB

    if vars['running_mode'] == "_mode_msa" or vars['running_mode'] == "_mode_no_pdb_no_msa" or vars['running_mode'] == "_mode_msa_tree":

        conseq_create_output()

    zip_all_outputs()
    create_download_link(vars['msa_fasta'], "Download Multiple Sequence Alignment in FASTA format")
    create_download_link(vars['gradesPE'], "Download Per-Residue Details")
    create_download_link(vars['Msa_percentageFILE'], "Download csv file showing residue variety per position in the MSA")
    create_download_link(vars['tree_file'], "Download Phylogenetic Tree")

    ## Arrange The HTML Output File

    print("The calculation is done.")
    LOG.close()
    
def run_consurf_on_colab(root_dir, git_dir, job_name, mode, substitution_model, rate4site_algorithm):
    
    try:
        
        run_consurf(root_dir, git_dir, job_name, mode, substitution_model, rate4site_algorithm)
    
    except Exception as e:

        if str(e) != "The error is not an exception":
   
            LOG.write(traceback.format_exc())
            LOG.close()
            raise(e)
        
        else:
        
            LOG.close()

