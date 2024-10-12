import streamlit as st
import streamlit.components.v1 as  components
# import numpy as np
import pandas as pd
from pandas import DataFrame
#from tabulate import tabulate
# import matplotlib.pyplot as plt
import ruptures as rpt
# import csv
import os 
from io import StringIO

import v0

import v0_0_1
import v0_1
import v2_1

##### Password #####

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "What's Radia's Favorite phrase ? No CAPS nor Spaces", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "What's Radia's Favorite phrase ? No CAPS nor Spaces", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

### Code ###

def main():
    st.title("Mechanical Test")
     
    ## Session State Initialisation
    
    if 'version' not in st.session_state:
        st.session_state.version = "v0.0.1"
    
    ## File management and Version
    
    st.header("CSV Browser")
    uploaded_files = st.file_uploader("Choose a file",accept_multiple_files=True,type="csv",)
    
    st.header("Version & Specimen")    
    
    ## Choose a Version
    with st.container():
        Version_col, Specimen_col = st.columns([1,1])
    with Version_col:
        VersionList = ("v0", "v0.0.1" , "v0.1", "v2" , "v2.1")
        VersionIndex = 4
        for i in range(len(VersionList)):
            if VersionList == st.session_state:
                        VersionIndex = i                
        Version = st.radio("Version",VersionList,index=VersionIndex)
        st.session_state.version = Version
        with st.expander("What does this version do ?"):
            if Version == VersionList[0]:
                st.write("This code is used to analyze materials with the following shape of curve and characteristics:")
                # st.image("CurveType1.png")
                st.write('The materials usually having this type of curve are natural leather, scaffolds, Faircraft constructs, FC Leather, non woven backings, etc… \n This version of the code analyzes automatically all the data of all the specimens in a folder, using predefined settings, unchangeable by the user and taken by default.')
            elif Version == VersionList[1]:
                st.write("This code is used to analyze materials with the following shape of curve and characteristics:")
                # st.image("CurveType1.png")
                st.write("The materials usually having this type of curve are natural leather, scaffolds, Faircraft constructs, FC Leather, non woven backings, etc…\n This version of the code analyzes automatically all the data of all the specimens in a folder, using predefined settings, unchangeable by the user and taken by default.\n The results obtained here are the Young’s modulus calculated as directed in the norm ISO 527-1, the tensile strength, the stress at tear, the tear load and the percentage of elongation at break.")

            elif Version == VersionList[2]:
                st.write("This code is used to analyze materials with the following shape of curve and characteristics:")
                # st.image("CurveType2.png")
                st.write("The materials usually having this type of curves are natural leather, scaffolds, Faircraft constructs, FC Leather, non woven backings, etc… \n This version of the code allows the user to manually set some variables in order to analyze curves and data more accurately. Each specimen will be run alone, and the settings will be adapted by the user according to the need. ")
                
                
            elif Version == VersionList[4]:
                st.write("This code is used to analyze materials with the following shape of curve and characteristics:")
                # st.image("CurveType2.png")
                st.write('The materials usually having this type of curve are knitted fabrics & woven fabric.\n This version of the code allows the user to manually set some variables in order to analyze curves and data more accurately. Each specimen will be run alone, and the settings will be adapted by the user according to the need')
    
    ## Choose a file to display
    if uploaded_files is not None:
        with Specimen_col:
            NumberOfSpecimen = len(uploaded_files)
            FileList = [uploaded_files[i-1].name for i in range(1,NumberOfSpecimen+1)]
            if st.session_state.version == VersionList[1]: # v0.0.1
                FileList += ["All"]
                FileList += ["To Google Sheet"]
            FileName = st.radio("Which sample you want to analyze ?", FileList)
        
        
        if FileName == "To Google Sheet":
            v0_0_1.SheetCode(uploaded_files)
            
            
        
        for uploaded_file in uploaded_files:
            if FileName == uploaded_file.name:
                if st.session_state.version == VersionList[0]: # v0
                    v0.code(uploaded_file)
                if st.session_state.version == VersionList[1]: # v0.0.1
                    v0_0_1.code(uploaded_file)
                if st.session_state.version == VersionList[2]: # v0.1
                    v0_1.code(uploaded_file)
                if st.session_state.version == VersionList[3]: # v2
                    v2_1.code(uploaded_file)
                if st.session_state.version == VersionList[4]: # v2.1
                    v2_1.code(uploaded_file)
            elif FileName == "All":
                v0_0_1.code(uploaded_file)
                    
            
        # if st.session_state.version == VersionList[1]:
        #     if FileName == "All":
        #         for uploaded_file in uploaded_files:
                    
    return

### Run ###

if check_password():
    main()





            
            
            
        
