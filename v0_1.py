import streamlit as st
import streamlit.components.v1 as  components
import numpy as np
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate
import matplotlib.pyplot as plt
import ruptures as rpt
import csv
import os 
from io import StringIO

###### Explainations #####




v0_1_explainations={'TearLoadSelectionMode':"**Default TearLoad Selection**: it's automatic \n\n **Multiple Tear Load Selection Range**: The part of the curve kept to calculate the tear load is asked \n \n **Manual Tear Load**: The index of the point  where the failure occurs and the tear load is calculated is entered by the user",
                    'WithChangeOfElasticLimit': "**No-Default Elastic Limit and Young’s Modulus**: The way the Young’s modulus and the elastic limit is calculated is the same as the one used in the code v0 \n \n **Yes-Manual Elastic Limit and Young’s Modulus**: The zone of the curve used to calculate the Young’s modulus and the elastic limit is set by the user",
                    'YMAndELSelectionRange' : "**Manual Elastic Limit and Young’s Modulus**: The zone of the curve used to calculate the Young’s modulus and the elastic limit is set by the user",
                    'MultipleTearLoadSelectionRange' : "**Tear Load** Select the part of the curve kept to calculate the tear load"}


###### Extracting the data from the csv files ######

# NB: Streamlit is currently seing the download csv_file as bytes_data
@st.cache_data
def getInformations(csv_file): 
    bytes_data = csv_file.read() ## Read the csv_file
    my_reader = bytes_data.decode("utf-8") ## Decoding the bytes_data
    t= np.zeros([len(my_reader)-11, 6]) ## array of data 
    i=0 # Row index
    for row in my_reader.splitlines():
        #Storing the name of the Sample (Echantillon)
        if i==0:
            SampleName=row.split(";")[1][1:-1]
            # st.write(SpecimenName)
            
        #Storing the name of the Specimen (Eprouvette), its thickness, its width and its length
        if i==7:
            if csv_file.name[-5] != row.split(";")[0]: # Checking is the sample number in the file_name is the one store in the csv file
                st.write("This is not the correct specimen. Please check the number")
            SpecimenThickness=float(row.split(";")[2][1:-1].replace(",","."))
            SpecimenWidth=float(row.split(";")[3][1:-1].replace(",","."))
            SpecimenLength=float(row.split(";")[4][1:-1].replace(",","."))
        #Storing the headers
        if i==10:
            Headers=row.split(";")
        #Storing the values
        if i>10 :
            if len(row)>0:
                for j in range(6):
                    t[i-11,j]=float(row.split(";")[j][1:-1].replace(",","."))
        i+=1
    # extracting the strain and stress
    x=t[:,4]
    y=t[:,3]
    
    # Creating dataframes
    df_SpecimenSpecs = pd.DataFrame(data= {'Specimen Name': [SampleName+"_"+ my_reader.splitlines()[7].split(";")[0]],
                                    'Specimen Thickness': [SpecimenThickness],
                                    'Specimen Width': [SpecimenWidth],
                                    'Specimen Length' : [SpecimenLength]
                                    })    
    df_data = pd.DataFrame(t,columns=Headers)
    
    return  df_SpecimenSpecs, df_data, SampleName


###### Analyzing the data ######




def process(df_SpecimenSpecs, df_data, SampleName): 
  
    #Extracting data from both DataFrames 
    SpecimenName = df_SpecimenSpecs['Specimen Name'].values[0]
    SpecimenNumber = SpecimenName[-1]
    SpecimenThickness = df_SpecimenSpecs['Specimen Thickness'].values[0]
    SpecimenWidth = df_SpecimenSpecs['Specimen Width'].values[0]
    SpecimenLength = df_SpecimenSpecs['Specimen Length'].values[0]
    x=df_data.values[:,4]
    y=df_data.values[:,3]
    
    ##Display Settings
    font= 7
    
    #Choosing if you want to run as défault or not
    st.header("Mode")
    
    col1,col2,col3= st.columns([5,1,5])
    
    Answer = ('No','Yes')    
    
    # with col1:
        
    #     Default = st.select_slider('Run as default?',options=['Yes','No'])
    # with col2:
    #     st.write("")
    #     with st.expander("Help?"):
    #         st.write("explanations to add")
            
            
    ######### widget Tear Load ###########
    
    TearLoadModes = ['Default','Multiple','Manual'] 
    
    with col1: 
        DTL_MSR_MTL = st.select_slider('Tear Load Selection Mode',options=TearLoadModes , value = 'Default')
        with st.expander("Help?"):
            st.write(v0_1_explainations['TearLoadSelectionMode'])
            container=st.container()

    with col3:
        st.write(" ")
        st.write(" ELC ")
        ELC = st.toggle('With change of elastic limit? ')
        with st.expander("Help?"):
            st.write(v0_1_explainations['WithChangeOfElasticLimit'])
            container=st.container()
    
    ############################### Yes ####################################       
    if DTL_MSR_MTL == TearLoadModes[0]: #Default=='Yes':
        if not ELC:
            #All default
            #data processing
            
            curveSelectionrange=3          #Sections of the curve for Young Modulus and Elastic Limit. Default: 3.
            polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
            cutcurvefite = 5               #Percent of left points on each side. Default 5%.
            tearSelectionRange=2           #Sections of the curve for tensile strenght. Default: 2, second half.
            correctionBeforeBreak=1        #Correction of the value before tearing. Default 1. Method to be improved.
    
            while (x[-1]==0) and (y[-1]==0) :
                x=np.delete (x, -1, 0)
                y=np.delete (y, -1, 0)
            #Shear Stress vs Strain plot
            
            #Figures            
            fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10)) 
            ax1.plot(x,y)
            ax1.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize= font)
            ax1.set_xlabel('Strain (%)',fontsize= font)
            ax1.set_ylabel('Stress (MPa)',fontsize= font)
        
        
            #Get the first elastic limit and young's modulus
            x50 = x[1:round(len(x)/curveSelectionrange)]
            y50 = y[1:round(len(y)/curveSelectionrange)]
            ax2.plot(x50,y50,'x')
            
            #Polyfit of the selected values
            coeffs1_50 = np.polyfit(x50, y50, polynomialFit)    #Get the fit. First order poylnomial order x. TO BE IMPROVED.
        
            # Get the x values for the fit at higher resolution
            xFit50 = np.linspace(x50[1], x50[len(x50)-1])
            # Get the estimated y values
            yFit50 = np.polyval(coeffs1_50, xFit50)
        
            #Plot them as a line
            ax2.plot(xFit50, yFit50, linewidth=3)
            ax2.set_title('Fitted data: Stress vs Strain', fontsize=font)
            ax2.set_xlabel('Strain (%)', fontsize=font)
            ax2.set_ylabel('Stress (MPa)',fontsize=font)
                    
            #Finding the first elastic limit and calculating the first Young's modulus
        
            #Selection of the values to be kept for the curves fitting
            #---------------------------------------------------------------------------------------------------------------------
            index1 = round((cutcurvefite/100) * len(xFit50))        #Percent of the curve to be discarded on each side for the fit
            index2 = round(((100-cutcurvefite)/100) * len(xFit50))  #Percent of the curve to be kept for the fit
            #index1=0
            #index2=len(xFit50)
            
            #Creating a class to store the slopes of the first and second line fit, and theur difference
            class Lines:
                def __init__(self, slopeDifferences, line1, line2):
                    self.slopeDifferences = slopeDifferences
                    self.line1 = line1
                    self.line2 = line2
                       
            #Creating the list to store all these data
            lineData = []
        
            #Fitting the selected part of the curve
            for k in range(index1, index2):
                #Get data in left side
                x1 = xFit50[1:k]
                y1 = yFit50[1:k]
                #Fit a line through the left side
                coefficients1 = np.polyfit(x1, y1, 1)
                #Get data in right side
                x2 = xFit50[k+1:len(xFit50)]
                y2 = yFit50[k+1:len(yFit50)]
                #Fit a line through the right side
                coefficients2 = np.polyfit(x2, y2, 1)
                #Compute difference in slopes, and store in the list of lineData along with line equation coefficients
                slopeDiff = abs(coefficients1[1] - coefficients2[1])
                lineData.append ( Lines ( slopeDiff, coefficients1, coefficients2))
                #If we want to print them
                #print(k, lineData[k-index1-1].slopeDifferences, ', ', lineData[k-index1-1].line1 )
            
            #Find index for which the slope difference is the greatest 
            allslopeDiff = np.zeros([index2-index1-1])
            for k in range(0 , index2-index1-1):
                allslopeDiff[k] = lineData[k].slopeDifferences
        
            maxslopeDiff = max ( allslopeDiff)
            indexOfMaxSlopeDiff = np.argmax( allslopeDiff)
            st.write (indexOfMaxSlopeDiff, maxslopeDiff)
        
            #Plot the slopes and slope's differences.
            ax3.plot(xFit50, yFit50)     #Plot the fitted data.
            #We can add grids if we want
            #    grid on
            
            
            
            
            ax3.set_xlabel('Strain (%)',fontsize=font)
            ax3.set_ylabel('Stress (MPa)',fontsize=font)
            #Use the equation of line1 to get fitted/regressed y1 values.
            slope1 = lineData[indexOfMaxSlopeDiff].line1[0]
            intercept1 = lineData[indexOfMaxSlopeDiff].line1[1]
            y1Fitted = slope1 * xFit50 + intercept1
        
            #Plot line 1 over/through data.
            ax3.plot(xFit50, y1Fitted, linewidth = 2, color='red')
        
            #Use the equation of line2 to get fitted/regressed y2 values.
            slope2 = lineData[indexOfMaxSlopeDiff].line2[0]
            intercept2 = lineData[indexOfMaxSlopeDiff].line2[1]
            y2Fitted = slope2 * xFit50 + intercept2
        
            #Plot line 2 over/through data.
            ax3.plot(xFit50, y2Fitted, linewidth= 2, color='red')
            xc = (intercept2 - intercept1) / (slope1 - slope2)
            ax3.axvline(xc, color='magenta')
            ax3.set_title('Plot of linear regressions overlaid',fontsize=font)
            
            ## Fig 4
            
            #Young Modulus calculations as the slope of the first curve. Units: kPa
            YoungModulus = slope1 * 1000
            MaxStress = max(y1Fitted)
            xYM = np.linspace(0, (MaxStress/slope1) ,100)
            yYM = xYM*YoungModulus/1000

            ax3.plot(xYM, yYM, color='green')
            #Elastic deformation as the intercept of the two lines.
            ElasticDeformation = xc
            
            #Tensile strength and tear load
            #---------------------------------------------------------------------------------------------------------------------
            
            #Default:
            xMax = x[round(len(x)/tearSelectionRange):len(x)]
            yMax = y[round(len(y)/tearSelectionRange):len(x)]
        
            dyMax=np.gradient(yMax)/np.gradient(xMax)             #Derivative of the Max curve.
        
            #Get rid of infinite values. TEST
            #Changing dyMax from a list to an array
            dyMaxArr=np.array(dyMax)
            #Getting the indexes of very high values (considered infinite) -> method to change if there is 'inf'
            index=np.where(dyMaxArr > 10E6 )   #Change the value for higher ranges (initiial value = 10E6 )
            dyMaxArr= np.delete(dyMaxArr, index ,axis = 0)
        
            #Find the point where the change in gradient is max
            #Array to stock the difference in gradient
            dyMaxVar = np.zeros([len(dyMax)-1])
            for k in range(0, len(dyMax)-2):
                dyMaxVar[k]=dyMax[k+1]-dyMax[k]
    
            #Finding the max and its index
            max_dyMaxVar = max ( dyMaxVar )
            indexOfmax_dyMaxVar = np.argmax( dyMaxVar )
            st.write(indexOfmax_dyMaxVar, max_dyMaxVar)
            #Default method
            ax4.plot(x,y,'.')
                #If i have half tear selection range
            ax4.axvline(x[indexOfmax_dyMaxVar+round(len(x)/tearSelectionRange)], color='magenta')
                    #If i have many tear selection range
            #ax4.axvline(x[indexOfmax_dyMaxVar+round(6*len(x)/tearSelectionRange)], color='magenta')
            ax4.set_xlabel('Strain (%)',fontsize=font)
            ax4.set_ylabel('Stress (MPa)',fontsize=font)
            ax4.set_title('Tearing point',fontsize=font)
            
        
            #Breaking index on the tearSelectionRange, strain and stress
            BreakIndex = indexOfmax_dyMaxVar
            originalIndexBreak = round(len(x)/tearSelectionRange)+BreakIndex
            originalStrainBreak = x[originalIndexBreak];
            originalStressBreak = y[originalIndexBreak-correctionBeforeBreak];
        
            #Tensile strenght in N/mm2 (conversion: 1 MPa = 1 N/mm2).
            TensileStrength = originalStressBreak
        
            #Tear Load (limite de rupture) in N.
            
            #Default method
            mm2Value = float(SpecimenWidth) * float(SpecimenThickness)  #Area of the measure.
            TearLoad = originalStressBreak * mm2Value
            
            #Percentage elongation at break 
            PercentageElongBreak = x[originalIndexBreak-correctionBeforeBreak]
            
            #Plastic deformation of the material 
            PlasticDeformation = PercentageElongBreak - ElasticDeformation

        elif ELC:
            #Data processing (fixed for all specimens)
            
            polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
            cutcurvefite = 5               #Percent of left points on each side. Default 5%.
            tearSelectionRange= 2          #Sections of the curve for tensile strenght. Default: 2, second half.
            correctionBeforeBreak=1        #Correction of the value before tearing. Default 1. Method to be improved.

            #Extracting Data
            while (x[-1]==0) and (y[-1]==0) :
                x=np.delete (x, -1, 0)
                y=np.delete (y, -1, 0)

            # Selection Range
            
            col_f,col_f1 = st.columns(2)
            with col_f:
                f = int(st.slider(":orange[End of the selection range for Young's modulus and Elastic Limit]", 
                                  min_value=0,
                                  max_value=len(x)-1,
                                  value = round((len(x)-1)/2),step=1))
                with st.expander("Help?"):
                    st.write(v0_1_explainations['YMAndELSelectionRange'])
            ## Plot
            with col_f1:

                fig_selectionRange, ax_selectionRange = plt.subplots()
                ax_selectionRange.plot(x,y,'+')
                ax_selectionRange.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize= font)
                ax_selectionRange.set_xlabel('Strain (%)',fontsize= font)
                ax_selectionRange.set_ylabel('Stress (MPa)',fontsize= font)
                ax_selectionRange.axvspan(0,x[f], alpha = 0.5 , color='orange')
                               
                st.pyplot(fig_selectionRange,clear_figure=True)
                
            #Figures            
            fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))            

            #Shear Stress vs Strain plot
            ax1.plot(x,y)
            ax1.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize=font)
            ax1.set_xlabel('Strain (%)',fontsize=font)
            ax1.set_ylabel('Stress (MPa)',fontsize=font)
            # mplt.show()
        
        
            #Take only the half of the curve to avoid tear point
            x50 = x[1:f]#*round(len(x)/curveSelectionrange)]
            y50 = y[1:f]#*round(len(y)/curveSelectionrange)]
            ax2.plot(x50,y50,'x')
    
            #Polyfit of the selected values
            coeffs1_50 = np.polyfit(x50, y50, polynomialFit)    #Get the fit. First order poylnomial order x. TO BE IMPROVED.
    
            # Get the x values for the fit at higher resolution
            xFit50 = np.linspace(x50[1], x50[len(x50)-1])
            # Get the estimated y values
            yFit50 = np.polyval(coeffs1_50, xFit50)
    
            #Plot them as a line
            ax2.plot(xFit50, yFit50, linewidth=3)
            ax2.set_title('Fitted data: Stress vs Strain',fontsize=font)
            ax2.set_xlabel('Strain (%)',fontsize=font)
            ax2.set_ylabel('Stress (MPa)',fontsize=font)
            # mplt.show()
    
    
            #Finding the elastic limit and calculating the Young modulus
    
            #Selection of the values to be kept for the curves fitting
            index1 = round((cutcurvefite/100) * len(xFit50))        #Percent of the curve to be discarded on each side for the fit
            index2 = round(((100-cutcurvefite)/100) * len(xFit50))  #Percent of the curve to be kept for the fit
            #index1=0
            #index2=len(xFit50)
        
            #Creating a class to store the slopes of the first and second line fit, and theur difference
            class Lines:
                def __init__(self, slopeDifferences, line1, line2):
                    self.slopeDifferences = slopeDifferences
                    self.line1 = line1
                    self.line2 = line2
            
            #Creating the list to store all these data
            lineData = []
    
            #Fitting the selected part of the curve
            for k in range(index1, index2):
                #Get data in left side
                x1 = xFit50[1:k]
                y1 = yFit50[1:k]
                #Fit a line through the left side
                coefficients1 = np.polyfit(x1, y1, 1)
                #Get data in right side
                x2 = xFit50[k+1:len(xFit50)]
                y2 = yFit50[k+1:len(yFit50)]
                #Fit a line through the right side
                coefficients2 = np.polyfit(x2, y2, 1)
                #Compute difference in slopes, and store in the list of lineData along with line equation coefficients
                slopeDiff = abs(coefficients1[1] - coefficients2[1])
                lineData.append ( Lines ( slopeDiff, coefficients1, coefficients2))
                #If we want to print them
                #print(k, lineData[k-index1-1].slopeDifferences, ', ', lineData[k-index1-1].line1 )
        
            #Find index for which the slope difference is the greatest 
            allslopeDiff = np.zeros([index2-index1-1])
            for k in range(0 , index2-index1-1):
                allslopeDiff[k] = lineData[k].slopeDifferences
    
            maxslopeDiff = max ( allslopeDiff)
            indexOfMaxSlopeDiff = np.argmax( allslopeDiff)
            st.write (indexOfMaxSlopeDiff, maxslopeDiff)
    
            #Plot the slopes and slope's differences.
            ax3.plot(xFit50, yFit50)     #Plot the fitted data.
            #We can add grids if we want
            #    grid on
            ax3.set_xlabel('Strain (%)',fontsize=font)
            ax3.set_ylabel('Stress (MPa)',fontsize=font)
            #Use the equation of line1 to get fitted/regressed y1 values.
            slope1 = lineData[indexOfMaxSlopeDiff].line1[0]
            intercept1 = lineData[indexOfMaxSlopeDiff].line1[1]
            y1Fitted = slope1 * xFit50 + intercept1
    
            #Plot line 1 over/through data.
            ax3.plot(xFit50, y1Fitted, linewidth = 2, color='red')
    
            #Use the equation of line2 to get fitted/regressed y2 values.
            slope2 = lineData[indexOfMaxSlopeDiff].line2[0]
            intercept2 = lineData[indexOfMaxSlopeDiff].line2[1]
            y2Fitted = slope2 * xFit50 + intercept2
    
            #Plot line 2 over/through data.
            ax3.plot(xFit50, y2Fitted, linewidth= 2, color='orange')
            xc = (intercept2 - intercept1) / (slope1 - slope2)
            ax3.axvline(xc, color='magenta')
            ax3.set_title('Plot of linear regressions overlaid',fontsize=font)
            # mplt.show()
        
            #Young Modulus calculations as the slope of the first curve. Units: kPa
            YoungModulus = slope1 * 1000
            MaxStress = max(y1Fitted)
            xYM = np.linspace(0, (MaxStress/slope1) ,100)
            yYM = xYM*YoungModulus/1000

            ax3.plot(xYM, yYM, color='green')
            #Elastic deformation as the intercept of the two lines.
            ElasticDeformation = xc
    
            #Tensile strength and tear load
            #---------------------------------------------------------------------------------------------------------------------
            #Default for half tear selection range
            xMax = x[round(len(x)/tearSelectionRange):len(x)]
            yMax = y[round(len(y)/tearSelectionRange):len(x)]
        
            dyMax=np.gradient(yMax)/np.gradient(xMax)             #Derivative of the Max curve.
    
            #Get rid of infinite values. TEST
            #Changing dyMax from a list to an array
            dyMaxArr=np.array(dyMax)
            #Getting the indexes of very high values (considered infinite) -> method to change if there is 'inf'
            index=np.where(dyMaxArr > 10E6 )   #Change the value for higher ranges (initiial value = 10E6 )
            dyMaxArr= np.delete(dyMaxArr, index ,axis = 0)
    
            #Find the point where the change in gradient is max
            #Array to stock the difference in gradient
            dyMaxVar = np.zeros([len(dyMax)-1])
            for k in range(0, len(dyMax)-2):
                dyMaxVar[k]=dyMax[k+1]-dyMax[k]
    
            #Finding the max and its index
            max_dyMaxVar = max ( dyMaxVar )
            indexOfmax_dyMaxVar = np.argmax( dyMaxVar )
            st.write(indexOfmax_dyMaxVar, max_dyMaxVar)
        
            #Default method
            ax4.plot(x,y)
                #If i have half tear selection range
            ax4.axvline(x[indexOfmax_dyMaxVar+round(len(x)/tearSelectionRange)], color='magenta')
                #If i have many tear selection range
            #mplt.axvline(x[indexOfmax_dyMaxVar+round(6*len(x)/tearSelectionRange)], color='magenta')
            ax4.set_xlabel('Strain (%)')
            ax4.set_ylabel('Stress (MPa)')
            ax4.set_title('Tearing point') 
            # mplt.show()
    
            #Breaking index on the tearSelectionRange, strain and stress
            BreakIndex = indexOfmax_dyMaxVar
            originalIndexBreak = round(len(x)/tearSelectionRange)+BreakIndex
            originalStrainBreak = x[originalIndexBreak];
            originalStressBreak = y[originalIndexBreak-correctionBeforeBreak];
            
            #Tensile strength in N/mm2 (conversion: 1 MPa = 1 N/mm2).
            TensileStrength = originalStressBreak
        
            #Tear Load (limite de rupture) in N.
    
            #Default method
            mm2Value = float(SpecimenWidth) * float(SpecimenThickness)  #Area of the measure.
            TearLoad = originalStressBreak * mm2Value
        
            #Percentage elongation at break 
            PercentageElongBreak = x[originalIndexBreak-correctionBeforeBreak]
        
            #Plastic deformation of the material 
            PlasticDeformation = PercentageElongBreak - ElasticDeformation
            
            # fig.tight_layout()
            # st.header("Graphs")
            # st.pyplot(fig, clear_figure=True)
    elif DTL_MSR_MTL == TearLoadModes[1]:
        if not ELC:
   
            #Data processing (fixed for all specimens)
            curveSelectionrange=2          #Sections of the curve for Young Modulus and Elastic Limit. Default: 2.
            polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
            cutcurvefite = 5               #Percent of left points on each side. Default 5%.
            correctionBeforeBreak=1        #Correction of the value before tearing. Default 1. Method to be improved.
            
            #Extracting Data
            while (x[-1]==0) and (y[-1]==0) :
                x=np.delete (x, -1, 0)
                y=np.delete (y, -1, 0)    
                        
            # Ask Selection Ranges
            col_d,col_d1 = st.columns(2)
            with col_d:
                d = st.slider(":red[Tear Load Selection Range]",
                              min_value=0,
                              max_value=len(x)-1,
                              step = 1,
                              value=(round((len(x)-1)/2),len(x)-1),
                              )
                with st.expander("Help?"):
                    st.write(v0_1_explainations['MultipleTearLoadSelectionRange'])

            ## Plot
            with col_d1:             
                fig_selectionRange, ax_selectionRange = plt.subplots()
                ax_selectionRange.plot(x,y)
                ax_selectionRange.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize= font)
                ax_selectionRange.set_xlabel('Strain (%)',fontsize= font)
                ax_selectionRange.set_ylabel('Stress (MPa)',fontsize= font)        
                ax_selectionRange.axvline(x[d[0]],color='r', linewidth = 1)
                ax_selectionRange.axvline(x[d[1]],color='r', linewidth = 3)
                ax_selectionRange.axvspan(x[d[0]],x[d[1]], alpha = 0.5 , color='r')
                
                st.pyplot(fig_selectionRange,clear_figure=True)
            
            #Sections of the curve for tensile strenght. Default: 2, second half.
            d_s=d[0]
            d_e=d[1]
            
            #Figures
            fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(10,10)) 
            
            #Shear Stress vs Strain plot
            ax1.plot(x,y)
            ax1.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize= font)
            ax1.set_xlabel('Strain (%)',fontsize= font)
            ax1.set_ylabel('Stress (MPa)',fontsize= font)
            # mplt.show()
        
        
            #Take only the half of the curve to avoid tear point
            x50 = x[1:round(len(x)/curveSelectionrange)]
            y50 = y[1:round(len(y)/curveSelectionrange)]
            ax2.plot(x50,y50,'x')
        
            #Polyfit of the selected values
            coeffs1_50 = np.polyfit(x50, y50, polynomialFit)    #Get the fit. First order poylnomial order x. TO BE IMPROVED.
        
            # Get the x values for the fit at higher resolution
            xFit50 = np.linspace(x50[1], x50[len(x50)-1])
            # Get the estimated y values
            yFit50 = np.polyval(coeffs1_50, xFit50)
        
            #Plot them as a line
            ax2.plot(xFit50, yFit50, linewidth=3)
            ax2.set_title('Fitted data: Stress vs Strain',fontsize= font)
            ax2.set_xlabel('Strain (%)',fontsize= font)
            ax2.set_ylabel('Stress (MPa)',fontsize= font)
            # mplt.show()
        
        
            #Finding the elastic limit and calculating the Young modulus
        
            #Selection of the values to be kept for the curves fitting
            index1 = round((cutcurvefite/100) * len(xFit50))        #Percent of the curve to be discarded on each side for the fit
            index2 = round(((100-cutcurvefite)/100) * len(xFit50))  #Percent of the curve to be kept for the fit
            #index1=0
            #index2=len(xFit50)
        
            #Creating a class to store the slopes of the first and second line fit, and theur difference
            class Lines:
                def __init__(self, slopeDifferences, line1, line2):
                    self.slopeDifferences = slopeDifferences
                    self.line1 = line1
                    self.line2 = line2
            
            #Creating the list to store all these data
            lineData = []
        
            #Fitting the selected part of the curve
            for k in range(index1, index2):
                #Get data in left side
                x1 = xFit50[1:k]
                y1 = yFit50[1:k]
                #Fit a line through the left side
                coefficients1 = np.polyfit(x1, y1, 1)
                #Get data in right side
                x2 = xFit50[k+1:len(xFit50)]
                y2 = yFit50[k+1:len(yFit50)]
                #Fit a line through the right side
                coefficients2 = np.polyfit(x2, y2, 1)
                #Compute difference in slopes, and store in the list of lineData along with line equation coefficients
                slopeDiff = abs(coefficients1[1] - coefficients2[1])
                lineData.append ( Lines ( slopeDiff, coefficients1, coefficients2))
                #If we want to print them
                #print(k, lineData[k-index1-1].slopeDifferences, ', ', lineData[k-index1-1].line1 )
        
            #Find index for which the slope difference is the greatest 
            allslopeDiff = np.zeros([index2-index1-1])
            for k in range(0 , index2-index1-1):
                allslopeDiff[k] = lineData[k].slopeDifferences
        
            maxslopeDiff = max ( allslopeDiff)
            indexOfMaxSlopeDiff = np.argmax( allslopeDiff)
            print ('indexOfMaxSlopeDiff: ',indexOfMaxSlopeDiff, 'maxslopeDiff: ', maxslopeDiff)
        
            #Plot the slopes and slope's differences.
            ax3.plot(xFit50, yFit50)     #Plot the fitted data.
            #We can add grids if we want
            #    grid on
            ax3.set_xlabel('Strain (%)',fontsize= font)
            ax3.set_ylabel('Stress (MPa)',fontsize= font)
            #Use the equation of line1 to get fitted/regressed y1 values.
            slope1 = lineData[indexOfMaxSlopeDiff].line1[0]
            intercept1 = lineData[indexOfMaxSlopeDiff].line1[1]
            y1Fitted = slope1 * xFit50 + intercept1
        
            #Plot line 1 over/through data.
            ax3.plot(xFit50, y1Fitted, linewidth = 2, color='red')
        
            #Use the equation of line2 to get fitted/regressed y2 values.
            slope2 = lineData[indexOfMaxSlopeDiff].line2[0]
            intercept2 = lineData[indexOfMaxSlopeDiff].line2[1]
            y2Fitted = slope2 * xFit50 + intercept2
        
            #Plot line 2 over/through data.
            ax3.plot(xFit50, y2Fitted, linewidth= 2, color='red')
            xc = (intercept2 - intercept1) / (slope1 - slope2)
            ax3.axvline(xc, color='magenta')
            ax3.set_title('Plot of linear regressions overlaid',fontsize= font)
            # mplt.show()
        
            #Young Modulus calculations as the slope of the first curve. Units: kPa
            YoungModulus = slope1 * 1000
            MaxStress = max(y1Fitted)
            xYM = np.linspace(0, (MaxStress/slope1) ,100)
            yYM = xYM*YoungModulus/1000

            ax3.plot(xYM, yYM, color='green')
            #Elastic deformation as the intercept of the two lines.
            ElasticDeformation = xc
        
            #Tensile strength and tear load
            #---------------------------------------------------------------------------------------------------------------------
            #If there is many tear points and i want to change the selection range
            xMax = x[d_s:d_e]#s*round(len(x)/tearSelectionRange):round(e*len(x)/tearSelectionRange)]
            yMax = y[d_s:d_e]#s*round(len(y)/tearSelectionRange):round(e*len(x)/tearSelectionRange)]
            ax4.plot(xMax,yMax,'.')
            ax4.set_xlabel('Strain (%)',fontsize= font)
            ax4.set_ylabel('Stress (MPa)',fontsize= font)
            ax4.set_title('Part of the curve for breaking point',fontsize= font)
            # mplt.show()
        
            dyMax=np.gradient(yMax)/np.gradient(xMax)             #Derivative of the Max curve.
        
            #Get rid of infinite values. TEST
            #Changing dyMax from a list to an array
            dyMaxArr=np.array(dyMax)
            #Getting the indexes of very high values (considered infinite) -> method to change if there is 'inf'
            index=np.where(dyMaxArr > 10E6 )   #Change the value for higher ranges (initiial value = 10E6 )
            dyMaxArr= np.delete(dyMaxArr, index ,axis = 0)
        
            #Find the point where the change in gradient is max
            #Array to stock the difference in gradient
            dyMaxVar = np.zeros([len(dyMax)-1])
            for k in range(0, len(dyMax)-2):
                dyMaxVar[k]=dyMax[k+1]-dyMax[k]
        
            #Finding the max and its index
            max_dyMaxVar = max ( dyMaxVar )
            indexOfmax_dyMaxVar = np.argmax( dyMaxVar )
            print (indexOfmax_dyMaxVar, max_dyMaxVar)
        
            ax5.plot(x,y)
            ax5.axvline(x[indexOfmax_dyMaxVar+round(d_s)], color='magenta')
            ax5.set_xlabel('Strain (%)',fontsize= font)
            ax5.set_ylabel('Stress (MPa)',fontsize= font)
            ax5.set_title('Tearing point',fontsize= font)
            # mplt.show()
        
            #Breaking index on the tearSelectionRange, strain and stress
            BreakIndex = indexOfmax_dyMaxVar
            originalIndexBreak = round(d_s)+BreakIndex
            originalStrainBreak = x[originalIndexBreak];
            originalStressBreak = y[originalIndexBreak-correctionBeforeBreak];
        
            #Tensile strenght in N/mm2 (conversion: 1 MPa = 1 N/mm2).
            TensileStrength = max (yMax)
        
            #Tear Load (limite de rupture) in N.
            mm2Value = float(SpecimenWidth) * float(SpecimenThickness)  #Area of the measure.
            TearLoad = originalStressBreak * mm2Value
        
            #Percentage elongation at break 
            PercentageElongBreak = x[originalIndexBreak-correctionBeforeBreak]
            
            #Plastic deformation of the material 
            PlasticDeformation = PercentageElongBreak - ElasticDeformation
        if ELC:
      
            #Data processing (fixed for all specimens)
            curveSelectionrange=2          #Sections of the curve for Young Modulus and Elastic Limit. Default: 2.
            polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
            cutcurvefite = 5               #Percent of left points on each side. Default 5%.
            correctionBeforeBreak=1        #Correction of the value before tearing. Default 1. Method to be improved.
            
            #Extracting Data
            while (x[-1]==0) and (y[-1]==0) :
                x=np.delete (x, -1, 0)
                y=np.delete (y, -1, 0)    
                        
            # Ask Selection Ranges
            col_d,col_d1 = st.columns(2)
            with col_d:
                f = int(st.slider(":orange[End of the selection range for Young's modulus and Elastic Limit]", 
                                  min_value=0,
                                  max_value=len(x)-1,
                                  value = round((len(x)-1)/2),
                                  step=1,)
                        )
                d = st.slider(
                    ":red[Tear Load Selection Range]",
                    min_value=0,
                    max_value=len(x)-1,
                    step = 1,
                    value=(round((len(x)-1)/2),len(x)-1),
                    )

                with st.expander("Help?"):
                    st.write(v0_1_explainations['YMAndELSelectionRange'])
                    st.write(v0_1_explainations['MultipleTearLoadSelectionRange'])
                    
            ## Plot
            with col_d1:             
                fig_selectionRange, ax_selectionRange = plt.subplots()
                ax_selectionRange.plot(x,y)
                ax_selectionRange.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize= font)
                ax_selectionRange.set_xlabel('Strain (%)',fontsize= font)
                ax_selectionRange.set_ylabel('Stress (MPa)',fontsize= font)        
                ax_selectionRange.axvline(x[d[0]],color='r', linewidth = 1)
                ax_selectionRange.axvline(x[d[1]],color='r', linewidth = 3)
                ax_selectionRange.axvspan(x[d[0]],x[d[1]], alpha = 0.5 , color='r')
                
                ax_selectionRange.axvline(0,color='orange', linewidth = 1)
                ax_selectionRange.axvline(x[f],color='orange', linewidth = 3)
                ax_selectionRange.axvspan(0,x[f], alpha = 0.5 , color='orange')

                st.pyplot(fig_selectionRange,clear_figure=True)
            
            #Sections of the curve for tensile strenght. Default: 2, second half.
            d_s=d[0]
            d_e=d[1]
            
            #Figures
            fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(10,10)) 
            
            #Shear Stress vs Strain plot
            ax1.plot(x,y)
            ax1.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize= font)
            ax1.set_xlabel('Strain (%)',fontsize= font)
            ax1.set_ylabel('Stress (MPa)',fontsize= font)
            # mplt.show()
        
        
            #Take only the half of the curve to avoid tear point
            x50 = x[1:f]
            y50 = y[1:f]
            ax2.plot(x50,y50,'x')
        
            #Polyfit of the selected values
            coeffs1_50 = np.polyfit(x50, y50, polynomialFit)    #Get the fit. First order poylnomial order x. TO BE IMPROVED.
        
            # Get the x values for the fit at higher resolution
            xFit50 = np.linspace(x50[1], x50[len(x50)-1])
            # Get the estimated y values
            yFit50 = np.polyval(coeffs1_50, xFit50)
        
            #Plot them as a line
            ax2.plot(xFit50, yFit50, linewidth=3)
            ax2.set_title('Fitted data: Stress vs Strain',fontsize= font)
            ax2.set_xlabel('Strain (%)',fontsize= font)
            ax2.set_ylabel('Stress (MPa)',fontsize= font)
            # mplt.show()
        
            #Finding the elastic limit and calculating the Young modulus
        
            #Selection of the values to be kept for the curves fitting
            index1 = round((cutcurvefite/100) * len(xFit50))        #Percent of the curve to be discarded on each side for the fit
            index2 = round(((100-cutcurvefite)/100) * len(xFit50))  #Percent of the curve to be kept for the fit
            #index1=0
            #index2=len(xFit50)
        
            #Creating a class to store the slopes of the first and second line fit, and theur difference
            class Lines:
                def __init__(self, slopeDifferences, line1, line2):
                    self.slopeDifferences = slopeDifferences
                    self.line1 = line1
                    self.line2 = line2
            
            #Creating the list to store all these data
            lineData = []
        
            #Fitting the selected part of the curve
            for k in range(index1, index2):
                #Get data in left side
                x1 = xFit50[1:k]
                y1 = yFit50[1:k]
                #Fit a line through the left side
                coefficients1 = np.polyfit(x1, y1, 1)
                #Get data in right side
                x2 = xFit50[k+1:len(xFit50)]
                y2 = yFit50[k+1:len(yFit50)]
                #Fit a line through the right side
                coefficients2 = np.polyfit(x2, y2, 1)
                #Compute difference in slopes, and store in the list of lineData along with line equation coefficients
                slopeDiff = abs(coefficients1[1] - coefficients2[1])
                lineData.append ( Lines ( slopeDiff, coefficients1, coefficients2))
                #If we want to print them
                #print(k, lineData[k-index1-1].slopeDifferences, ', ', lineData[k-index1-1].line1 )
        
            #Find index for which the slope difference is the greatest 
            allslopeDiff = np.zeros([index2-index1-1])
            for k in range(0 , index2-index1-1):
                allslopeDiff[k] = lineData[k].slopeDifferences
        
            maxslopeDiff = max ( allslopeDiff)
            indexOfMaxSlopeDiff = np.argmax( allslopeDiff)
            st.write('indexOfMaxSlopeDiff:',indexOfMaxSlopeDiff, 'maxslopeDiff: ',maxslopeDiff)
        
            #Plot the slopes and slope's differences.
            ax3.plot(xFit50, yFit50)     #Plot the fitted data.
            #We can add grids if we want
            #    grid on
            ax3.set_xlabel('Strain (%)',fontsize= font)
            ax3.set_ylabel('Stress (MPa)',fontsize= font)
            #Use the equation of line1 to get fitted/regressed y1 values.
            slope1 = lineData[indexOfMaxSlopeDiff].line1[0]
            intercept1 = lineData[indexOfMaxSlopeDiff].line1[1]
            y1Fitted = slope1 * xFit50 + intercept1
        
            #Plot line 1 over/through data.
            ax3.plot(xFit50, y1Fitted, linewidth = 2, color='red')
        
            #Use the equation of line2 to get fitted/regressed y2 values.
            slope2 = lineData[indexOfMaxSlopeDiff].line2[0]
            intercept2 = lineData[indexOfMaxSlopeDiff].line2[1]
            y2Fitted = slope2 * xFit50 + intercept2
        
            #Plot line 2 over/through data.
            ax3.plot(xFit50, y2Fitted, linewidth= 2, color='red')
            xc = (intercept2 - intercept1) / (slope1 - slope2)
            ax3.axvline(xc, color='magenta')
            ax3.set_title('Plot of linear regressions overlaid',fontsize= font)
            # mplt.show()
        
            #Young Modulus calculations as the slope of the first curve. Units: kPa
            YoungModulus = slope1 * 1000
            MaxStress = max(y1Fitted)
            xYM = np.linspace(0, (MaxStress/slope1) ,100)
            yYM = xYM*YoungModulus/1000

            ax3.plot(xYM, yYM, color='green')
            #Elastic deformation as the intercept of the two lines.
            ElasticDeformation = xc
        
            #Tensile strength and tear load
            #---------------------------------------------------------------------------------------------------------------------
            #If there is many tear points and i want to change the selection range
            xMax = x[d_s:d_e]#s*round(len(x)/tearSelectionRange):round(e*len(x)/tearSelectionRange)]
            yMax = y[d_s:d_e]#s*round(len(y)/tearSelectionRange):round(e*len(x)/tearSelectionRange)]
            ax4.plot(xMax,yMax,'.')
            ax4.set_xlabel('Strain (%)',fontsize= font)
            ax4.set_ylabel('Stress (MPa)',fontsize= font)
            ax4.set_title('Part of the curve for breaking point',fontsize= font)
            # mplt.show()
        
            dyMax=np.gradient(yMax)/np.gradient(xMax)             #Derivative of the Max curve.
        
            #Get rid of infinite values. TEST
            #Changing dyMax from a list to an array
            dyMaxArr=np.array(dyMax)
            #Getting the indexes of very high values (considered infinite) -> method to change if there is 'inf'
            index=np.where(dyMaxArr > 10E6 )   #Change the value for higher ranges (initiial value = 10E6 )
            dyMaxArr= np.delete(dyMaxArr, index ,axis = 0)
        
            #Find the point where the change in gradient is max
            #Array to stock the difference in gradient
            dyMaxVar = np.zeros([len(dyMax)-1])
            for k in range(0, len(dyMax)-2):
                dyMaxVar[k]=dyMax[k+1]-dyMax[k]
        
            #Finding the max and its index
            max_dyMaxVar = max ( dyMaxVar )
            indexOfmax_dyMaxVar = np.argmax( dyMaxVar )
            st.write("indexOfmax_dyMaxVar: ",indexOfmax_dyMaxVar, "max_dyMaxVar: " , max_dyMaxVar)
        
            ax5.plot(x,y)
            ax5.axvline(x[indexOfmax_dyMaxVar+round(d_s)], color='magenta')
            ax5.set_xlabel('Strain (%)',fontsize= font)
            ax5.set_ylabel('Stress (MPa)',fontsize= font)
            ax5.set_title('Tearing point',fontsize= font)
            # mplt.show()
        
            #Breaking index on the tearSelectionRange, strain and stress
            BreakIndex = indexOfmax_dyMaxVar
            originalIndexBreak = round(d_s)+BreakIndex
            originalStrainBreak = x[originalIndexBreak];
            originalStressBreak = y[originalIndexBreak-correctionBeforeBreak];
        
            #Tensile strenght in N/mm2 (conversion: 1 MPa = 1 N/mm2).
            TensileStrength = max (yMax)
        
            #Tear Load (limite de rupture) in N.
            mm2Value = float(SpecimenWidth) * float(SpecimenThickness)  #Area of the measure.
            TearLoad = originalStressBreak * mm2Value
        
            #Percentage elongation at break 
            PercentageElongBreak = x[originalIndexBreak-correctionBeforeBreak]
            
            #Plastic deformation of the material 
            PlasticDeformation = PercentageElongBreak - ElasticDeformation
    
    elif DTL_MSR_MTL == TearLoadModes[2]:
        if not ELC:
            #Data processing (fixed for all specimens)
            curveSelectionrange=2          #Sections of the curve for Young Modulus and Elastic Limit. Default: 2.
            polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
            cutcurvefite = 5               #Percent of left points on each side. Default 5%.
            tearSelectionRange=2           #Sections of the curve for tensile strenght. Default: 2, second half.
            correctionBeforeBreak=1        #Correction of the value before tearing. Default 1. Method to be improved.
           
            #Extracting Data
            while (x[-1]==0) and (y[-1]==0) :
                x=np.delete (x, -1, 0)
                y=np.delete (y, -1, 0)
            
            #Shear Stress vs Strain plot
         
            fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10)) 
            ax1.plot(x,y)
            ax1.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize= font)
            ax1.set_xlabel('Strain (%)',fontsize= font)
            ax1.set_ylabel('Stress (MPa)',fontsize= font)
        
        
            #Get the first elastic limit and young's modulus
            x50 = x[1:round(len(x)/curveSelectionrange)]
            y50 = y[1:round(len(y)/curveSelectionrange)]
            ax2.plot(x50,y50,'x')
            
            #Polyfit of the selected values
            coeffs1_50 = np.polyfit(x50, y50, polynomialFit)    #Get the fit. First order poylnomial order x. TO BE IMPROVED.
        
            # Get the x values for the fit at higher resolution
            xFit50 = np.linspace(x50[1], x50[len(x50)-1])
            # Get the estimated y values
            yFit50 = np.polyval(coeffs1_50, xFit50)
        
            #Plot them as a line
            ax2.plot(xFit50, yFit50, linewidth=3)
            ax2.set_title('Fitted data: Stress vs Strain', fontsize=font)
            ax2.set_xlabel('Strain (%)', fontsize=font)
            ax2.set_ylabel('Stress (MPa)',fontsize=font)
                    
            #Finding the first elastic limit and calculating the first Young's modulus
        
            #Selection of the values to be kept for the curves fitting
            #---------------------------------------------------------------------------------------------------------------------
            index1 = round((cutcurvefite/100) * len(xFit50))        #Percent of the curve to be discarded on each side for the fit
            index2 = round(((100-cutcurvefite)/100) * len(xFit50))  #Percent of the curve to be kept for the fit
            #index1=0
            #index2=len(xFit50)
            
            #Creating a class to store the slopes of the first and second line fit, and theur difference
            class Lines:
                def __init__(self, slopeDifferences, line1, line2):
                    self.slopeDifferences = slopeDifferences
                    self.line1 = line1
                    self.line2 = line2
                       
            #Creating the list to store all these data
            lineData = []
        
            #Fitting the selected part of the curve
            for k in range(index1, index2):
                #Get data in left side
                x1 = xFit50[1:k]
                y1 = yFit50[1:k]
                #Fit a line through the left side
                coefficients1 = np.polyfit(x1, y1, 1)
                #Get data in right side
                x2 = xFit50[k+1:len(xFit50)]
                y2 = yFit50[k+1:len(yFit50)]
                #Fit a line through the right side
                coefficients2 = np.polyfit(x2, y2, 1)
                #Compute difference in slopes, and store in the list of lineData along with line equation coefficients
                slopeDiff = abs(coefficients1[1] - coefficients2[1])
                lineData.append ( Lines ( slopeDiff, coefficients1, coefficients2))
                #If we want to print them
                #print(k, lineData[k-index1-1].slopeDifferences, ', ', lineData[k-index1-1].line1 )
            
            #Find index for which the slope difference is the greatest 
            allslopeDiff = np.zeros([index2-index1-1])
            for k in range(0 , index2-index1-1):
                allslopeDiff[k] = lineData[k].slopeDifferences
        
            maxslopeDiff = max ( allslopeDiff)
            indexOfMaxSlopeDiff = np.argmax( allslopeDiff)
            st.write (indexOfMaxSlopeDiff, maxslopeDiff)
        
            #Plot the slopes and slope's differences.
            ax3.plot(xFit50, yFit50)     #Plot the fitted data.
            #We can add grids if we want
            #    grid on
            ax3.set_xlabel('Strain (%)',fontsize=font)
            ax3.set_ylabel('Stress (MPa)',fontsize=font)
            #Use the equation of line1 to get fitted/regressed y1 values.
            slope1 = lineData[indexOfMaxSlopeDiff].line1[0]
            intercept1 = lineData[indexOfMaxSlopeDiff].line1[1]
            y1Fitted = slope1 * xFit50 + intercept1
        
            #Plot line 1 over/through data.
            ax3.plot(xFit50, y1Fitted, linewidth = 2, color='red')
        
            #Use the equation of line2 to get fitted/regressed y2 values.
            slope2 = lineData[indexOfMaxSlopeDiff].line2[0]
            intercept2 = lineData[indexOfMaxSlopeDiff].line2[1]
            y2Fitted = slope2 * xFit50 + intercept2
        
            #Plot line 2 over/through data.
            ax3.plot(xFit50, y2Fitted, linewidth= 2, color='red')
            xc = (intercept2 - intercept1) / (slope1 - slope2)
            ax3.axvline(xc, color='magenta')
            ax3.set_title('Plot of linear regressions overlaid',fontsize=font)
            
            ## Fig 4
            
            #Young Modulus calculations as the slope of the first curve. Units: kPa
            YoungModulus = slope1 * 1000
            MaxStress = max(y1Fitted)
            xYM = np.linspace(0, (MaxStress/slope1) ,100)
            yYM = xYM*YoungModulus/1000

            ax3.plot(xYM, yYM, color='green')
            #Elastic deformation as the intercept of the two lines.
            ElasticDeformation = xc
            
            #Tensile strength and tear load
            #---------------------------------------------------------------------------------------------------------------------
            
            #Default:
            xMax = x[round(len(x)/tearSelectionRange):len(x)]
            yMax = y[round(len(y)/tearSelectionRange):len(x)]
        
            dyMax=np.gradient(yMax)/np.gradient(xMax)             #Derivative of the Max curve.
        
            #Get rid of infinite values. TEST
            #Changing dyMax from a list to an array
            dyMaxArr=np.array(dyMax)
            #Getting the indexes of very high values (considered infinite) -> method to change if there is 'inf'
            index=np.where(dyMaxArr > 10E6 )   #Change the value for higher ranges (initiial value = 10E6 )
            dyMaxArr= np.delete(dyMaxArr, index ,axis = 0)
        
            #Find the point where the change in gradient is max
            #Array to stock the difference in gradient
            dyMaxVar = np.zeros([len(dyMax)-1])
            for k in range(0, len(dyMax)-2):
                dyMaxVar[k]=dyMax[k+1]-dyMax[k]
    
            #Finding the max and its index
            max_dyMaxVar = max ( dyMaxVar )
            indexOfmax_dyMaxVar = np.argmax( dyMaxVar )
            st.write('indexOfmax_dyMaxVar: ',indexOfmax_dyMaxVar, 'max_dyMaxVar: ', max_dyMaxVar)
              
            #Index of the value where the tear load is

            spi = st.slider("Choose the index of the value where you want the tear load to be",min_value =1, max_value= len(x))
            
            fig_tearLoad,ax_tearLoad = plt.subplots()
            ax_tearLoad.plot(x,y)    
            if len(x)>10:
                ax_tearLoad.plot(x[-10], y[-10], 'r+') #1
                ax_tearLoad.annotate('10', (x[-10], y[-10]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>30:           
                ax_tearLoad.plot(x[-30], y[-30], 'r+') #2
                ax_tearLoad.annotate('30', (x[-30], y[-30]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>50:
                ax_tearLoad.plot(x[-50], y[-50], 'r+') #3
                ax_tearLoad.annotate('50', (x[-50], y[-50]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>100:
                ax_tearLoad.plot(x[-100], y[-100], 'r+') #4
                ax_tearLoad.annotate('100', (x[-100], y[-100]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>200:
                ax_tearLoad.plot(x[-200], y[-200], 'r+') #5
                ax_tearLoad.annotate('200', (x[-200], y[-200]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>500:
                ax_tearLoad.plot(x[-500], y[-500], 'r+') #6
                ax_tearLoad.annotate('500', (x[-500], y[-500]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            ax_tearLoad.axvline(x[-spi], color='magenta')
            ax_tearLoad.plot(x[-spi], y[-spi], '*', color='yellow')
            ax_tearLoad.set_xlabel('Strain (%)',fontsize=font)
            ax_tearLoad.set_ylabel('Stress (MPa)',fontsize=font)
            ax_tearLoad.set_title('Tearing point',fontsize=font)  
            st.pyplot(fig_tearLoad)
            
            ax4.plot(x,y)    
            if len(x)>10:
                ax4.plot(x[-10], y[-10], 'r+') #1
                ax4.annotate('10', (x[-10], y[-10]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>30:           
                ax4.plot(x[-30], y[-30], 'r+') #2
                ax4.annotate('30', (x[-30], y[-30]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>50:
                ax4.plot(x[-50], y[-50], 'r+') #3
                ax4.annotate('50', (x[-50], y[-50]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>100:
                ax4.plot(x[-100], y[-100], 'r+') #4
                ax4.annotate('100', (x[-100], y[-100]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>200:
                ax4.plot(x[-200], y[-200], 'r+') #5
                ax4.annotate('200', (x[-200], y[-200]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>500:
                ax4.plot(x[-500], y[-500], 'r+') #6
                ax4.annotate('500', (x[-500], y[-500]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            ax4.axvline(x[-spi], color='magenta')
            ax4.plot(x[-spi], y[-spi], '*', color='yellow')
            ax4.set_xlabel('Strain (%)',fontsize=font)
            ax4.set_ylabel('Stress (MPa)',fontsize=font)
            ax4.set_title('Tearing point',fontsize=font)  
            
            
                    #Breaking index on the tearSelectionRange, strain and stress
            BreakIndex = indexOfmax_dyMaxVar
            originalIndexBreak = round(len(x)/tearSelectionRange)+BreakIndex
            originalStrainBreak = x[originalIndexBreak];
            originalStressBreak = y[originalIndexBreak-correctionBeforeBreak];
            #Tensile strenght in N/mm2 (conversion: 1 MPa = 1 N/mm2).
            TensileStrength = max (yMax)
        
            #Tear Load (limite de rupture) in N
            Stressatbreak = y[-spi]
            mm2Value = float(SpecimenWidth) * float(SpecimenThickness)  #Area of the measure.
            TearLoad = y[-spi] * mm2Value 
        
            #Percentage elongation at break 
            PercentageElongBreak = x[originalIndexBreak-correctionBeforeBreak]
            
            #Plastic deformation of the material 
            PlasticDeformation = PercentageElongBreak - ElasticDeformation
        elif ELC:
            #Data processing (fixed for all specimens)
            
            polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
            cutcurvefite = 5               #Percent of left points on each side. Default 5%.
            tearSelectionRange= 2          #Sections of the curve for tensile strenght. Default: 2, second half.
            correctionBeforeBreak=1        #Correction of the value before tearing. Default 1. Method to be improved.
            
            #Extracting Data
            while (x[-1]==0) and (y[-1]==0) :
                x=np.delete (x, -1, 0)
                y=np.delete (y, -1, 0)
            
            # Selection Range
            
            col_f,col_f1 = st.columns(2)
            with col_f:
                f = int(st.slider(":orange[End of the selection range for Young's modulus and Elastic Limit]", 
                                  min_value=0,
                                  max_value=len(x)-1,
                                  value = round((len(x)-1)/2),step=1))
                with st.expander("Help?"):
                    st.write(v0_1_explainations['YMAndELSelectionRange'])
            ## Plot
            with col_f1:
            
                fig_selectionRange, ax_selectionRange = plt.subplots()
                ax_selectionRange.plot(x,y,'+')
                ax_selectionRange.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize= font)
                ax_selectionRange.set_xlabel('Strain (%)',fontsize= font)
                ax_selectionRange.set_ylabel('Stress (MPa)',fontsize= font)
                ax_selectionRange.axvspan(0,x[f], alpha = 0.5 , color='orange')
                               
                st.pyplot(fig_selectionRange,clear_figure=True)
            
            #Figures            
            fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))            

            #Shear Stress vs Strain plot
            ax1.plot(x,y)
            ax1.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize=font)
            ax1.set_xlabel('Strain (%)',fontsize=font)
            ax1.set_ylabel('Stress (MPa)',fontsize=font)
            # mplt.show()
        
        
            #Take only the half of the curve to avoid tear point
            x50 = x[1:f]#*round(len(x)/curveSelectionrange)]
            y50 = y[1:f]#*round(len(y)/curveSelectionrange)]
            ax2.plot(x50,y50,'x')
    
            #Polyfit of the selected values
            coeffs1_50 = np.polyfit(x50, y50, polynomialFit)    #Get the fit. First order poylnomial order x. TO BE IMPROVED.
    
            # Get the x values for the fit at higher resolution
            xFit50 = np.linspace(x50[1], x50[len(x50)-1])
            # Get the estimated y values
            yFit50 = np.polyval(coeffs1_50, xFit50)
    
            #Plot them as a line
            ax2.plot(xFit50, yFit50, linewidth=3)
            ax2.set_title('Fitted data: Stress vs Strain',fontsize=font)
            ax2.set_xlabel('Strain (%)',fontsize=font)
            ax2.set_ylabel('Stress (MPa)',fontsize=font)
            # mplt.show()
    
    
            #Finding the elastic limit and calculating the Young modulus
    
            #Selection of the values to be kept for the curves fitting
            index1 = round((cutcurvefite/100) * len(xFit50))        #Percent of the curve to be discarded on each side for the fit
            index2 = round(((100-cutcurvefite)/100) * len(xFit50))  #Percent of the curve to be kept for the fit
            #index1=0
            #index2=len(xFit50)
        
            #Creating a class to store the slopes of the first and second line fit, and theur difference
            class Lines:
                def __init__(self, slopeDifferences, line1, line2):
                    self.slopeDifferences = slopeDifferences
                    self.line1 = line1
                    self.line2 = line2
            
            #Creating the list to store all these data
            lineData = []
    
            #Fitting the selected part of the curve
            for k in range(index1, index2):
                #Get data in left side
                x1 = xFit50[1:k]
                y1 = yFit50[1:k]
                #Fit a line through the left side
                coefficients1 = np.polyfit(x1, y1, 1)
                #Get data in right side
                x2 = xFit50[k+1:len(xFit50)]
                y2 = yFit50[k+1:len(yFit50)]
                #Fit a line through the right side
                coefficients2 = np.polyfit(x2, y2, 1)
                #Compute difference in slopes, and store in the list of lineData along with line equation coefficients
                slopeDiff = abs(coefficients1[1] - coefficients2[1])
                lineData.append ( Lines ( slopeDiff, coefficients1, coefficients2))
                #If we want to print them
                #print(k, lineData[k-index1-1].slopeDifferences, ', ', lineData[k-index1-1].line1 )
        
            #Find index for which the slope difference is the greatest 
            allslopeDiff = np.zeros([index2-index1-1])
            for k in range(0 , index2-index1-1):
                allslopeDiff[k] = lineData[k].slopeDifferences
    
            maxslopeDiff = max ( allslopeDiff)
            indexOfMaxSlopeDiff = np.argmax( allslopeDiff)
            st.write (indexOfMaxSlopeDiff, maxslopeDiff)
    
            #Plot the slopes and slope's differences.
            ax3.plot(xFit50, yFit50)     #Plot the fitted data.
            #We can add grids if we want
            #    grid on
            ax3.set_xlabel('Strain (%)',fontsize=font)
            ax3.set_ylabel('Stress (MPa)',fontsize=font)
            #Use the equation of line1 to get fitted/regressed y1 values.
            slope1 = lineData[indexOfMaxSlopeDiff].line1[0]
            intercept1 = lineData[indexOfMaxSlopeDiff].line1[1]
            y1Fitted = slope1 * xFit50 + intercept1
    
            #Plot line 1 over/through data.
            ax3.plot(xFit50, y1Fitted, linewidth = 2, color='red')
    
            #Use the equation of line2 to get fitted/regressed y2 values.
            slope2 = lineData[indexOfMaxSlopeDiff].line2[0]
            intercept2 = lineData[indexOfMaxSlopeDiff].line2[1]
            y2Fitted = slope2 * xFit50 + intercept2
    
            #Plot line 2 over/through data.
            ax3.plot(xFit50, y2Fitted, linewidth= 2, color='red')
            xc = (intercept2 - intercept1) / (slope1 - slope2)
            ax3.axvline(xc, color='magenta')
            ax3.set_title('Plot of linear regressions overlaid',fontsize=font)
            # mplt.show()
        
            #Young Modulus calculations as the slope of the first curve. Units: kPa
            YoungModulus = slope1 * 1000
            MaxStress = max(y1Fitted)
            xYM = np.linspace(0, (MaxStress/slope1) ,100)
            yYM = xYM*YoungModulus/1000

            ax3.plot(xYM, yYM, color='green')
            #Elastic deformation as the intercept of the two lines.
            ElasticDeformation = xc
    
            #Tensile strength and tear load
            #---------------------------------------------------------------------------------------------------------------------
            #Default for half tear selection range
            xMax = x[round(len(x)/tearSelectionRange):len(x)]
            yMax = y[round(len(y)/tearSelectionRange):len(x)]
        
            dyMax=np.gradient(yMax)/np.gradient(xMax)             #Derivative of the Max curve.
    
            #Get rid of infinite values. TEST
            #Changing dyMax from a list to an array
            dyMaxArr=np.array(dyMax)
            #Getting the indexes of very high values (considered infinite) -> method to change if there is 'inf'
            index=np.where(dyMaxArr > 10E6 )   #Change the value for higher ranges (initiial value = 10E6 )
            dyMaxArr= np.delete(dyMaxArr, index ,axis = 0)
    
            #Find the point where the change in gradient is max
            #Array to stock the difference in gradient
            dyMaxVar = np.zeros([len(dyMax)-1])
            for k in range(0, len(dyMax)-2):
                dyMaxVar[k]=dyMax[k+1]-dyMax[k]
    
            #Finding the max and its index
            max_dyMaxVar = max ( dyMaxVar )
            indexOfmax_dyMaxVar = np.argmax( dyMaxVar )
            st.write('indexOfmax_dyMaxVar: ', indexOfmax_dyMaxVar, 'max_dyMaxVar: ',max_dyMaxVar)
            
            #Index of the value where the tear load is

            spi = st.slider("Choose the index of the value where you want the tear load to be",min_value =1, max_value= len(x))
            
            fig_tearLoad,ax_tearLoad = plt.subplots()
            ax_tearLoad.plot(x,y)    
            if len(x)>10:
                ax_tearLoad.plot(x[-10], y[-10], 'r+') #1
                ax_tearLoad.annotate('10', (x[-10], y[-10]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>30:           
                ax_tearLoad.plot(x[-30], y[-30], 'r+') #2
                ax_tearLoad.annotate('30', (x[-30], y[-30]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>50:
                ax_tearLoad.plot(x[-50], y[-50], 'r+') #3
                ax_tearLoad.annotate('50', (x[-50], y[-50]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>100:
                ax_tearLoad.plot(x[-100], y[-100], 'r+') #4
                ax_tearLoad.annotate('100', (x[-100], y[-100]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>200:
                ax_tearLoad.plot(x[-200], y[-200], 'r+') #5
                ax_tearLoad.annotate('200', (x[-200], y[-200]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>500:
                ax_tearLoad.plot(x[-500], y[-500], 'r+') #6
                ax_tearLoad.annotate('500', (x[-500], y[-500]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            ax_tearLoad.axvline(x[-spi], color='magenta')
            ax_tearLoad.plot(x[-spi], y[-spi], '*', color='yellow')
            ax_tearLoad.set_xlabel('Strain (%)',fontsize=font)
            ax_tearLoad.set_ylabel('Stress (MPa)',fontsize=font)
            ax_tearLoad.set_title('Tearing point',fontsize=font)  
            st.pyplot(fig_tearLoad)
            
            ax4.plot(x,y)    
            if len(x)>10:
                ax4.plot(x[-10], y[-10], 'r+') #1
                ax4.annotate('10', (x[-10], y[-10]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>30:           
                ax4.plot(x[-30], y[-30], 'r+') #2
                ax4.annotate('30', (x[-30], y[-30]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>50:
                ax4.plot(x[-50], y[-50], 'r+') #3
                ax4.annotate('50', (x[-50], y[-50]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>100:
                ax4.plot(x[-100], y[-100], 'r+') #4
                ax4.annotate('100', (x[-100], y[-100]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>200:
                ax4.plot(x[-200], y[-200], 'r+') #5
                ax4.annotate('200', (x[-200], y[-200]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            if len(x)>500:
                ax4.plot(x[-500], y[-500], 'r+') #6
                ax4.annotate('500', (x[-500], y[-500]), textcoords="offset points", xytext=(0,10),ha='center',fontsize=font/2) 
            ax4.axvline(x[-spi], color='magenta')
            ax4.plot(x[-spi], y[-spi], '*', color='yellow')
            ax4.set_xlabel('Strain (%)',fontsize=font)
            ax4.set_ylabel('Stress (MPa)',fontsize=font)
            ax4.set_title('Tearing point',fontsize=font)  
            
            
            #Breaking index on the tearSelectionRange, strain and stress
            BreakIndex = -spi
            #Tensile strenght in N/mm2 (conversion: 1 MPa = 1 N/mm2).
            TensileStrength = max (yMax)
        
            #Tear Load (limite de rupture) in N
            Stressatbreak = y[-spi]
            mm2Value = float(SpecimenWidth) * float(SpecimenThickness)  #Area of the measure.
            TearLoad = y[-spi] * mm2Value 
        
            #Percentage elongation at break 
            PercentageElongBreak = x[-spi]
        
    #Wrote repport
    #print('STRESS AT BREAK: ' + str(Stressatbreak) )
    st.header("Results")
    with st.expander("Text"):
        st.write('Report of the extracted values, curve: ' + SpecimenName)
        st.write("Young Modulus: "+ str(YoungModulus) +" kPa")
        st.write("Elastic deformation limit: " + str(ElasticDeformation) + " %")
        st.write("Tensile strength: "+ str(TensileStrength) +" MPa")
        st.write("Tear load: "+ str(TearLoad) +" N")
        st.write("Percentage of elongation at break: "+ str(PercentageElongBreak) +" %")
        st.write(" ")

    results = {"Specimen Number": [SpecimenNumber],"Name of file" : [SampleName], "%Polymer": [""], "Porogen": [""], "Cales":[""],
              "Thickness":[SpecimenThickness] , "Length":[SpecimenLength] , "Height":[SpecimenWidth] ,
              "Young modulus (kPa)":[str(YoungModulus)],
              "Elastic Limit(%)":[ElasticDeformation],
              "Tensile Strength (MPa)":[TensileStrength], "Tear load (N)": [TearLoad],
                                        "Percentage of elongation at break (%)":[PercentageElongBreak]}
    file = pd.DataFrame(data=results)
    with st.expander("Dataframe"):
        st.dataframe(results) 
    
    st.download_button(
        label= "Download CSV",
        data = file.to_csv().encode('utf-8'),
        file_name ="Results_" + str(SpecimenNumber) + ".csv"
        )
    
    
    fig.tight_layout()
    st.header("Graphs")
    st.pyplot(fig, clear_figure=True)
            
    fig.tight_layout()
    st.header("Graphs")
    st.pyplot(fig, clear_figure=True)
    return

def code(uploaded_file):
    df_SpecimenSpecs, df_data, SampleName = getInformations(uploaded_file)
    with st.container():
        st.header("Data Processing: " + df_SpecimenSpecs['Specimen Name'][0][len(SampleName)+1:])
        st.write(df_SpecimenSpecs) #.style.hide_index().format(decimal='.', precision=2).to_html(), unsafe_allow_html= True)
        process(df_SpecimenSpecs, df_data, SampleName)
    return
        
        