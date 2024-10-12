import numpy as np
import matplotlib.pyplot as plt
import csv
import streamlit as st
import streamlit.components.v1 as  components


import pandas as pd
from pandas import DataFrame
from tabulate import tabulate
import plotly.graph_objects as go
import ruptures as rpt

class Specimen:
#Each specimen has a name, thickness, width, length, xdata, ydata, Young's modulus, elastic deformation, tensile strength, tear load and percentage of elongation at break
    def __init__(self, SN, ST, SW, SL, xdata, ydata, YM, ED, TS, TL, SAB, PEB):
        self.SN = SN
        self.ST = ST
        self.SW = SW
        self.SL = SL
        self.xdata = xdata
        self.ydata = ydata
        self.YM = YM
        self.ED = ED
        self.TS = TS
        self.TL = TL
        self.SAB = SAB
        self.PEB = PEB        
ListSpecimen = []     


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
    # extracting the strain (x) and stress (y)
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
def process(df_SpecimenSpecs, df_data,SampleName): 
    #Extracting data from both DataFrames 
    SpecimenName = df_SpecimenSpecs['Specimen Name'].values[0]
    SpecimenNumber = SpecimenName[len(SampleName)+1:]
    SpecimenThickness = df_SpecimenSpecs['Specimen Thickness'].values[0]
    SpecimenWidth = df_SpecimenSpecs['Specimen Width'].values[0]
    SpecimenLength = df_SpecimenSpecs['Specimen Length'].values[0]
    x=df_data.values[:,4]
    y=df_data.values[:,3]
    
    ##Display Settings
    font= 7
    
    #Data processing (fixed for all specimens)
    curveSelectionrange=2          #Sections of the curve for Young Modulus and Elastic Limit. Default: 2.
    polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
    tearSelectionRange=2           #Sections of the curve for tensile strenght. Default: 2, second half.
    
    s= Specimen(SpecimenName, SpecimenThickness, SpecimenWidth, SpecimenLength, x, y,0,0,0,0,0,0)
     
    #For normal traction test :
    # La norme ISO 527-2 définit le module comme la pente de la courbe entre 0,05 % et 0,25 % de déformation à l’aide d’une corde 
    # ou d’un calcul de pente de régression linéaire. Comme le calcul du module commence à 0,05 % de déformation, il est extrêmement 
    # important que des précontraintes appropriées soient appliquées au matériau pour éliminer tout relâchement ou toute force de 
    # compression induite par la préhension de l’échantillon. Il ne doit pas dépasser 0,05 % de la déformation ou 1 % de la résistance
    # du matériau à la traction.
    # Source : https://www.instron.com/fr-fr/testing-solutions/iso-standards/iso-527-2
        #Finding the Tensile strength in N / mm^2
    
    # Marianne has chosen to remove the toe part of the curve (taking ou 5% from the beinning) 
    # This part of the curve can be eliminated with prestress
    # The percentage might be a lot as the tensile strengh is low compared to normal traction tests on steel
    # However, those 5% has to be limited between the beginning and another value as the strain with the maximum value of Stress or at the rupture
    # Indeed, the machine does not stop at the rigth moment and those 5% would represent more than it
        
    #Tensile strength in N/mm2 (conversion: 1 MPa = 1 N/mm2).
    TensileStrength = np.max(y)
    i_TensileStrength = np.argmax(y)
    
    ## Cas 0.05 % du déplacement à la rupture
    
    ## Cas 1 % de la contrainte maximale
    
    ## Cas Removing the toe part of the curve (taking out 5% from the beginning between it and the strain with maximum stress)
    
    
    
    
    #Removing the toe part of the curve (taking out 5% from the beginning)
    
    #Creation of those copies of x and y named X and Y where those 5 % and the negative stress values (Y) are removed
    iStressEnd = len(y)-1
    while y[iStressEnd] <= 0:
        iStressEnd -=1
    i_pretension = round(0.05*len(x))
    x_pretention = x[i_pretension]
    y_pretention = y[i_pretension]
    
    X= x[i_pretension:1+ iStressEnd] - x_pretention * np.ones(iStressEnd+1 - i_pretension)
    Y= y[i_pretension:1+ iStressEnd] # - y_pretention * np.ones(iStressEnd+1 - i_pretension)
    
    
    
    

    
    
    
    
    # x=x[a:]
    # y=y[a:]
    
    # y0=y[0]
    
    # x=x-x[0]
    # y=y-y[0]
    
    #Initialization of Figure
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(10,10)) 
    
    #Shear Stress vs Strain plot
    ax1.plot(x,y)
    ax1.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize = font)
    ax1.set_xlabel('Strain (%)',fontsize = font)
    ax1.set_ylabel('Stress (MPa)',fontsize = font)
    
    #Tensile strength in N/mm2 (conversion: 1 MPa = 1 N/mm2).
    TensileStrength = max(y) + y0
    
    #Calculating the Young's modulus and the elastic limit
    #---------------------------------------------------------------------------------------------------------------------------------
    
    #Take only the half of the curve to avoid tear point
    x50 = x[1:round(len(x)/curveSelectionrange)]
    y50 = y[1:round(len(y)/curveSelectionrange)]

    #Polyfit of the selected values
    coeffs1_50 = np.polyfit(x50, y50, polynomialFit)

    # Get the x values for the fit at higher resolution
    xFit50 = np.linspace(x50[1], x50[len(x50)-1])
    # Get the corresponding y values for the fitted curve
    yFit50 = np.polyval(coeffs1_50, xFit50)

    #Plotting
    ax2.plot(x50,y50,'x')
    ax2.plot(xFit50, yFit50, linewidth=3)
    ax2.set_title('First part of the curve: \n Raw data & Fitted data',fontsize = font)
    ax2.set_xlabel('Strain (%)',fontsize = font)
    ax2.set_ylabel('Stress (MPa)',fontsize = font)

    #Selection of the values to be kept for the curves fitting
    index1 = round((5/100) * len(xFit50))
    index2 = round(((100-5)/100) * len(xFit50))
    
    
    x1 = x[find_nearest(x, 0.05)]
    y1 = y[find_nearest(x, 0.05)]
    x2 = x[find_nearest(x, 0.25)]
    y2 = y[find_nearest(x, 0.25)]    
    
    #Young's modulus
    YoungModulus = ((y2-y1)/(x2-x1))
    
    #Plotting
    MaxStress = max(y)
    xYM = np.linspace(0, (MaxStress/YoungModulus) ,100)
    yYM = xYM*YoungModulus
    
    ax3.plot(x,y,'.')
    ax3.plot(xYM, yYM, color='red')
    ax3.set_title("Stress vs strain curve \n + \n Line of slope the Young's modulus",fontsize = font)
    ax3.set_xlabel('Strain (%)',fontsize = font)
    ax3.set_ylabel('Stress (MPa)',fontsize = font)

    #Calculating the tensile strength, tear load and percentage of elongation at break
    #---------------------------------------------------------------------------------------------------------------------------------
    
    #Keeping the values of the second part of the curve
    xMax = x[round(len(x)/tearSelectionRange):len(x)]
    yMax = y[round(len(x)/tearSelectionRange):len(x)]    
    
    ax4.plot(xMax, yMax)
    ax4.set_title('Second part of the curve : \n Raw data & Moving Average Values & Fitted data',fontsize = font)
    ax4.set_xlabel('Strain (%)',fontsize = font)
    ax4.set_ylabel('Stress (MPa)',fontsize = font)
    
    #Smoothing the curve
    window_size = 4
  
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
  
    # Loop through the array t o
    #consider every window of size 3
    while i < len(yMax) - window_size + 1:
  
        # Calculate the average of current window
        window_average = round(np.sum(yMax[i:i+window_size]) / window_size, 2)
      
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
      
        # Shift window to right by one position
        i += 1
      
    yMax = moving_averages
    xMax = xMax[2:len(xMax)-1]
    
    coeffs_movav = np.polyfit(xMax, yMax, 6)
    # Get the x values for the fit at higher resolution
    xFit_movav = np.linspace(xMax[1], xMax[len(xMax)-1])
    # Get the corresponding y values for the fitted curve
    yFit_movav = np.polyval(coeffs_movav, xFit_movav)
    
    
    #Plotting moving averages
    ax4.plot(xMax, yMax)
    ax4.plot(xFit_movav, yFit_movav)

        
    
    xMax = xFit_movav
    yMax = yFit_movav
    
    #Tear load
    
    #Calculating the gradient
    dyMax=np.gradient(yMax)/np.gradient(xMax) 
    
    
    #Finding the point where the gradient is max
    MaxDerivative = max (abs(dyMax))
    indexOfmax_MaxDerivative = np.argmax( abs(dyMax) )
    
    #Tear load (Limite de rupture) (N)
    StressAtBreak = yMax[indexOfmax_MaxDerivative-2] + y0
    mm2Value = float(SpecimenWidth) * float(SpecimenThickness)  #Area of the measure.
    TearLoad = StressAtBreak * mm2Value 
    
    #Percetnage of elongation at break
    PercentageElongBreak = xMax[indexOfmax_MaxDerivative-2]
    
    ax5.plot(x,y,'.')
    ax5.axvline(xMax[indexOfmax_MaxDerivative], color='orange')
    ax5.plot(PercentageElongBreak, StressAtBreak, '*', color='yellow')
    ax5.set_xlabel('Strain (%)',fontsize = font)
    ax5.set_ylabel('Stress (MPa)',fontsize = font)
    ax5.set_title('Tearing point',fontsize = font)
    
    #     #REPORT-------------------------------------------------------------------
    # #Module d'Young (pente de la partie linéaire) (N/mm2)
    # #Limite élastique (N)
    # #Tensile strength (le max de la courbe) (N/mm2)
    # #Tear Load (limite de rupture) (N)
    # #Percentage elongation at break  (%)
    # print('Report of the extracted values, curve: ' + SpecimenName)
    # print("Young Modulus: "+ str(YoungModulus*1000) +" kPa")
    # print("Tensile strength: "+ str(TensileStrength) +" MPa")
    # print("Stress at tear: " + str(StressAtBreak) + " MPa") 
    # print("Tear load: "+ str(TearLoad) +" N")
    # print("Percentage of elongation at break: "+ str(PercentageElongBreak) +" %")
    # print(" ")

    s.YM = YoungModulus*1000
    s.TS = TensileStrength
    s.TL = TearLoad
    s.SAB = StressAtBreak
    s.PEB = PercentageElongBreak
    
    #Wrote repport
    #print('STRESS AT BREAK: ' + str(Stressatbreak) )
    st.header("Results")
    with st.expander("Text"):
        st.write('Report of the extracted values, curve: ' + SpecimenName)
        st.write("Young Modulus: "+ str(YoungModulus*1000) +" kPa")
        st.write("Tensile strength: "+ str(TensileStrength) +" MPa")
        st.write("Stress at tear: " + str(StressAtBreak) + " MPa")
        st.write("Tear load: "+ str(TearLoad) +" N")
        st.write("Percentage of elongation at break: "+ str(PercentageElongBreak) +" %")
        st.write(" ")

    # SampleName=SpecimenName
    # while SampleName[-1] != "_":
    #     SampleName = SampleName[:-1]
    # SampleName = SampleName[:-1]
    results = {"Specimen Number": [SpecimenNumber],"Name of file" : [SampleName], "%Polymer": [""], "Porogen": [""], "Cales":[""],
              "Thickness":[SpecimenThickness] , "Length":[SpecimenLength] , "Height":[SpecimenWidth] ,
              "Young modulus (kPa)":[str(YoungModulus*1000)], 
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
    
    return #s, fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6))


def SheetCode(uploaded_files): ## Only table for google sheet
    st.write('Which sample to display?')
    FileList = [st.checkbox(uploaded_file.name,value=True) for uploaded_file in uploaded_files]

    SpecimenNumbers,SampleNames,SpecimenThicknesses, SpecimenLengths, SpecimenWidths, YMs, TSs, TLs, PEBs = [],[],[],[],[],[],[],[],[]


    for i in range(len(uploaded_files)):
        if FileList[i]:
            df_SpecimenSpecs, df_data, SampleName = getInformations(uploaded_files[i])
            #Extracting data from both DataFrames 
            SpecimenName = df_SpecimenSpecs['Specimen Name'].values[0]
            SpecimenNumber = int(SpecimenName[len(SampleName)+1:])
            SpecimenThickness = df_SpecimenSpecs['Specimen Thickness'].values[0]
            SpecimenWidth = df_SpecimenSpecs['Specimen Width'].values[0]
            SpecimenLength = df_SpecimenSpecs['Specimen Length'].values[0]
            x=df_data.values[:,4]
            y=df_data.values[:,3]
            
            #Data processing (fixed for all specimens)
            curveSelectionrange=2          #Sections of the curve for Young Modulus and Elastic Limit. Default: 2.
            polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
            tearSelectionRange=2           #Sections of the curve for tensile strenght. Default: 2, second half.
                
            s= Specimen(SpecimenName, SpecimenThickness, SpecimenWidth, SpecimenLength, x, y,0,0,0,0,0,0)
            
            #Removing extra zeros from x and y
            while (x[-1]==0) and (y[-1]==0) :
                x=np.delete (x, -1, 0)
                y=np.delete (y, -1, 0)
                
            #Removing the toe part of the curve (taking out 5% from the beginning)
            #This part of the curve can be eliminated with prestress
            a= round(0.05*len(x))
            x=x[a:]
            y=y[a:]
            
            y0=y[0]
            
            x=x-x[0]
            y=y-y[0]
                        
            #Tensile strength in N/mm2 (conversion: 1 MPa = 1 N/mm2).
            TensileStrength = max(y) + y0
            
            #Calculating the Young's modulus and the elastic limit
            #---------------------------------------------------------------------------------------------------------------------------------
            
            #Take only the half of the curve to avoid tear point
            x50 = x[1:round(len(x)/curveSelectionrange)]
            y50 = y[1:round(len(y)/curveSelectionrange)]

            #Polyfit of the selected values
            coeffs1_50 = np.polyfit(x50, y50, polynomialFit)

            # Get the x values for the fit at higher resolution
            xFit50 = np.linspace(x50[1], x50[len(x50)-1])
            # Get the corresponding y values for the fitted curve
            yFit50 = np.polyval(coeffs1_50, xFit50)

            #Selection of the values to be kept for the curves fitting
            index1 = round((5/100) * len(xFit50))
            index2 = round(((100-5)/100) * len(xFit50))
               
            x1 = x[find_nearest(x, 0.05)]
            y1 = y[find_nearest(x, 0.05)]
            x2 = x[find_nearest(x, 0.25)]
            y2 = y[find_nearest(x, 0.25)]    
            
            #Young's modulus
            YoungModulus = ((y2-y1)/(x2-x1))
            
            MaxStress = max(y)
            xYM = np.linspace(0, (MaxStress/YoungModulus) ,100)
            yYM = xYM*YoungModulus
            

            #Calculating the tensile strength, tear load and percentage of elongation at break
            #---------------------------------------------------------------------------------------------------------------------------------
            
            #Keeping the values of the second part of the curve
            xMax = x[round(len(x)/tearSelectionRange):len(x)]
            yMax = y[round(len(x)/tearSelectionRange):len(x)]    
                        
            #Smoothing the curve
            window_size = 4
          
            i = 0
            # Initialize an empty list to store moving averages
            moving_averages = []
          
            # Loop through the array t o
            #consider every window of size 3
            while i < len(yMax) - window_size + 1:
          
                # Calculate the average of current window
                window_average = round(np.sum(yMax[i:i+window_size]) / window_size, 2)
              
                # Store the average of current
                # window in moving average list
                moving_averages.append(window_average)
              
                # Shift window to right by one position
                i += 1
              
            yMax = moving_averages
            xMax = xMax[2:len(xMax)-1]
            
            coeffs_movav = np.polyfit(xMax, yMax, 6)
            # Get the x values for the fit at higher resolution
            xFit_movav = np.linspace(xMax[1], xMax[len(xMax)-1])
            # Get the corresponding y values for the fitted curve
            yFit_movav = np.polyval(coeffs_movav, xFit_movav)
            
            xMax = xFit_movav
            yMax = yFit_movav
            
            #Tear load
            
            #Calculating the gradient
            dyMax=np.gradient(yMax)/np.gradient(xMax) 
            
            
            #Finding the point where the gradient is max
            MaxDerivative = max (abs(dyMax))
            indexOfmax_MaxDerivative = np.argmax( abs(dyMax) )
            
            #Tear load (Limite de rupture) (N)
            StressAtBreak = yMax[indexOfmax_MaxDerivative-2] + y0
            mm2Value = float(SpecimenWidth) * float(SpecimenThickness)  #Area of the measure.
            TearLoad = StressAtBreak * mm2Value 
            
            #Percetnage of elongation at break
            PercentageElongBreak = xMax[indexOfmax_MaxDerivative-2]
            
            
            
            #Wrote repport
            #print('STRESS AT BREAK: ' + str(Stressatbreak) )
            
            
            
            
            SpecimenNumbers.append(SpecimenNumber)
            SampleNames.append(SpecimenName)
            SpecimenThicknesses.append(SpecimenThickness)
            SpecimenLengths.append(SpecimenLength)
            SpecimenWidths.append(SpecimenWidth)
            YMs.append(YoungModulus*1000)
            TSs.append(TensileStrength)
            TLs.append(TearLoad)
            PEBs.append(PercentageElongBreak)
            
    ## Mean ##
    SpecimenNumbers.append(None)
    SampleNames.append("Mean")
    SpecimenThicknesses.append(np.mean(SpecimenThicknesses))
    SpecimenLengths.append(np.mean(SpecimenLengths))
    SpecimenWidths.append(np.mean(SpecimenWidths))
    YMs.append(np.mean(YMs))
    TSs.append(np.mean(TSs))
    TLs.append(np.mean(TLs))
    PEBs.append(np.mean(PEBs))
    n=len(SpecimenNumbers)+2
    
    ## STd ##
    SpecimenNumbers.append(None)
    SampleNames.append("Standard Deviation")
    SpecimenThicknesses.append(np.std(SpecimenThicknesses[:-1]))
    
    # st.write(SpecimenThicknesses[:-1],SpecimenThicknesses)
    
    SpecimenLengths.append(np.std(SpecimenLengths[:-1]))
    SpecimenWidths.append(np.std(SpecimenWidths[:-1]))
    YMs.append(np.std(YMs[:-1]))
    TSs.append(np.std(TSs[:-1]))
    TLs.append(np.std(TLs[:-1]))
    PEBs.append(np.std(PEBs[:-1]))
    
    results = {"Specimen Number": SpecimenNumbers,"Name of file" : SampleNames, #"%Polymer": [""]*n, "Porogen": [""]*n, "Cales":[""]*n,
                      "Thickness":SpecimenThicknesses , "Length":SpecimenLengths , "Height":SpecimenWidths ,
                      "Young modulus (kPa)":YMs, 
                      "Tensile Strength (MPa)":TSs, "Tear load (N)": TLs,
                                                "Percentage of elongation at break (%)":PEBs}
    file = pd.DataFrame(data=results)
    st.dataframe(results) 
    # st.table(results)
    return

    

###### Code ######
def code(uploaded_file):
    df_SpecimenSpecs, df_data, SampleName = getInformations(uploaded_file)
    with st.container():
        st.header("Data Processing: " + df_SpecimenSpecs['Specimen Name'][0][len(SampleName)+1:])
        st.write(df_SpecimenSpecs) #.style.hide_index().format(decimal='.', precision=2).to_html(), unsafe_allow_html= True)
        process(df_SpecimenSpecs, df_data,SampleName)
    return