import numpy as np
import matplotlib.pyplot as plt
import csv
import streamlit as st
import streamlit.components.v1 as  components
import numpy as np
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate
# import matplotlib.pyplot as plt
import ruptures as rpt
# import csv
import os 
from io import StringIO

class Specimen:
#Each specimen has a name, thickness, width, length, xdata, ydata, Young's modulus, elastic deformation, tensile strength, tear load and percentage of elongation at break
    def __init__(self, SN, ST, SW, SL, xdata, ydata, YM, ED, SEL, TS, TL, PEB):
        self.SN = SN
        self.ST = ST
        self.SW = SW
        self.SL = SL
        self.xdata = xdata
        self.ydata = ydata
        self.YM = YM
        self.ED = ED
        self.SEL = SEL
        self.TS = TS
        self.TL = TL
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


def process(df_SpecimenSpecs, df_data): 
    
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
    
    #Data processing (fixed for all specimens)
    curveSelectionrange=2          #Sections of the curve for Young Modulus and Elastic Limit. Default: 2.
    polynomialFit = 6              #Polynomial degree for fitting. Default: 6. Method to be improved...
    cutcurvefite = 5               #Percent of left points on each side. Default 5%.
    tearSelectionRange=2           #Sections of the curve for tensile strenght. Default: 2, second half.
    correctionBeforeBreak=1        #Correction of the value before tearing. Default 
        
    s= Specimen(SpecimenName, SpecimenThickness, SpecimenWidth, SpecimenLength, x, y,0,0,0,0,0,0)
    
    #Removing extra zeros from x and y
    while (x[-1]==0) and (y[-1]==0) :
        x=np.delete (x, -1, 0)
        y=np.delete (y, -1, 0)
        
    #Initialization of Figure
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(10,10)) 
    
    #Shear Stress vs Strain plot
    ax1.plot(x,y)
    ax1.set_title('Data analysis of: '+ SpecimenName + '\n Raw data: Stress vs Strain',fontsize = font)
    ax1.set_xlabel('Strain (%)',fontsize = font)
    ax1.set_ylabel('Stress (MPa)',fontsize = font)
    
    #Tensile strength in N/mm2 (conversion: 1 MPa = 1 N/mm2).
    TensileStrength = max (y) + y0
    
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
    
    #Creating a class to store the slopes of the first and second line fit, and their difference
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
        
    #Find index for which the slope difference is the greatest 
    allslopeDiff = np.zeros([index2-index1-1])
    for k in range(0 , index2-index1-1):
        allslopeDiff[k] = lineData[k].slopeDifferences
    
    maxslopeDiff = max ( allslopeDiff)
    indexOfMaxSlopeDiff = np.argmax( allslopeDiff)

    #Plot the slopes and slope's differences.
    ax3.plot(xFit50, yFit50)     #Plot the fitted data.
    #We can add grids if we want
    #    grid on
    ax3.set_xlabel('Strain (%)',fontsize = font)
    ax3.set_ylabel('Stress (MPa)',fontsize = font)
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
    ax3.set_title('Plot of linear regressions overlaid',fontsize = font)
    
    #Young Modulus calculations as the slope of the first curve. Units: kPa
    YoungModulus = slope1 * 1000
    
    #Elastic deformation as the intercept of the two lines.
    ElasticDeformation = xc
    indx = find_nearest(xFit50, xc)
    StressAtElasticDeformation = yFit50[indx]
    
    #Tensile strength and tear load
    #-----------------------------------------------------------------------------------------------------------------------------
    #Default:
    xMax = x[round(len(x)/tearSelectionRange):len(x)]
    yMax = y[round(len(y)/tearSelectionRange):len(x)]
    
    #If there is many tear points
    #xMax = x[round(len(x)/tearSelectionRange):round(4*len(x)/tearSelectionRange)]
    #yMax = y[round(len(y)/tearSelectionRange):round(4*len(x)/tearSelectionRange)]
    #mplt.plot(xMax,yMax,'.')
    #mplt.xlabel('Strain (%)')
    #mplt.ylabel('Stress (MPa)')
    #mplt.title('Part of the curve for breaking point')
    #mplt.show()
    
    dyMax=np.gradient(yMax)/np.gradient(xMax)             #Derivative of the Max curve.
    
    #Get rid of infinite values. TEST
    #Changing dyMax from a list to an array
    dyMaxArr=np.array(dyMax)
    #Getting the indexes of very high values (considered infinite) -> method to change if there is 'inf'
    index=np.where(dyMaxArr > 10E6 )   #Change the value for higher ranges (initiial value = 10E6 )
    dyMaxArr= np.delete(dyMaxArr, index ,axis = 0)
    
    #----------------------------------------------------------------------------------------------------------------------------
    #Detection with Matlab
    #    MaxBreak = ischange(dyMax,'linear');                    %Detect abrupt changes. Possible to change the detection method.
    #    BreakIndex = find(MaxBreak);                            %Find the non zero elements.
    #    if ~isempty(BreakIndex)
    #----------------------------------------------------------------------------------------------------------------------------
    
    #Find the point where the change in gradient is max
    #Array to stock the difference in gradient
    dyMaxVar = np.zeros([len(dyMax)-1])
    for k in range(0, len(dyMax)-2):
        dyMaxVar[k]=dyMax[k+1]-dyMax[k]
    
    #Finding the max and its index
    max_dyMaxVar = max ( dyMaxVar )
    indexOfmax_dyMaxVar = np.argmax( dyMaxVar )
    #print (indexOfmax_dyMaxVar, max_dyMaxVar)
    
    #Default method
    ax4.plot(x,y,'.')
    ax4.axvline(x[indexOfmax_dyMaxVar+round(len(x)/tearSelectionRange)], color='magenta')
    ax4.set_xlabel('Strain (%)',fontsize = font)
    ax4.set_ylabel('Stress (MPa)',fontsize = font)
    ax4.set_title('Tearing point',fontsize = font)
    #Manual point setting
    #mplt.plot(x,y,'.')    
    #mplt.plot(x[-30], y[-30], 'r+')
    #mplt.axvline(x[-30], color='magenta')
    #mplt.xlabel('Strain (%)')
    #mplt.ylabel('Stress (MPa)')
    #mplt.title('Tearing point')
        
    #Method to be changed-------------------------------------------------------------------------------------------------

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
    #Manual point setting 
    #Stressatbreak = y[-30]
    #TearLoad = y[-30] * mm2Value
    
    
    #Percentage elongation at break 
    PercentageElongBreak = x[originalIndexBreak-correctionBeforeBreak]

    #Plastic deformation of the material 
    PlasticDeformation = PercentageElongBreak - ElasticDeformation
        
    #REPORT-------------------------------------------------------------------
    #Module d'Young (pente de la partie linéaire) (N/mm2)
    #Limite élastique (N)
    #Tensile strength (le max de la courbe) (N/mm2)
    #Tear Load (limite de rupture) (N)
    #Percentage elongation at break  (%)*
    matrixMechanics = []
    matrixMechanics.append( SpecimenName )
    matrixMechanics.append( YoungModulus )
    matrixMechanics.append( ElasticDeformation )
    matrixMechanics.append( TensileStrength ) 
    matrixMechanics.append( TearLoad )
    matrixMechanics.append( PercentageElongBreak )

    #Wrote repport
    
    st.header("Results")
    with st.expander("Text"):
        #st.write('STRESS AT BREAK: ' + str(Stressatbreak) )
        st.write('Report of the extracted values, curve: ' + SpecimenName)
        st.write("Young Modulus: "+ str(YoungModulus*1000) +" kPa")
        st.write("Elastic limit: "+ str(ElasticDeformation) +" %")
        st.write("Stress at elastic limit: " + str(StressAtElasticDeformation) + " Mpa")
        st.write("Tensile strength: "+ str(TensileStrength) +" MPa")
        st.write("Tear load: "+ str(TearLoad) +" N")
        st.write("Percentage of elongation at break: "+ str(PercentageElongBreak) +" %")
        st.write(" ")        
    
    
    s.YM = YoungModulus
    s.ED = ElasticDeformation
    s.SEL = StressAtElasticDeformation
    s.TS = TensileStrength
    s.TL = TearLoad
    s.PEB = PercentageElongBreak

    SampleName=SpecimenName
    while SampleName[-1] != "_":
        SampleName = SampleName[:-1]
    SampleName = SampleName[:-1]
    
    results = {"Specimen Number": [SpecimenNumber],"Name of file" : [SampleName], "%Polymer": [""], "Porogen": [""], "Cales":[""],
              "Thickness":[SpecimenThickness] , "Length":[SpecimenLength] , "Height":[SpecimenWidth] ,
              "Young modulus (kPa)":[str(YoungModulus*1000)], "Elastic Limit(%)": [ElasticDeformation],
              "Stress at elastic limit(MPa)" : StressAtElasticDeformation,
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
    
    return # s, fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6))

###### Code ######
def code(uploaded_file):
    df_SpecimenSpecs, df_data, SampleName = getInformations(uploaded_file)
    with st.container():
        st.header("Data Processing: " + df_SpecimenSpecs['Specimen Name'][0][len(SampleName)+1:])
        st.write(df_SpecimenSpecs) #.style.hide_index().format(decimal='.', precision=2).to_html(), unsafe_allow_html= True)
        process(df_SpecimenSpecs, df_data)
    return



