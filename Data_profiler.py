import streamlit as st
# st.set_page_config(layout="wide",initial_sidebar_state="collapsed") 
# st.markdown( """ <style> [data-testid="collapsedControl"] { display: none } </style> """, unsafe_allow_html=True, )
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Page_layout import main_page
# from streamlit_extras.switch_page_button import switch_page
# from st_pages import Page, Section, add_page_title, show_pages, hide_pages
# from streamlit_card import card
# from streamlit_option_menu import option_menu
# selected_menu = option_menu(None, ["Home",'Topics'], 
#     icons=['house', 'briefcase'], 
#     menu_icon="cast", default_index=1, orientation="horizontal",
#     styles={
#         "container": {"padding": "0!important", "background-color": "dark blue"},
#         "icon": {"color": "white", "font-size": "25px"}, 
#         "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#75857e"},
#         "nav-link-selected": {"background-color": "navy Blue"},
#     }
# )
# if selected_menu=="Home":
#     switch_page("Topics")

# img="media/banner.png"
# st. image(img,use_column_width=True)
# st.divider()

def profiler(df,container):
    overview,variables,interaction,corelation,missing_values=container.tabs(('Overview','Variables','Interaction','Corelation','Missing Values'))
    with overview:
        overview.subheader("General Statistics")
        col1,col2=overview.columns((7,3))
        html=f"""<table>
                    <tr><th>Number of variables	</th><th>Number of observations</th><th>Duplicate Rows</th>
                    <th>Duplicate Rows Percentage</th><th>No of numirical variables </th><th>No of Categorical Variables</th></tr>
                    <tr><td>{df.shape[1]}</td><td>{df.shape[0]}</td> <td>{df.duplicated().sum()}</td>
                    <td>{(df.duplicated().sum()/df.shape[0]*100) :.1f} %</td> <td>{len(df.select_dtypes(include=['int','float']).columns)}</td> <td>{len(df.select_dtypes(include=['object']).columns)}</td></tr></table>"""
        col1.markdown(html,unsafe_allow_html=True)
        html="<tr><th>Column Name </th><th>Column Data Type</th></tr>"
        for col in df.columns:
            html=html + f'<tr> <td>{col} </td> <td>{df[col].dtype}</td></tr>'
        html=html+'</table>'
        # main_page.alignh(2,col2)
        col2.markdown(html,unsafe_allow_html=True)
    with variables:
        varname=variables.selectbox('Select variable',options=df.columns)
        col1,col2=variables.columns((7,3))
        col1.subheader(varname + ": Summary")
        if df[varname].dtype=='object':
            maxlength=0
            col_values = df[varname][df[varname].notnull()].tolist()
            for val in col_values:
                if  maxlength<=len(val):
                    maxlength=len(val)
            minlength=maxlength
            for val in col_values:
                if minlength>=len(val):
                   minlength=len(val)
            sum=0
            medlength=0
            for val in col_values:
                sum=sum+len(val)
            meanlength=sum/df.shape[0]
            html=f"""<table>
                        <tr><th>Distinct Values </th> <th>Missing Values </th><th>Missing Values percentage </th> </th><th> Maximum length </th><th> Minimum length </th>
                            <th> Mean length </th>	</tr>
                        <tr><td>{df[varname].nunique()}</td><td>{(df[varname].isnull().sum())}</td><td>{(df[varname].isnull().sum()/df.shape[0]*100) :.1f}%</td><td>{maxlength}</td><td>{minlength}</td><td>{meanlength :.2f}</td></tr></table>"""
            col1.markdown(html,unsafe_allow_html=True)
            Categories,plot=variables.tabs(('Statistics','Plot'))
            with Categories:
                Categories.subheader(varname + " : Categories")
                Categories.write(df[varname].value_counts())
            with plot :
                plot.subheader(varname + ": Plot")
                col1,col2=plot.columns((5,5))
                count=df[varname].value_counts()
                fig, ax = plt.subplots(figsize =(4, 4))
                plt.cla()
                count.plot.pie(autopct='%1.1f%%')
                plt.title(f'Pie of {varname}')
                col2.pyplot(fig)
                fig2, ax = plt.subplots(figsize =(5, 5))
                plt.cla()
                col_values = df[varname][df[varname].notnull()].tolist()
                plt.hist(col_values, bins=20, color='blue', edgecolor='black')
                # Add labels and title
                plt.xlabel(varname)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {varname}')
                col1.pyplot(fig2)

            # with common:
            #     common.subheader(varname + " : Common Values")
            #     # col1,col2=common.columns((3,7))
            #     # col1.subheader("Minimum 10 values")
            #     # col1.write(df[varname].nsmallest(10))
            #     # col2.subheader("Maximum 10 values")
            #     # col2.write((df[varname].nlargest(10)))

        else:
            html=f"""<table>
                        <tr><th>Distinct Values </th><th>Minimum value </th><th>Maximum value </th>
                        <th>Missing Values </th><th>Missing Values percentage </th><th>Mean</th></th><th>Median</th></tr>
                        <tr><td>{df[varname].nunique()}</td><td>{df[varname].min()}</td> <td>{df[varname].max()}</td>
                        <td>{(df[varname].isnull().sum()) }</td> <td>{(df[varname].isnull().sum()/df.shape[0]*100) :.1f}%</td> <td>{df[varname].mean():.2f}</td><td>{df[varname].median():.2f}</td></tr></table>"""
            col1.markdown(html,unsafe_allow_html=True)
            statistics,plot,common=variables.tabs(('Statistics','Plot','Common Values'))

            with statistics:
                col1,col2=statistics.columns((7,3))
                col1.subheader(varname + ": Statistics")
                col1.dataframe(df[varname].describe(),use_container_width=True,)
            with plot :
                
                col1,col2=plot.columns((4,3))
                col1.subheader(varname + ": Plot")
                fig, ax = plt.subplots(figsize =(5, 4))
                plt.hist(df[varname], bins=20, color='blue', edgecolor='black')
                # Add labels and title
                plt.xlabel(varname)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {varname}')

                col1.pyplot(fig)
            with common:
                common.subheader(varname + " : Common Values")
                col1,col2,col3,col4=common.columns((2,2,2,2))
                
                col1.subheader("Minimum 10 values")
                col1.write(df[varname].nsmallest(10))
                col2.subheader("Maximum 10 values")
                col2.write((df[varname].nlargest(10)))
                # df['A'].value_counts().head(10)
                col3.subheader("Most common 10 values")
                col3.write((df[varname].value_counts().sort_values().tail(10)))
                col4.subheader("Least common 10 values")
                col4.write((df[varname].value_counts().sort_values().head(10)))


    with interaction:
        interaction.markdown(" ### Findout interaction between variables")
        col1,col2=interaction.columns((5,5))
        numcol= list(df.select_dtypes(include=['int','float']).columns)
        if len(numcol)>1:
            intercol1=col1.selectbox("Select the first column",options=numcol)
            col2list=list(set(numcol)-set(intercol1))
            intercol2=col2.selectbox("Select the Seconed column",options=col2list)
            corr = df[intercol1].corr(df[intercol2])
            col1.markdown(f"### Correlation between  {intercol1} and   {intercol2} is  {corr}")
            fig, ax = plt.subplots(figsize =(15,8))
            plt.scatter(df[intercol1], df[intercol2])
            plt.xlabel(intercol1)
            plt.ylabel(intercol2)
            plt.title(f'Scatter plot {intercol1 } and {intercol2}')
            interaction.pyplot(fig)
        else:interaction.markdown(" ### Cannot Calculate interactions insufficent number of numiric columns ")
    
    with corelation:
        plt.clf()
        corfig,ax=plt.subplots(figsize =(10,4))
        numcol= list(df.select_dtypes(include=['int','float']).columns)
        
        if len(numcol)>=2:
            dfcopy=df[numcol]
            correl_method= corelation.selectbox("Correlation Calculation method ",options=["Auto","Spearman","Pearson","kendals"])


            if  correl_method=="Auto":
                corelation.subheader("Corelation Matrix")
                corelation.dataframe(dfcopy.corr())
                # Calculate the correlation matrix
                corr_matrix = dfcopy.corr()
                # Create a heatmap of the correlation matrix
                mask = np.zeros_like(corr_matrix)
                mask[np.triu_indices_from(mask)] = True
                # Create a heatmap of the correlation matrix with the upper triangle masked
                corelation.subheader("Heat Map")
                sns.heatmap(corr_matrix, annot=True, cmap='Paired', mask=mask)
                corelation.pyplot(corfig)
                
            elif  correl_method=="Spearman":
                corelation.subheader("Spearman Corelation Matrix")
                corelation.dataframe(dfcopy.corr(method="spearman"))
                # Calculate the correlation matrix
                corr_matrix = dfcopy.corr(method="spearman")
                mask = np.zeros_like(corr_matrix)
                mask[np.triu_indices_from(mask)] = True
                # Create a heatmap of the correlation matrix with the upper triangle masked
                corelation.subheader("Heat Map")
                sns.heatmap(corr_matrix, annot=True, cmap='Set2', mask=mask)
                corelation.pyplot(corfig)
            
            elif correl_method=="Pearson":
                corelation.subheader("Pearson Corelation Matrix")
                corelation.dataframe(dfcopy.corr(method="pearson"))
                # Calculate the correlation matrix
                corr_matrix =dfcopy.corr(method="pearson")
                mask = np.zeros_like(corr_matrix)
                mask[np.triu_indices_from(mask)] = True
                # Create a heatmap of the correlation matrix with the upper triangle masked
                corelation.subheader("Heat Map")
                sns.heatmap(corr_matrix, annot=True, cmap='Paired', mask=mask)
                corelation.pyplot(corfig)
                
            elif correl_method=="kendals":
                corelation.subheader("Kendal Corelation Matrix")
                corelation.dataframe(dfcopy.corr(method="pearson"))
                # Calculate the correlation matrix
                corr_matrix =dfcopy.corr(method="pearson")
                mask = np.zeros_like(corr_matrix)
                mask[np.triu_indices_from(mask)] = True
                # Create a heatmap of the correlation matrix with the upper triangle masked
                corelation.subheader("Heat Map")
                sns.heatmap(corr_matrix, annot=True, cmap='mako', mask=mask)
                corelation.pyplot(corfig)
        else:
            corelation.markdown("###  Cannot Calculate corelation matrix Insufficent number of numiric columns ")
         
    with missing_values:
        col1,col2=missing_values.columns((3 ,7))
        plt.clf()
        null_counts = df.isnull().sum()
       
        col1.markdown("## Sum of missing values in each columns")
        col1.dataframe(null_counts)
        nullpercentage=null_counts/df.shape[0]*100
        # plot null distribution for each column
        nullpercentage.plot(kind='bar')
        plt.title('Missing values Percentage Distribution for Each Column')
        plt.xlabel('Column')
        plt.ylabel('Percentage of Missing Values %')
        col2.pyplot(plt)

# st.title("Exploratory Data Analysis")
# st.subheader("Upload Your Dataset")
# file = st.file_uploader("Upload Your Dataset")
# if file: 
#     df = pd.read_csv(file, index_col=None)
#     # df.to_csv('dataset.csv', index=None)
#     st.subheader("Sample data")
#     st.dataframe(df)

#     profiler(df,st)                
#             # profile_df = df.profile_report()
#             # st.write(profile_df)
#             # st_profile_report(profile_df,height=700)
