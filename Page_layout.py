
import streamlit as st
from PIL import Image


class main_page(object):
    def __init__(self) :
        #####Main Page s
        st.write("---")
        self.image=Image.open('media/bedo2.png')
        st.image(self.image,width=250)
        st.title('Bedo AI Computer Vision Virtual labs')
        st.write("---")
        #sidebar Content
        st.sidebar.image('media/images/nural2.jpeg',width=300)
        st.sidebar.header("What do you want to learn today")
        self.categories=st.sidebar.selectbox("Computer vision Virtual labs ",options=["Computer Vision"])

        self.learntype=st.sidebar.radio("learning environment",options=['Image Formation and Transformation','Image Processing',
        'Image Stabilization','Camera Calibration','Feature Extraction and Recognition',
        'Advanced 3D Scene Understanding','Distance Measures and Motion Analysis','Computational Photography and Rendering',
        'Projects'])





    #-------------------------------------------------------------------------------


    
    def select_dataset_Image_formation_Transformation():
        Datasets=st.selectbox("Actions",options=["Intro","Image Formation","Image Transformation"])
        return Datasets
    
    def select_dataset_Feature_Extraction_Recognition():
        Datasets=st.selectbox("Actions",options=["Intro","Feature Detection","Classification","Object Tracking","Action Recognition",'Detection','Teachable Machine'])
        return Datasets
    
    def select_dataset_Advanced_3DScene_Understanding():
        Datasets=st.selectbox("Actions",options=["Intro","Structure from Motion and SLAM (Simultaneous Localization and Mapping)","3D Reconstruction","Depth Estimation"])
        return Datasets

    def select_dataset_Computational_Photography_Rendering():
        Datasets=st.selectbox("Actions",options=["Intro",'Computational Photography','Image-Based Rendering'])
        return Datasets
    
    def select_dataset_Distance_Measures_MotionAnalysis():
        Datasets=st.selectbox("Actions",options=["Intro",'Distance Measures','Motion Estimation'])
        return Datasets
    
    def projects():
        Datasets=st.selectbox("Projects",options=["Projects List",'Cartoonize an Image','Simple Face Recognition System','Interactive Menu Selection using Hand Gestures','Automated Multiple Choice Answer Sheet Grader'],key = 'project')
        return Datasets


    def alignh(lines,colname):
       for i in range (1 , lines):
         colname.markdown("#")
