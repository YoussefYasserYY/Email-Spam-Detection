import streamlit as st

def split(df):
    st.sidebar.header('Splitting Data')
    from sklearn.model_selection import train_test_split
    train_size = st.sidebar.slider("Select Test Size (as a percentage)", min_value=0.01, max_value=0.99, step=0.01, value=0.80)
    X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Category, train_size=train_size, random_state=42)
    train_size = int(len(X_train) / (len(X_train) + len(X_test))*100)
    test_size = int(len(X_test) / (len(X_train) + len(X_test))*100)
    cl1,cl2 = st.columns(2)
    st.sidebar.subheader(f'Train Size: {train_size}%')
    st.sidebar.subheader(f'Test Size: {test_size}%')
    col1 ,col2 = st.columns(2)
    col1.subheader('Feature')
    col1.write(X_train)
    col2.subheader('Target')
    col2.write(y_train)
    return X_train,y_train,X_test ,y_test
    
    
    
def train(X_train,y_train,X_test ,y_test):
    st.sidebar.subheader('Choosing Classifier')
    from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
    # classifier = st.sidebar.radio('Classifier',['Naive Bayes','Random Forest'],horizontal=True)
    classifier = st.sidebar.radio('Classifier',['Naive Bayes'],horizontal=True)
    
    # if classifier == 'Naive Bayes':
    trainer = st.sidebar.radio('Naive Bayes probabilistic algorithms',['Multinomial','Bernoulli'],horizontal=True)
    if trainer =='Multinomial':
        st.subheader("Naive Bayes Multinomial")
        st.markdown('''Naive Bayes Multinomial is a probabilistic classification algorithm based on Bayes' theorem and is specifically designed for text classification tasks where the features used for classification are discrete, such as word counts in a document or the frequency of terms in a document. It is a variation of the Naive Bayes algorithm that works well with data that has multiple categories and where each feature can take on a countable number of values (e.g., the number of times a word appears in a document).''')
        trainer =MultinomialNB()
    elif trainer == 'Bernoulli':
        st.subheader("Naive Bayes Bernoulli")
        st.markdown('''Naive Bayes Bernoulli is another variant of the Naive Bayes algorithm, specifically designed for binary data, where each feature is a binary attribute (typically 0 or 1). It's commonly used for tasks where you want to classify data into one of two classes, making it well-suited for problems like spam detection, sentiment analysis (positive or negative sentiment), and document categorization (e.g., relevant or irrelevant).''')
        trainer = BernoulliNB()
    # else:
    #     st.sidebar.header("Random Forest Classifier Parameters")
    #     n_estimators = st.sidebar.slider("**Number of Estimators**", 1, 1000, 100)
    #     st.sidebar.write('The number of decision trees in the forest.')
    #     max_depth = st.sidebar.slider("**Max Depth**", 1, 100, 10)
    #     st.sidebar.write('The maximum depth of each decision tree. It controls the depth of the tree and helps prevent overfitting.')
    #     min_samples_split = st.sidebar.slider("**Min Samples Split**", 2, 10, 2)
    #     st.sidebar.write('The minimum number of samples required to split an internal node.')
    #     min_samples_leaf = st.sidebar.slider("**Min Samples Leaf**", 1, 10, 1)
    #     st.sidebar.write('The minimum number of samples required to be in a leaf node.')
    #     from sklearn.ensemble import RandomForestClassifier
    #     # Create a RandomForestClassifier with user-defined parameters
    #     trainer = RandomForestClassifier(
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         min_samples_split=min_samples_split,
    #         min_samples_leaf=min_samples_leaf,
    #         random_state=42,
    #     )
    #     st.markdown('''Random Forest is a versatile and powerful ensemble machine learning algorithm that is used for both classification and regression tasks. It is based on the idea of creating multiple decision trees during training and combining their predictions to make more accurate and robust predictions. Random Forest is known for its high accuracy, scalability, and ability to handle large datasets with high-dimensional features.''')
    
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import Pipeline
    clf=Pipeline([
        ('vectorizer',CountVectorizer()),
        ('nb',trainer)
    ])
    st.write(X_train)
    if st.sidebar.button('Train',type = 'primary'):
        clf.fit(X_train,y_train)
        st.session_state['clf'] = clf
        st.success('Train Finished!')
    show = st.sidebar.checkbox('Show Metrics')
    if show:
        try:
            c1,c2 = st.columns(2)
            clf = st.session_state['clf']
            y_pred = clf.predict(X_test)
            from sklearn.metrics import  accuracy_score,precision_score, recall_score, f1_score
            from sklearn.metrics import classification_report
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            precision = precision_score(y_test, y_pred,pos_label='spam')
            recall = recall_score(y_test, y_pred,pos_label='spam')
            f1 = f1_score(y_test, y_pred,pos_label='spam')
            accuracy = accuracy_score(y_test, y_pred)
            c1.subheader(f'Accuracy {int(accuracy*100)}%')
            c1.subheader(f"Precision: {precision}")
            c1.subheader(f"Recall: {recall}")
            c1.subheader(f"F1-score: {f1}")
            report = classification_report(y_test, y_pred, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.drop('support', axis=1, inplace=True)

            # Create a heatmap to visualize the classification metrics
            c2.subheader("Classification Report:")
            c2.write(df)
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(df, annot=True, cmap='Blues', fmt=".2f")
            plt.xlabel('Metrics')
            plt.ylabel('Labels')
            plt.title('Classification Metrics')
            st.pyplot(fig)
        except:
            st.warning('Complete Training First')
    st.image('image4_v2LFcq0.max-1200x1200.png')
            

def test():
    mail = []
    
    st.markdown('''
    ##### Examples:
    ''')
    st.code('Sounds great! Are you home now?')
    st.code('''
    Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES  spam ''')
    mail.append(st.text_area('',placeholder='paste email to classify'))
    try:
        if st.button('Predict',type = 'primary'):
            clf = st.session_state['clf']
            y = clf.predict(mail)
            if y=='ham':
                st.success('Non-Spam')
            else:
                st.warning('Spam')
    except:
        st.warning('Finish Training First')


def ex(link):
    col1,col2 = st.columns([0.5,0.5])
    col1.markdown("Click button to open Jupyter Notebook 👉🏼")
    juputer= col2.button(f'{link} Code')
    if juputer:
        import subprocess
        subprocess.Popen(['jupyter', 'notebook',link+".ipynb"])