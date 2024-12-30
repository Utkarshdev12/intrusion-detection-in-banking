from flask import Flask,render_template,url_for,redirect,request,jsonify,render_template_string,flash
app = Flask(__name__)
import pandas as pd 
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib


import mysql.connector
mydb = mysql.connector.connect(
    host='localhost',
    port=3306,          
    user='root',        
    passwd='',          
    database='cyber'  
)

mycur = mydb.cursor()


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        address = request.form['Address']
        
        if password == confirmpassword:
            # Check if user already exists
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                # Insert new user without hashing password
                sql = 'INSERT INTO users (name, email, password, Address) VALUES (%s, %s, %s, %s)'
                val = (name, email, password, address)
                mycur.execute(sql, val)
                mydb.commit()
                
                msg = 'User registered successfully!'
                return render_template('registration.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[2]  
            # Check if the password matches the stored password
            if password == stored_password:
                return redirect('/upload')
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')



app.secret_key = "your_secret_key"  
UPLOAD_FOLDER = r'FRONTEND\WSN-DS.csv'  

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set upload folder configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for uploading the data
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Save file and process it
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # Read the CSV file using pandas
            try:
                data = pd.read_csv(filename)
                # Data is successfully read, you can further process it
                flash('Data uploaded and processed successfully!')
            except Exception as e:
                flash(f'Failed to read the file: {str(e)}')
            return redirect(request.url)
    return render_template('upload.html')




# Route to view the data
@app.route('/viewdata')
def viewdata():
    # Load the dataset
    dataset_path = 'WSN-DS.csv'  # Make sure this path is correct to the uploaded file
    df = pd.read_csv(dataset_path)
    df = df.head(1000)

    # Convert the dataframe to HTML table
    data_table = df.to_html(classes='table table-striped table-bordered', index=False)

    # Render the HTML page with the table
    return render_template('viewdata.html', table=data_table)



# Load CSV file into a DataFrame
df = pd.read_csv('WSN-DS.csv')

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in df.columns:
    if df[col].dtype == 'object':  # Check if the column is categorical (object type)
        df[col] = le.fit_transform(df[col])

# Split dataset
x = df.drop('Attack type', axis=1)
y = df['Attack type']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    
}

# Route for the algorithm selection
@app.route('/algo', methods=['GET', 'POST'])
def algo():
    selected_model = None
    accuracy = None
    confusion_matrix_ = None
    classification_report_ = None
    if request.method == 'POST':
        # Get the selected model from the dropdown
        selected_model = request.form['model']
        # Train the selected model and evaluate it
        if selected_model in models:
            model = models[selected_model]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            confusion_matrix_ = confusion_matrix(y_test, y_pred)
            classification_report_ = classification_report(y_test, y_pred, output_dict=True)
    return render_template('algo.html', models=list(models.keys()), selected_model=selected_model, accuracy=accuracy, confusion_matrix_=confusion_matrix_, classification_report_=classification_report_)







# Load your model (ensure the path is correct)
rf_model = joblib.load('random_forest_model.pkl')
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    attack_type = None
    if request.method == 'POST':
        # Get input values from the form
        abc = [
            int(request.form['id']),
            int(request.form['Time']),
            int(request.form['Is_CH']),
            int(request.form['who_CH']),
            float(request.form['Dist_To_CH']),
            int(request.form['ADV_S']),
            int(request.form['ADV_R']),
            int(request.form['JOIN_S']),
            int(request.form['JOIN_R']),
            int(request.form['SCH_S']),
            int(request.form['SCH_R']),
            int(request.form['Rank']),
            float(request.form['DATA_S']),
            float(request.form['DATA_R']),
            float(request.form['Data_Sent_To_BS']),
            float(request.form['dist_CH_To_BS']),
            int(request.form['send_code']),
            float(request.form['Expaned_Energy'])
        ]
        
        # Predict
        result = rf_model.predict([abc])[0]
        if result == 0:
            attack_type = "normal"
        elif result == 1:
            attack_type = "Grayhole"
        elif result == 2:
            attack_type = "Blackhole"
        elif result == 3:
            attack_type = "tdma"
        else:
            attack_type = "Flooding"

    return render_template('prediction.html', attack_type=attack_type)

if __name__ == '__main__':
    app.run(debug=True)