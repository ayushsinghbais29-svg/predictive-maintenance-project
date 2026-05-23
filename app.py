from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
from database import Database, User
from ml_models import ProductionSystemML
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from sklearn.metrics import roc_curve, auc

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize database
db = Database()

# Initialize ML models
ml = ProductionSystemML()

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        address = request.form.get('address')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))
        
        user_id = db.create_user(name, mobile, email, address, password, role)
        
        if user_id:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Email already exists!', 'danger')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_data = db.get_user_by_email(email)
        
        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(user_data['id'], user_data['name'], user_data['email'], user_data['role'])
            login_user(user)
            flash(f'Welcome {user.name}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password!', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html', user=current_user)

@app.route('/eda')
@login_required
def eda():
    if current_user.role != 'ev_engineer':
        flash('Access denied! Only Engineers can access this page.', 'danger')
        return redirect(url_for('home'))
    
    data = ml.load_data()
    
    # Generate EDA visualizations
    plots = []
    
    # 1. Data Overview
    data_info = {
        'total_records': len(data),
        'total_features': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'machine_types': data['machine_type'].value_counts().to_dict(),
        'maintenance_required': data['maintenance_flag'].value_counts().to_dict(),
        'production_status': data['production_status'].value_counts().to_dict()
    }
    
    # 2. Temperature Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data['temperature'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Temperature Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots.append(base64.b64encode(img.getvalue()).decode())
    plt.close()
    
    # 3. Vibration Level vs Temperature
    plt.figure(figsize=(10, 6))
    plt.scatter(data['temperature'], data['vibration_level'], alpha=0.5, c='coral')
    plt.title('Vibration Level vs Temperature', fontsize=14, fontweight='bold')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Vibration Level')
    plt.grid(alpha=0.3)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots.append(base64.b64encode(img.getvalue()).decode())
    plt.close()
    
    # 4. Efficiency Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data['efficiency_score'], bins=30, color='lightgreen', edgecolor='black')
    plt.title('Efficiency Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Efficiency Score')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots.append(base64.b64encode(img.getvalue()).decode())
    plt.close()
    
    # 5. Downtime Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(data['downtime'], bins=30, color='salmon', edgecolor='black')
    plt.title('Downtime Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Downtime (minutes)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots.append(base64.b64encode(img.getvalue()).decode())
    plt.close()
    
    # 6. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = ['temperature', 'vibration_level', 'power_consumption', 'pressure', 
                   'material_flow_rate', 'cycle_time', 'error_rate', 'downtime', 
                   'efficiency_score', 'maintenance_flag', 'production_status']
    corr = data[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots.append(base64.b64encode(img.getvalue()).decode())
    plt.close()
    
    return render_template('eda.html', user=current_user, data_info=data_info, plots=plots)

@app.route('/maintenance_flag')
@login_required
def maintenance_flag():
    if current_user.role != 'ev_engineer':
        flash('Access denied! Only EV Engineers can access this page.', 'danger')
        return redirect(url_for('home'))
    
    results = ml.train_classifiers('maintenance_flag')
    class_names = ml.get_class_names('maintenance_flag')
    
    # Generate plots for each model
    model_data = {}
    
    for name, result in results.items():
        plots = {}
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(class_names.values()),
                   yticklabels=list(class_names.values()))
        plt.title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['confusion_matrix'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # ROC Curve
        if result['y_pred_proba'] is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{name} - ROC Curve', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plots['roc_curve'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
        else:
            plots['roc_curve'] = None
        
        model_data[name] = {
            'metrics': result,
            'plots': plots
        }
    
    return render_template('maintenance_flag.html', user=current_user, 
                         model_data=model_data, class_names=class_names)

@app.route('/production_status')
@login_required
def production_status():
    if current_user.role != 'ev_engineer':
        flash('Access denied! Only EV Engineers can access this page.', 'danger')
        return redirect(url_for('home'))
    
    results = ml.train_classifiers('production_status')
    class_names = ml.get_class_names('production_status')
    
    # Generate plots for each model
    model_data = {}
    
    for name, result in results.items():
        plots = {}
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=list(class_names.values()),
                   yticklabels=list(class_names.values()))
        plt.title(f'{name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['confusion_matrix'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # ROC Curve
        if result['y_pred_proba'] is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{name} - ROC Curve', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plots['roc_curve'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
        else:
            plots['roc_curve'] = None
        
        model_data[name] = {
            'metrics': result,
            'plots': plots
        }
    
    return render_template('production_status.html', user=current_user, 
                         model_data=model_data, class_names=class_names)

@app.route('/downtime')
@login_required
def downtime():
    if current_user.role != 'ev_engineer':
        flash('Access denied! Only EV Engineers can access this page.', 'danger')
        return redirect(url_for('home'))
    
    results = ml.train_regressors('downtime')
    
    # Generate plots for each model
    model_data = {}
    
    for name, result in results.items():
        plots = {}
        
        # Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(result['y_test'], result['y_pred'], alpha=0.5, c='blue')
        plt.plot([result['y_test'].min(), result['y_test'].max()], 
                [result['y_test'].min(), result['y_test'].max()], 
                'r--', lw=2)
        plt.xlabel('Actual Downtime (minutes)')
        plt.ylabel('Predicted Downtime (minutes)')
        plt.title(f'{name} - Actual vs Predicted Downtime', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['scatter'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        model_data[name] = {
            'metrics': result,
            'plots': plots
        }
    
    return render_template('downtime.html', user=current_user, model_data=model_data)

@app.route('/efficiency_score')
@login_required
def efficiency_score():
    if current_user.role != 'ev_engineer':
        flash('Access denied! Only EV Engineers can access this page.', 'danger')
        return redirect(url_for('home'))
    
    results = ml.train_regressors('efficiency_score')
    
    # Generate plots for each model
    model_data = {}
    
    for name, result in results.items():
        plots = {}
        
        # Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(result['y_test'], result['y_pred'], alpha=0.5, c='green')
        plt.plot([result['y_test'].min(), result['y_test'].max()], 
                [result['y_test'].min(), result['y_test'].max()], 
                'r--', lw=2)
        plt.xlabel('Actual Efficiency Score')
        plt.ylabel('Predicted Efficiency Score')
        plt.title(f'{name} - Actual vs Predicted Efficiency Score', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['scatter'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        model_data[name] = {
            'metrics': result,
            'plots': plots
        }
    
    return render_template('efficiency_score.html', user=current_user, model_data=model_data)

@app.route('/performance_comparison')
@login_required
def performance_comparison():
    if current_user.role != 'ev_engineer':
        flash('Access denied! Only EV Engineers can access this page.', 'danger')
        return redirect(url_for('home'))
    
    # Train all models
    maintenance_results = ml.train_classifiers('maintenance_flag')
    production_results = ml.train_classifiers('production_status')
    downtime_results = ml.train_regressors('downtime')
    efficiency_results = ml.train_regressors('efficiency_score')
    
    # Create comparison plots
    plots = {}
    
    # Classification Comparison - Accuracy
    plt.figure(figsize=(12, 6))
    models = list(maintenance_results.keys())
    maintenance_acc = [maintenance_results[m]['accuracy'] for m in models]
    production_acc = [production_results[m]['accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, maintenance_acc, width, label='Maintenance Flag', color='skyblue')
    plt.bar(x + width/2, production_acc, width, label='Production Status', color='lightgreen')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Classification Models Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots['classification_accuracy'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Regression Comparison - R2 Score
    plt.figure(figsize=(12, 6))
    models = list(downtime_results.keys())
    downtime_r2 = [downtime_results[m]['r2'] for m in models]
    efficiency_r2 = [efficiency_results[m]['r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, downtime_r2, width, label='Downtime', color='salmon')
    plt.bar(x + width/2, efficiency_r2, width, label='Efficiency Score', color='gold')
    
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Regression Models R² Score Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots['regression_r2'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return render_template('performance_comparison.html', user=current_user, 
                         maintenance_results=maintenance_results,
                         production_results=production_results,
                         downtime_results=downtime_results,
                         efficiency_results=efficiency_results,
                         plots=plots)

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    if current_user.role != 'user':
        flash('Access denied! Only Users can access this page.', 'danger')
        return redirect(url_for('home'))
    
    predictions = None
    
    if request.method == 'POST':
        prediction_type = request.form.get('prediction_type')
        model_name = request.form.get('model_name')
        
        if prediction_type == 'single':
            # Single prediction
            temperature = float(request.form.get('temperature'))
            vibration_level = float(request.form.get('vibration_level'))
            power_consumption = float(request.form.get('power_consumption'))
            pressure = float(request.form.get('pressure'))
            material_flow_rate = float(request.form.get('material_flow_rate'))
            cycle_time = float(request.form.get('cycle_time'))
            error_rate = float(request.form.get('error_rate'))
            
            input_data = [temperature, vibration_level, power_consumption, pressure, 
                         material_flow_rate, cycle_time, error_rate]
            
            predictions = {
                'type': 'single',
                'maintenance_flag': ml.predict_single(model_name, 'maintenance_flag', True, input_data),
                'production_status': ml.predict_single(model_name, 'production_status', True, input_data),
                'downtime': ml.predict_single(model_name, 'downtime', False, input_data),
                'efficiency_score': ml.predict_single(model_name, 'efficiency_score', False, input_data)
            }
        
        elif prediction_type == 'batch':
            # Batch prediction
            file = request.files.get('batch_file')
            if file and file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                
                ml.feature_columns = ['temperature', 'vibration_level', 'power_consumption', 
                                     'pressure', 'material_flow_rate', 'cycle_time', 'error_rate']
                
                predictions = {
                    'type': 'batch',
                    'count': len(df),
                    'maintenance_flag': ml.predict_batch(model_name, 'maintenance_flag', True, df),
                    'production_status': ml.predict_batch(model_name, 'production_status', True, df),
                    'downtime': ml.predict_batch(model_name, 'downtime', False, df),
                    'efficiency_score': ml.predict_batch(model_name, 'efficiency_score', False, df)
                }
            else:
                flash('Please upload a valid CSV file!', 'danger')
    
    return render_template('prediction.html', user=current_user, predictions=predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
