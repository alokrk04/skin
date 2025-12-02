from flask import Flask, render_template, request, url_for, redirect, session, flash
from werkzeug.utils import secure_filename
import json
import requests
import requests
from flask import request, jsonify

import openai
from flask import send_from_directory

import os
# Set environment variables before importing TensorFlow to reduce
# multi-threading / forking related issues on macOS (mutex errors).
# Limit OpenMP threads and reduce TF logging noise.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from reportlab.lib.utils import ImageReader  # Add this import at the top
from reportlab.lib.utils import ImageReader
from PIL import Image  # Add this import
from PIL import UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

app = Flask(__name__)

# Flask Configurations
app.config['UPLOAD_FOLDER'] = 'static/tests'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = 'your_secret_key'  # Required for Flask sessions
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Dependencies for model
dependencies = {'auc_roc': AUC}
verbose_name = {
    0: 'Actinic keratoses and intraepithelial carcinomae',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Pyogenic granulomas and hemorrhage',
    6: 'Melanoma',
}

# Load your model
model_path = os.path.join('model', 'skin.h5')
# Lazy-load the model to avoid loading it during module import time.
# This prevents issues with Flask's debug reloader and process forking
# which can trigger C++ mutex errors on some macOS/Python/TensorFlow setups.
model = None

def get_model():
    """Return the loaded Keras model, loading it on first use."""
    global model
    if model is None:
        model = load_model(model_path, custom_objects=dependencies)
    return model


# Define User and Test models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    tests = db.relationship('Test', backref='owner', lazy=True)


class Test(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(150), nullable=False)
    result = db.Column(db.String(150), nullable=False)
    patient_name = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    upload_date = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    pdf_filename = db.Column(db.String(150), nullable=True)  # New column for storing PDF filename


# Create tables
with app.app_context():
    db.create_all()


# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash("Login failed. Check username and/or password.", "danger")
    return render_template("login.html")


@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            hashed_password = generate_password_hash(password)

            new_user = User(username=username, password=hashed_password)

            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match. Please try again.')

    return render_template('register.html')


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/previous')
@login_required
def previous():
    """Display all previous tests for the current user"""
    user_tests = Test.query.filter_by(user_id=current_user.id) \
        .order_by(Test.upload_date.desc()) \
        .all()
    tests_data = []
    for test in user_tests:
        # Handle both string and datetime objects
        if isinstance(test.upload_date, str):
            try:
                # Parse string date if stored as string
                upload_date = datetime.strptime(test.upload_date, '%Y-%m-%d_%H-%M-%S')
                formatted_date = upload_date.strftime('%b %d, %Y %I:%M %p')
            except ValueError:
                formatted_date = test.upload_date  # Fallback to raw string
        else:
            # Already a datetime object
            formatted_date = test.upload_date.strftime('%b %d, %Y %I:%M %p')

        tests_data.append({
            'id': test.id,
            'image_filename': test.image_filename,
            'pdf_filename': test.pdf_filename,
            'result': test.result,
            'patient_name': test.patient_name,
            'upload_date': test.upload_date,  # Original value
            'formatted_date': formatted_date  # Formatted string
        })

    return render_template("previous_tests.html", tests=tests_data)


@app.route('/view_test/<int:test_id>')
@login_required
def view_test(test_id):
    """View details of a specific test"""
    test = Test.query.filter_by(id=test_id, user_id=current_user.id).first_or_404()

    return render_template("test_details.html",
                           test=test,
                           img_path=url_for('static', filename=f'tests/{test.image_filename}'),
                           pdf_url=url_for('static', filename=f'reports/{test.pdf_filename.split("/")[-1]}'))


@app.route('/download_report/<int:test_id>')
@login_required
def download_report(test_id):
    """Download PDF report for a specific test"""
    test = Test.query.filter_by(id=test_id, user_id=current_user.id).first_or_404()
    pdf_filename = test.pdf_filename.split('/')[-1]
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'reports'),
        pdf_filename,
        as_attachment=True
    )


@app.route('/view_report/<int:test_id>')
@login_required
def view_report(test_id):
    """View PDF report in browser"""
    test = Test.query.filter_by(id=test_id, user_id=current_user.id).first_or_404()
    pdf_filename = test.pdf_filename.split('/')[-1]
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'reports'),
        pdf_filename,
        as_attachment=False
    )

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(28, 28))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(1, 28, 28, 3)

    # Ensure model is loaded in the current process
    model_instance = get_model()
    predict_x = model_instance.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)

    predicted_class = classes_x[0]
    accuracy = float(np.max(predict_x)) * 100  # Convert to percentage

    return verbose_name[predicted_class], round(accuracy, 2)



from reportlab.lib.utils import ImageReader
from PIL import Image
import tempfile

from io import BytesIO
from reportlab.lib.utils import ImageReader
from PIL import Image


@app.route("/submit", methods=['POST'])
@login_required
def get_output():
    try:
        # 1. Validate file upload
        if 'my_image' not in request.files:
            raise ValueError("No file uploaded")

        img_file = request.files['my_image']
        if img_file.filename == '':
            raise ValueError("No file selected")

        if not allowed_file(img_file.filename):
            raise ValueError("Invalid file type")

        # 2. Process upload
        patient_name = request.form.get('patient_name', 'Unknown')
        upload_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = secure_filename(f"{patient_name}_{upload_date}_{img_file.filename}")
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            img_file.save(img_path)
        except IOError as e:
            raise RuntimeError(f"Failed to save image: {str(e)}")

        # 3. Make prediction
        try:
            predict_result, predict_accuracy = predict_label(img_path)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

        # 4. Generate PDF with error-proof image handling
        pdf_filename = f"report_{os.path.splitext(filename)[0]}.pdf"
        pdf_path = os.path.join('static/reports', pdf_filename)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        try:
            with Image.open(img_path) as pil_img:
                # Convert image to compatible format
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # Create in-memory image buffer
                img_buffer = BytesIO()
                pil_img.save(img_buffer, format='JPEG', quality=90)
                img_buffer.seek(0)

                # PDF Generation
                c = canvas.Canvas(pdf_path, pagesize=letter)
                width, height = letter

                # Add text content
                y_pos = height - 40
                c.setFont("Helvetica-Bold", 14)
                c.drawString(72, y_pos, "Diagnostic Report")
                y_pos -= 30

                # Patient info
                c.setFont("Helvetica", 12)
                c.drawString(72, y_pos, f"Patient: {patient_name}")
                y_pos -= 20
                c.drawString(72, y_pos, f"Date: {upload_date.replace('_', ' ')}")
                y_pos -= 30

                # Diagnosis
                c.setFont("Helvetica-Bold", 12)
                c.drawString(72, y_pos, "Diagnosis:")
                c.setFont("Helvetica", 12)
                c.drawString(72, y_pos - 20, predict_result)
                y_pos -= 40

                # Accuracy
                c.setFont("Helvetica-Bold", 12)
                c.drawString(72, y_pos, "Prediction Confidence:")
                c.setFont("Helvetica", 12)
                c.drawString(72, y_pos - 20, f"{predict_accuracy}%")
                y_pos -= 50

                # Embed image with fallback
                try:
                    img_width = min(400, pil_img.width)
                    img_height = int(pil_img.height * (img_width / pil_img.width))
                    x_pos = (width - img_width) / 2

                    c.drawImage(ImageReader(img_buffer),
                                x_pos, y_pos - img_height,
                                width=img_width,
                                height=img_height,
                                preserveAspectRatio=True)

                    # Image caption
                    c.drawString(x_pos, y_pos - img_height - 15, "Skin Lesion Image")
                except Exception as img_error:
                    app.logger.error(f"Image embedding failed: {img_error}")
                    c.drawString(72, y_pos - 20, "[Image unavailable in this report]")

                c.save()

        except UnidentifiedImageError:
            raise ValueError("Invalid image file - cannot process")
        except Exception as e:
            raise RuntimeError(f"PDF generation failed: {str(e)}")

        # 5. Database operations with transaction
        try:
            new_test = Test(
                image_filename=filename,
                result=predict_result,
                accuracy=predict_accuracy,
                patient_name=patient_name,
                user_id=current_user.id,
                upload_date=upload_date,
                pdf_filename=f"reports/{pdf_filename}"
            )
            db.session.add(new_test)
            db.session.commit()
        except SQLAlchemyError as e:
            db.session.rollback()
            raise RuntimeError(f"Database error: {str(e)}")

        # Return success
        return render_template("prediction.html",
                               prediction=predict_result,
                               patient_name=request.form['patient_name'],  # Make sure to pass this

                               upload_date=datetime.now(),
                               img_path=url_for('static', filename=f'tests/{filename}'),
                               pdf_url=url_for('static', filename=f'reports/{pdf_filename}'),
                               test_id=new_test.id)

    except ValueError as ve:
        flash(str(ve), 'error')
        return redirect(url_for('index'))
    except RuntimeError as re:
        flash(f"Processing error: {str(re)}", 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash("An unexpected error occurred", 'error')
        app.logger.exception("Unexpected error in submit route")
        return redirect(url_for('index'))


# Initialize Gemini API





@app.route("/download_pdf/<test_id>")
@login_required
def download_pdf(test_id):
    test = Test.query.get(test_id)
    if not test or test.user_id != current_user.id:
        return redirect(url_for('previous'))

    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)

    # Generate PDF filename with unique timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"report_{test.id}_{timestamp}.pdf"
    pdf_path = os.path.join('static', pdf_filename)

    try:
        # Create PDF
        c = canvas.Canvas(pdf_path, pagesize=letter)

        # PDF Content
        y_position = 750
        c.drawString(100, y_position, f"Patient Name: {test.patient_name}")
        y_position -= 20
        c.drawString(100, y_position, f"Test Result: {test.result}")
        y_position -= 20
        c.drawString(100, y_position, f"Image Filename: {test.image_filename}")
        y_position -= 20
        c.drawString(100, y_position, f"Upload Date: {test.upload_date}")

        # Diagnosis information
        diagnosis = {
            'Actinic keratoses and intraepithelial carcinomae': "Precancerous condition",
            'Basal cell carcinoma': "Type of skin cancer",
            'Benign keratosis-like lesions': "Non-cancerous lesions",
            'Dermatofibroma': "Benign fibrous tumor",
            'Melanocytic nevi': "Common moles",
            'Pyogenic granulomas and hemorrhage': "Benign vascular lesion",
            'Melanoma': "Severe skin cancer"
        }
        y_position -= 30
        c.drawString(100, y_position, "Diagnosis:")
        y_position -= 20
        c.drawString(120, y_position, diagnosis.get(test.result, "No specific diagnosis available"))

        c.save()

        # Update the test record with PDF filename
        test.pdf_filename = pdf_filename
        db.session.commit()

        # Verify PDF was created
        if not os.path.exists(pdf_path):
            flash("Failed to generate PDF", "error")
            return redirect(url_for('previous'))

        return redirect(url_for('static', filename=pdf_filename))

    except Exception as e:
        db.session.rollback()
        flash(f"Error generating PDF: {str(e)}", "error")
        return redirect(url_for('previous'))


# Replace with your actual API Key
GEMINI_API_KEY = "AIzaSyDhwCoV5szND-Fwlyqz_PDygsdgS3yCYHk"
# Using a known valid model endpoint
MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

def call_gemini_api(prompt: str) -> str:
    """
    Calls the Gemini API to generate content based on the provided prompt.

    Args:
        prompt: The text prompt to send to the Gemini API.

    Returns:
        The text response from the API or an error message.
    """
    if not GEMINI_API_KEY:
        return {"ok": False, "message": 'GEMINI_API_KEY not set in environment (GEMINI_API_KEY)'}

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        # Optional: Add generationConfig here if needed
        # "generationConfig": {
        #     "temperature": 0.7,
        #     "maxOutputTokens": 1024
        # }
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=payload,
            timeout=60 # Increased timeout for potentially longer generations
        )

        # Check for HTTP errors first
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        # Process successful response
        data = response.json()

        # Check the expected structure of the successful response
        if 'candidates' in data and data['candidates']:
            candidate = data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content'] and candidate['content']['parts']:
                 return candidate['content']['parts'][0]['text']
            else:
                # Handle cases where the candidate structure is unexpected (e.g., safety blocked)
                return f"Error: Received response, but content format is unexpected. Response: {json.dumps(data)}"
        else:
             # Handle cases where 'candidates' might be missing (e.g., prompt feedback)
             if 'promptFeedback' in data:
                 return f"Error: Prompt blocked or issue detected. Feedback: {json.dumps(data['promptFeedback'])}"
             return f"Error: Unexpected response format from Gemini API. Response: {json.dumps(data)}"


    except requests.exceptions.HTTPError as http_err:
        # Handle specific HTTP errors (like 400, 401, 429, 500, etc.)
        error_details = "Unknown error details"
        try:
            # Try to get more detailed error info from the response body
             error_info = response.json().get('error', {})
             error_details = error_info.get('message', json.dumps(error_info))
        except json.JSONDecodeError:
            error_details = response.text # Use raw text if JSON parsing fails
        return f"HTTP Error {response.status_code}: {error_details}"

    except requests.exceptions.RequestException as req_err:
        # Handle network-related errors (DNS failure, refused connection, timeout, etc.)
        return f"Network Error: {str(req_err)}"
    except Exception as e:
        # Catch any other unexpected errors during the process
        return f"Unexpected Error: {str(e)}"



@app.route("/api/gemini", methods=["POST"])
@login_required
def api_gemini():
    try:
        data = request.get_json()

        patient_name = data.get("patient_name", "")
        test_result = data.get("test_result", "")
        user_message = data.get("user_message", "")

        if not user_message:
            return jsonify({"reply": "Please type a message."}), 400

        # Build prompt for Gemini model
        prompt = f"""
        Patient Name: {patient_name}
        Diagnosis: {test_result}

        User Query: {user_message}

        Provide a safe, medically accurate explanation in simple language.
        """

        # Call your existing Gemini function
        gemini_reply = call_gemini_api(prompt)

        return jsonify({"reply": gemini_reply})

    except Exception as e:
        return jsonify({"reply": f"Error communicating with AI: {str(e)}"}), 500




if __name__ == '__main__':
    # Disable the reloader to avoid forking the process after TensorFlow
    # has been imported/initialized which can lead to C++ mutex errors
    # on macOS. In development you can set `use_reloader=True` if you
    # restructure model loading to be safe.
    app.run(debug=True, use_reloader=False, port=5001)
