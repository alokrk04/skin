# Skin Lesion Classification System

A Flask-based web application for automated skin lesion classification using deep learning. This system uses TensorFlow/Keras neural networks to classify skin lesions into 7 different categories with high accuracy.

## Overview

This application provides an intelligent system for classifying skin lesions into the following categories:

1. **Actinic keratoses and intraepithelial carcinomae (AKIEC)**
2. **Basal cell carcinoma (BCC)**
3. **Benign keratosis-like lesions (BKL)**
4. **Dermatofibroma (DF)**
5. **Melanocytic nevi (NV)**
6. **Pyogenic granulomas and hemorrhage (VASC)**
7. **Melanoma (MEL)**

## Features

- 🔐 **User Authentication**: Secure login and registration system
- 📸 **Image Upload & Analysis**: Upload skin lesion images for classification
- 🤖 **AI-Powered Classification**: Pre-trained deep learning model for accurate predictions
- 📊 **Results Dashboard**: View classification results with confidence scores
- 📝 **Test History**: Track and review previous test results
- 📄 **PDF Reports**: Generate downloadable PDF reports for test results
- 💾 **Database Integration**: SQLite database for user and test data persistence

## Project Structure

```
skin/
├── app.py                          # Main Flask application
├── best_model.h5                   # Pre-trained model weights
├── requirements.txt                # Python dependencies
├── DATASET/                        # Dataset files
│   ├── HAM10000_metadata.csv      # Metadata for HAM10000 dataset
│   └── hmnist_28_28_RGB.csv       # HMNIST dataset
├── model/                          # Model files
│   ├── model.py                    # Model architecture definition
│   ├── skin.h5                     # Trained model weights
│   ├── skin1.h5                    # Alternative model
│   └── skin2.h5                    # Alternative model
├── static/                         # Static files
│   ├── assets/                     # CSS, JavaScript, vendor libraries
│   │   ├── css/                    # Stylesheets
│   │   ├── js/                     # JavaScript files
│   │   ├── scss/                   # SCSS files
│   │   └── vendor/                 # Third-party libraries (Bootstrap, FontAwesome, etc.)
│   ├── reports/                    # Generated PDF reports
│   └── tests/                      # Uploaded test images
├── templates/                      # HTML templates
│   ├── first.html                  # Landing page
│   ├── index.html                  # Home page
│   ├── login.html                  # Login page
│   ├── register.html               # Registration page
│   ├── prediction.html             # Prediction interface
│   ├── results.html                # Results display
│   └── previous_tests.html         # Test history
├── upload/                         # Upload directory by skin type
│   ├── akiec/                      # AKIEC samples
│   ├── bcc/                        # BCC samples
│   ├── bkl/                        # BKL samples
│   ├── df/                         # DF samples
│   ├── nv/                         # NV samples
│   ├── mel/                        # MEL samples
│   └── vasc/                       # VASC samples
├── instance/                       # Flask instance folder (SQLite database)
└── __pycache__/                    # Python cache

```

## Requirements

- Python 3.10+
- Flask 3.1.0
- TensorFlow 2.20.0
- Keras 3.12.0
- SQLAlchemy 2.0.39
- Flask-SQLAlchemy 3.1.1
- Flask-Login 0.6.3
- ReportLab 4.4.5
- Pillow 11.1.0
- NumPy 2.1.3
- OpenAI API

## Installation

### 1. Clone or download the project

```bash
cd skin
```

### 2. Create a virtual environment (Optional but recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install flask==3.1.0
pip install tensorflow==2.20.0
pip install flask-sqlalchemy==3.1.1
pip install flask-login==0.6.3
pip install reportlab==4.4.5
pip install pillow==11.1.0
pip install openai
```

### 4. Configure the application

- Update `SECRET_KEY` in `app.py` with a secure secret key
- Set up OpenAI API key if using AI features
- Ensure the `upload/` directory exists with proper subdirectories

## Usage

### Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000` by default.

### Access the Application

1. **Landing Page**: Visit `http://localhost:5000/first`
2. **Register**: Create a new account at `/register`
3. **Login**: Login with your credentials at `/login`
4. **Upload Image**: Go to the prediction page and upload a skin lesion image
5. **View Results**: See classification results with confidence scores
6. **Download Report**: Generate and download PDF reports of test results
7. **View History**: Check previous test results in the test history page

## API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` or `/first` | GET | Landing page |
| `/login` | GET, POST | User login |
| `/register` | GET, POST | User registration |
| `/index` | GET | Home page (requires login) |
| `/prediction` | GET, POST | Image upload and prediction |
| `/results` | POST | Display prediction results |
| `/previous_tests` | GET | View test history |
| `/predict` | POST | API endpoint for predictions (JSON) |
| `/logout` | GET | User logout |

## Database Models

### User Model
- `id`: Primary key
- `username`: Unique username
- `password`: Hashed password
- `tests`: Relationship to Test records

### Test Model
- `id`: Primary key
- `image_filename`: Uploaded image filename
- `result`: Classification result
- `patient_name`: Patient name
- `user_id`: Foreign key to User
- `upload_date`: Test date
- `accuracy`: Prediction confidence score
- `pdf_filename`: Generated PDF report filename

## File Upload Configuration

- **Max File Size**: 16 MB
- **Allowed Extensions**: `.png`, `.jpg`, `.jpeg`
- **Upload Path**: `static/tests/`

## Model Information

The application uses a pre-trained deep learning model (`skin.h5`) that has been trained on skin lesion datasets:

- **Model Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input**: 224x224 RGB images
- **Output**: 7-class classification with confidence scores
- **Custom Metrics**: AUC-ROC for model evaluation

## Dependencies Explanation

| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| TensorFlow/Keras | Deep learning framework |
| SQLAlchemy | ORM for database operations |
| Flask-SQLAlchemy | Flask integration with SQLAlchemy |
| Flask-Login | User authentication management |
| ReportLab | PDF report generation |
| Pillow | Image processing |
| NumPy | Numerical computing |
| OpenAI | AI API integration |
| Werkzeug | WSGI utilities |

## Security Considerations

⚠️ **Important**: Before deploying to production:

1. Change the `SECRET_KEY` to a strong, random value
2. Set `SQLALCHEMY_DATABASE_URI` to use a production database
3. Enable HTTPS/SSL
4. Implement rate limiting
5. Add CSRF protection
6. Validate and sanitize all user inputs
7. Store sensitive credentials in environment variables
8. Use a production WSGI server (Gunicorn, etc.)

## Troubleshooting

### Module Not Found Errors

If you encounter `ModuleNotFoundError`, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
pip install --user openai tensorflow flask-sqlalchemy flask-login reportlab
```

### Database Issues

To reset the database:

1. Delete `instance/users.db` if it exists
2. Restart the application

### Model Loading Issues

Ensure `model/skin.h5` exists and is accessible. The model file should be in the same directory structure.

## Performance Notes

- First prediction may take a few seconds as the model is loaded into memory
- Image preprocessing includes resizing to 224x224 pixels
- Prediction accuracy depends on image quality and lighting

## Future Enhancements

- [ ] Multi-image batch processing
- [ ] Real-time model predictions via WebSocket
- [ ] Advanced analytics dashboard
- [ ] Model interpretation (Grad-CAM visualization)
- [ ] Mobile application
- [ ] Integration with medical imaging standards (DICOM)
- [ ] Deployment with Docker
- [ ] CI/CD pipeline

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, questions, or suggestions, please contact the development team or create an issue in the repository.

## Dataset Attribution

This project uses skin lesion datasets including:
- **HAM10000**: Large collection of multi-source dermatoscopic images
- **HMNIST**: Harvard Medical School skin lesion dataset

## Disclaimer

⚠️ **Medical Disclaimer**: This application is intended for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult with a qualified dermatologist for medical advice.

---

**Last Updated**: November 26, 2025
**Version**: 1.0.0
