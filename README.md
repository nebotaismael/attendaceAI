# README.md

# Attendance Verification System

This project is an attendance verification system that utilizes face recognition and anti-spoofing techniques to ensure secure and accurate attendance tracking. The system is built using Flask and integrates with Firebase for user management and data storage.

## Project Structure

```
attendance-verification-system
├── app.py                          # Main application logic
├── .env                             # Environment variables
├── models                           # Directory for model files
│   └── antispoofing_model.h5       # Pre-trained anti-spoofing model
│   └── README.md                    # Documentation for models
├── firebase                         # Firebase service account credentials
│   └── serviceAccountKey.json       # Firebase credentials
├── temp                             # Temporary files directory
│   └── .gitkeep                     # Keeps the temp directory in Git
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore file
```

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/attendance-verification-system.git
   cd attendance-verification-system
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   - Create a `.env` file in the root directory and add the following variables:
     ```
     JWT_SECRET=your_jwt_secret_key
     FIREBASE_CREDENTIALS=path/to/serviceAccountKey.json
     FIREBASE_STORAGE_BUCKET=your_firebase_storage_bucket
     ANTISPOOFING_MODEL_PATH=models/antispoofing_model.h5
     ENCRYPTION_KEY=your_encryption_key
     ```

5. **Download or Train the Anti-Spoofing Model**:
   - If you do not have the `antispoofing_model.h5`, you can download a pre-trained model from a repository or website that provides anti-spoofing models. Alternatively, you can train your own model using a suitable dataset and save it as `antispoofing_model.h5` in the `models` directory.

6. **Run the Application**:
   ```
   python app.py
   ```

## Usage

- The application provides API endpoints for user authentication and attendance verification. You can interact with these endpoints using tools like Postman or cURL.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.