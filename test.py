from datetime import datetime, timedelta
import os
from flask import Flask, jsonify, request
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pymongo import MongoClient
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import base64
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from flask_jwt_extended import create_access_token, current_user, get_jwt_identity, jwt_required, JWTManager
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Tambahkan ini untuk mengaktifkan CORS

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/flaskquran"
mongo = PyMongo(app)
client = MongoClient('mongodb://localhost:27017/')
db = client['Quran']
collection = db['audio']

app.config["JWT_SECRET_KEY"] = "super-secret"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)
jwt = JWTManager(app)

class User:
    def __init__(self, email, name, password, age, gender):
        self.email = email
        self.name = name
        self.password = password
        self.age = age
        self.gender = gender

    def to_dict(self):
        return {
            "email": self.email,
            "name": self.name,
            "password": self.password,
            "age": self.age,
            "gender": self.gender
        }

@jwt.user_lookup_loader
def user_lookup_callback(_jwt_header, jwt_data):
    identity = jwt_data["sub"]
    user = mongo.db.users.find_one({"_id": ObjectId(identity)})
    return user

@app.route("/user", methods=['GET', 'POST', 'PUT', 'DELETE'])
@jwt_required()
def user():
    if request.method == 'POST':
        dataDict = request.get_json()
        email = dataDict["email"]
        name = dataDict["name"]
        age = dataDict["age"]
        gender = dataDict["gender"]
        user = User(
            email=email,
            name=name,
            password="",  # Password tidak digunakan pada endpoint ini
            age=age,
            gender=gender
        )
        mongo.db.users.insert_one(user.to_dict())

        return {
            "message": "Successfully created user",
            "data": f"email: {email}, name: {name}, age: {age}, gender: {gender}"
        }, 200

    elif request.method == 'PUT':
        dataDict = request.get_json()
        user_id = dataDict["id"]
        email = dataDict.get("email")
        name = dataDict.get("name")
        age = dataDict.get("age")
        gender = dataDict.get("gender")

        if not user_id:
            return {
                "message": "ID required"
            }, 400

        update_fields = {}
        if email:
            update_fields["email"] = email
        if name:
            update_fields["name"] = name
        if age:
            update_fields["age"] = age
        if gender:
            update_fields["gender"] = gender

        mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update_fields})
        return {
            "message": "Successfully updated user"
        }, 200

    elif request.method == 'DELETE':
        dataDict = request.get_json()
        user_id = dataDict["id"]

        if not user_id:
            return {
                "message": "ID required"
            }, 400

        mongo.db.users.delete_one({"_id": ObjectId(user_id)})
        return {
            "message": "Successfully deleted user"
        }, 200

    else:
        users = mongo.db.users.find()
        users_list = []
        for user in users:
            user["_id"] = str(user["_id"])
            users_list.append(user)
        return jsonify(users_list), 200


@app.post('/register')
def signup():
    dataDict = request.get_json()
    name = dataDict["name"]
    email = dataDict["email"]
    password = dataDict["password"]
    re_password = dataDict["re_password"]
    age = dataDict["age"]
    gender = dataDict["gender"]

    if password != re_password:
        return {
            "message": "Passwords do not match!"
        }, 400

    if not email:
        return {
            "message": "Email is required"
        }, 400

    hashed_password = PasswordHasher().hash(password)
    new_user = User(
        email=email,
        name=name,
        password=hashed_password,
        age=age,
        gender=gender
    )
    mongo.db.users.insert_one(new_user.to_dict())

    return {
        "message": "Successfully registered user"
    }, 201


@app.post("/login")
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return {
            "message": "Email and password are required"
        }, 400

    user = mongo.db.users.find_one({"email": email})

    try:
        if not user or not PasswordHasher().verify(user["password"], password):
            return {
                "message": "Invalid email or password"
            }, 400
    except VerifyMismatchError:
        return {
            "message": "Invalid email or password"
        }, 400

    access_token = create_access_token(identity=str(user["_id"]))

    return {
        "token_access": access_token
    }, 200


@app.get("/myprofile")
@jwt_required()
def profile():
    user = current_user

    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "name": user["name"],
        "age": user["age"],
        "gender": user["gender"]
    }


@app.get("/who")
@jwt_required()
def protected():
    return jsonify(
        id=str(current_user["_id"]),
        email=current_user["email"],
        name=current_user["name"],
        age=current_user["age"],
        gender=current_user["gender"]
    )


@app.get("/whoami")
@jwt_required()
def whoami():
    user_identity = get_jwt_identity()
    user = mongo.db.users.find_one({"_id": ObjectId(user_identity)})
    
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "name": user["name"],
        "age": user["age"],
        "gender": user["gender"]
    }, 200


def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with tf.device('/cpu:0'):
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            features.append(mfccs)
            # Plot and save the MFCC
            plt.figure(figsize=(10, 4))
            plt.imshow(mfccs.reshape(-1, 1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()
            plt.savefig('running/MFCC_{}.png'.format(os.path.basename(file_path)))
            plt.close()
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T,axis=0)
            features.append(chroma)
            # Plot and save the Chroma
            plt.figure(figsize=(10, 4))
            plt.imshow(chroma.reshape(-1, 1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('Chroma')
            plt.tight_layout()
            plt.savefig('running/Chroma_{}.png'.format(os.path.basename(file_path)))
            plt.close()
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T,axis=0)
            features.append(mel)
            # Plot and save the Mel spectrogram
            plt.figure(figsize=(10, 4))
            plt.imshow(mel.reshape(-1, 1), cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig('running/Mel_{}.png'.format(os.path.basename(file_path)))
            plt.close()
    return np.concatenate(features)


model_path = "Model/model.h5"
model = tf.keras.models.load_model(model_path)

def predict_audio(file_path):
    feature = extract_features(file_path)
    prediction = model.predict(np.expand_dims(feature, axis=0))
    predicted_class = np.argmax(prediction)
    return predicted_class

class_labels = {0: "Idgham_Bighunnah", 1: "Ikhfa_Haqiqi", 2: "Izhar_Halqi"}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        predicted_class = predict_audio(file_path)
        result = class_labels[predicted_class]

        # Menyimpan hasil prediksi ke MongoDB
        log_entry = {
            "filename": file.filename,
            "predicted_class": result,
            "timestamp": datetime.now()
        }
        try:
            collection.insert_one(log_entry)
            print(f"Prediction for {file.filename} saved to MongoDB")
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")

        return jsonify({"predicted_class": result})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('runs'):
        os.makedirs('runs')
    app.run(host='0.0.0.0', port=5000)
