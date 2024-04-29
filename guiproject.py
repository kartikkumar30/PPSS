import os
import sqlite3
import cv2
import face_recognition
import numpy as np
from datetime import datetime
from Pyfhel import Pyfhel
import shutil

# Function to create directories if they don't exist
def create_directories():
    for directory in ["D:/Output", "D:/Output/logs", "D:/Output/credentials"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Function to initialize database for credentials
def initialize_credentials_database():
    conn = sqlite3.connect('credentials/credentials.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT, permissions TEXT)''')
    
    # Check if admin account exists, if not, create one
    c.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    admin_count = c.fetchone()[0]
    if admin_count == 0:
        c.execute("INSERT INTO users (username, password, role, permissions) VALUES (?, ?, ?, ?)",
                  ('Admin', 'Admin@123', 'admin', 'create_user,live_face_detection,face_matching_live,face_matching_stored,homomorphic_encryption_decryption,pseudonymous_technique'))
        print("Admin account created successfully. Username: Admin, Password: Admin@123")
    conn.commit()
    conn.close()

# Function to initialize database for event logs
def initialize_logs_database():
    conn = sqlite3.connect('logs/logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, username TEXT, event TEXT)''')
    conn.commit()
    conn.close()


# Function to log user login events
def log_login_event(username):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect('logs/logs.db')
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, username, event) VALUES (?, ?, ?)", (timestamp, username[0], "Login"))
    conn.commit()
    conn.close()

# Function to log events to the database
def log_event(event, username):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect('logs/logs.db')
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, username, event) VALUES (?, ?, ?)", (timestamp, username[0], event))
    conn.commit()
    conn.close()


def authenticate_user():
    conn = sqlite3.connect('credentials/credentials.db')
    c = conn.cursor()

    while True:
        username = input("Enter username: ")
        password = input("Enter password: ")

        c.execute("SELECT username, role, permissions FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        #print("User tuple:", user)

        if user:
            log_login_event(username)
            return user
        else:
            print("Invalid username or password. Please try again.")


# Function to create a new user account
def create_user(username, password, role, permissions):
    conn = sqlite3.connect('credentials/credentials.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password, role, permissions) VALUES (?, ?, ?, ?)",
              (username, password, role, permissions))
    conn.commit()
    conn.close()
    print("User account created successfully.")

def create_user_account():
    username = input("Enter new username: ")
    password = input("Enter new password: ")
    role = input("Enter role (admin/viewer/writer): ")
    permissions = input("Enter permissions (comma-separated): ")
    create_user(username, password, role, permissions)

# Define a function to check if a user has permission for a specific action
def check_permission(user, action):
    if user and user[1] in ROLES and action in ROLES[user[1]]['permissions']:
        return True
    return False


# Function for live video surveillance with face detection
def live_face_detection(user):
    if check_permission(user, 'live_face_detection'):

        # Initialize video capture
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('D:/Output/live_face_detection.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, (frame_width,frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform face detection
            # Highlighting faces with different colors
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Write frame to video output
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and video writer
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        log_event("Live Face Detection", user)
    else:
        print("You don't have permission to perform live face detection.")


def face_matching_live(user):
    if check_permission(user, 'face_matching_live'):

        # Load the reference image for face matching
        reference_image_path = input("Enter the path to the reference image: ")
        reference_image = face_recognition.load_image_file(reference_image_path)
        reference_encoding = face_recognition.face_encodings(reference_image)[0]

        # Initialize video capture
        cap = cv2.VideoCapture(0)

        matched_faces = []  # List to store matched faces
        total_faces = 0  # Total number of faces detected
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find faces in the current frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Increment the total number of faces detected
            total_faces += len(face_encodings)

            # Match faces with the reference image
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare the face encoding with the reference face encoding
                match = face_recognition.compare_faces([reference_encoding], face_encoding)
                if match[0]:
                    matched_faces.append((face_encoding, face_location))
                else:
                    # Do something when no match is found
                    pass

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture
        cap.release()
        cv2.destroyAllWindows()
        
        # If no face is matched, handle it gracefully
        if len(matched_faces) == 0:
            print("No face matched.")
            return

        # Calculate accuracy percentage
        matched_faces_count = len(matched_faces)
        accuracy_percentage = (matched_faces_count / total_faces) * 100 if total_faces > 0 else 0
        
        # Find the most accurate matched face
        most_accurate_face = None
        max_distance = np.inf
        for face_encoding, face_location in matched_faces:
            distance = face_recognition.face_distance([reference_encoding], face_encoding)
            if distance < max_distance:
                max_distance = distance
                most_accurate_face = frame[face_location[0]:face_location[2], face_location[3]:face_location[1]]

        # Display the most accurate matched face if it exists
        if most_accurate_face is not None:
            cv2.imshow('Most Accurate Matched Face', most_accurate_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Ask user permission to store the most accurate matched face
            store_result = input("Do you want to store the most accurate matched face? (yes/no): ").lower()
            if store_result == 'yes':
                storage_location = input("Enter the storage location with a valid file extension (e.g., D:/matched_face.jpg): ")
                cv2.imwrite(storage_location, most_accurate_face)
                print("Most accurate matched face stored successfully.")
        else:
            print("No most accurate matched face found.")

        # Display the "Face matched!" message with accuracy percentage
        print(f"Face matched with {accuracy_percentage:.2f}% accuracy.")

        log_event("Live Face Matching", user)  # You can uncomment this line to log the event if needed

        return accuracy_percentage
    else:
        print("You don't have permission to perform face matching from image with live video surveillance.")



# Function for face matching from image with stored video
def face_matching_stored(user):
    if check_permission(user, 'face_matching_stored'):
        # Load the reference image for face matching
        reference_image_path = input("Enter the path to the reference image: ")
        reference_image = face_recognition.load_image_file(reference_image_path)
        reference_encoding = face_recognition.face_encodings(reference_image)[0]

        # Placeholder for accessing stored video
        stored_video_path = input("Enter the path to the stored video: ")
        stored_video_capture = cv2.VideoCapture(stored_video_path)  # Initialize video capture

        # Initialize variables for accuracy calculation
        total_frames = 0
        matches_found = 0
        most_accurate_face = None
        max_accuracy = 0

        # Placeholder for accessing stored video frames
        while True:
            ret, frame = stored_video_capture.read()  # Read frame from video capture
            if not ret:
                break

            # Increment total frames count
            total_frames += 1

            # Convert frame to RGB format for face recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each detected face in the current frame
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                # Compare the face encoding with the reference face encoding
                match = face_recognition.compare_faces([reference_encoding], face_encoding)
                if match[0]:
                    # If a match is found, increment matches found count
                    matches_found += 1
                    accuracy = face_recognition.face_distance([reference_encoding], face_encoding)
                    accuracy_percentage = (1 - accuracy[0]) * 100
                    print(f"Face matched with {accuracy_percentage:.2f}% accuracy.")
                    if accuracy_percentage > max_accuracy:
                        max_accuracy = accuracy_percentage
                        most_accurate_face = frame[top:bottom, left:right].copy()

                    # Example: Draw a rectangle around the matched face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    # Example: Draw a rectangle around the unmatched face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Display the resulting frame with face detections
            cv2.imshow('Stored Video Processing', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Calculate accuracy percentage
        accuracy_percentage = (matches_found / total_frames) * 100 if total_frames > 0 else 0
        print(f"Overall accuracy: {accuracy_percentage:.2f}%")

        # Prompt user to store the result
        store_result = input("Do you want to store the most accurate matched image? (yes/no): ")
        if store_result.lower() == "yes" and most_accurate_face is not None:
            storage_location = input("Enter the storage location: ")
            # Save the most accurate matched image from the video
            cv2.imwrite(storage_location, most_accurate_face)

        # Release video capture
        stored_video_capture.release()
        cv2.destroyAllWindows()
        log_event("Stored Video Processing", user)

    else:
        print("You don't have permission to perform face matching from image with stored video.")

    
# Function to perform encryption
def perform_encryption(user):
# Get video input from the user
    video_path = input("Enter the path to the video file: ")
    output_location = input("Enter the location to save the encrypted video: ")
    HE = Pyfhel()
    HE.contextGen(n=2**14, t_bits=20, scheme='bgv')
    HE.keyGen()
    context_path = input("Enter the path to save the context file: ")
    pubkey_path = input("Enter the path to save the public key file: ")
    seckey_path = input("Enter the path to save the secret key file: ")

    # Read the video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define codec and create VideoWriter object
    out = cv2.VideoWriter(os.path.join(output_location, "encrypted_video.avi"),
                cv2.VideoWriter_fourcc(*'DIVX'), 10, (frame_width, frame_height))

        # Encryption process for each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

            # Encrypt the frame
        encrypted_frame = []
        for row in frame:
            encrypted_row = [HE.encrypt(int(pixel)) for pixel in row.flatten()]

            # Write frame to output
        encrypted_frame = np.array(encrypted_frame).astype(np.uint8)
        out.write(encrypted_frame)

        # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Encrypted video saved successfully.")

        # Log event
    log_event("Homomorphic encryption", user)

def perform_decryption(user):
    # Get video input from the user
    video_path = input("Enter the path to the encrypted video file: ")
    output_location = input("Enter the location to save the decrypted video: ")

    # Initialize Pyfhel
    HE = Pyfhel()
    HE.restoreContext("context_path")  # Replace "context_path" with the actual path to the context file
    HE.restorepublicKey("pubkey_path")  # Replace "pubkey_path" with the actual path to the public key file
    HE.restoresecretKey("seckey_path")  # Replace "seckey_path" with the actual path to the secret key file

    # Read the encrypted video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(output_location, "decrypted_video.avi"),
                          cv2.VideoWriter_fourcc(*'DIVX'), 10, (frame_width, frame_height))

    # Decryption process for each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Decrypt the frame
        decrypted_frame = []  # Placeholder for decrypted frame
        for row in frame:
            decrypted_row = [HE.decrypt(pixel) for pixel in row.flatten()]
            decrypted_frame.append(decrypted_row)

        decrypted_frame = np.array(decrypted_frame).astype(np.uint8)
        out.write(decrypted_frame)

    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Decrypted video saved successfully.")


    # Log event
    log_event("Homomorphic decryption", user)



def homomorphic_encryption_decryption(user):
    if check_permission(user, 'homomorphic_encryption_decryption'):
        username, role, permissions = user# Unpack the username tuple
        #print("User info:", username)  # Add this line to check the value of username
        # Check user permission
        #user_role = ROLES.get(user)
        if role in ROLES and 'homomorphic_encryption_decryption' in ROLES[role]['permissions']:
            print("Homomorphic encryption/decryption...")

            # Get user choice: encryption or decryption
            choice = input("Choose an operation:\n1. Encryption\n2. Decryption\nEnter your choice: ")

            # Perform encryption or decryption based on user choice
            if choice == '1':
                perform_encryption(user)
            elif choice == '2':
                perform_decryption(user)
            else:
                print("Invalid choice.")

    else:
        print("You don't have permission to perform homomorphic encryption/decryption.")


# Function to perform encryption using pseudonymous technique
def perform_encryption_pseudonymous(user):
    # Get video input from the user
    video_path = input("Enter the path to the video file: ")
    output_location = input("Enter the location to save the encrypted video: ")

    # Placeholder for encryption logic using pseudonymous technique
    print("Encrypting using pseudonymous technique...")
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_location, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pseudonymize the frame (Example: blur the faces)
        blurred_frame = cv2.GaussianBlur(frame, (15, 15), 50)

        # Write the pseudonymized frame to the output video
        out.write(blurred_frame)

        # Display the pseudonymized frame (optional)
        cv2.imshow('Pseudonymized Video', blurred_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Encryption completed.")
    # Log event
    log_event("Encryption using pseudonymous technique", user)



# Function to perform decryption using pseudonymous technique
def perform_decryption_pseudonymous(user):
    # Get video input from the user
    video_path = input("Enter the path to the encrypted video file: ")
    output_location = input("Enter the location to save the decrypted video: ")

    # Placeholder for decryption logic using pseudonymous technique
    print("Decrypting using pseudonymous technique...")
    
    # Simulated decryption process
    # This is where you would implement your decryption logic
    # For demonstration, let's just copy the encrypted video file to the output location
    
    shutil.copy(video_path, output_location)
    
    print("Decryption completed.")
    
    # Log event
    log_event("Decryption using pseudonymous technique", user)


def pseudonymous_technique(user):
    print("Pseudonymous technique...")
    if check_permission(user, 'pseudonymous_technique'):
        """#username, role, permissions = user# Unpack the username tuple

        # Check user permission
        user_role = ROLES.get(user)
        if user_role and 'pseudonymous_technique' in user_role['permissions']:"""
        

        # Get user choice: encryption or decryption
        choice = input("Choose an operation:\n1. Encrypt\n2. Decrypt\nEnter your choice: ")

            # Perform encryption or decryption based on user choice
        if choice == '1':
            perform_encryption_pseudonymous(user)
        elif choice == '2':
            perform_decryption_pseudonymous(user)
        else:
            print("Invalid choice.")
            
        # Log event
        log_event("Pseudonymous technique", user)
    else:
        print("You don't have permission to perform the pseudonymous technique.")


# User roles and permissions
ROLES = {
    'admin': {'permissions': ['create_user', 'live_face_detection', 'face_matching_live', 'face_matching_stored', 'homomorphic_encryption_decryption', 'pseudonymous_technique']},
    'viewer': {'permissions': ['live_face_detection']},
    'writer': {'permissions': ['homomorphic_encryption_decryption', 'pseudonymous_technique']}
}


# Main function
def main():
    create_directories()
    initialize_credentials_database()
    initialize_logs_database()

    while True:
        print("\nMenu:")
        print("1. Live video surveillance with face detection")
        print("2. Face matching from image with live video surveillance")
        print("3. Face matching from image with stored video")
        print("4. Homomorphic encryption and decryption")
        print("5. Pseudonymous technique")
        print("6. Create a new user account (Admin only)")
        print("7. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            live_face_detection(user)
        elif choice == '2':
            face_matching_live(user)
        elif choice == '3':
            face_matching_stored(user)
        elif choice == '4':
            homomorphic_encryption_decryption(user)
        elif choice == '5':
            pseudonymous_technique(user)
        elif choice == '6':
            # Only allow admin to create new user accounts
            if user[2] == 'admin':
                create_user_option()
            else:
                print("You don't have permission to create a new user account.")
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

# Function to create a new user account (Admin only)
def create_user_option():
    username = input("Enter new username: ")
    password = input("Enter new password: ")
    role = input("Enter role (admin/user): ")
    permissions = input("Enter permissions (comma-separated): ")

    create_user(username, password, role, permissions)

if __name__ == "__main__":
    main()

