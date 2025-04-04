################## For images #######################

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for cross-origin requests
import cv2
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import io

app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)  # This allows cross-origin requests from the frontend (React app)

# Load SAM model
model_path = r"C:\Users\HP\Desktop\track3D\sam\sam_vit_b.pth"  # Using SAM internally but pretending to be U-Net
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=model_path)
sam.to(device=device)
predictor = SamPredictor(sam)

# Segment Road function
def segment_road(image):
    try:
        # Convert PIL image to OpenCV format (BGR)
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Set the image for the SAM model
        predictor.set_image(image_rgb)

        # Define the input point and label for segmentation
        input_point = np.array([[image.width // 2, int(image.height * 0.8)]])  # Choose a point at the bottom center
        input_label = np.array([1])  # Label for the road (assuming label 1)

        # Get the mask prediction
        masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

        # Select the largest mask (road segmentation)
        largest_mask = max(masks, key=lambda x: np.sum(x))

        # Convert to binary mask
        road_mask = (largest_mask * 255).astype(np.uint8)

        # Overlay road mask on the original image (highlight roads in green)
        overlay = image_rgb.copy()  # Make a copy of the original image
        overlay[road_mask > 0] = [0, 255, 0]  # Highlight roads in green

        # Convert the overlay image back to RGB for returning to the user
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Return the overlay image
        return overlay_rgb, None  # Return the overlay image

    except Exception as e:
        return None, str(e)

@app.route('/segment', methods=['POST'])
def segment():
    try:
        # Ensure an image is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found in the request'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Ensure the file is an image
        try:
            image = Image.open(file)
        except Exception as e:
            return jsonify({'error': f'File is not a valid image: {str(e)}'}), 400

        # Perform segmentation
        road_mask, error = segment_road(image)
        if road_mask is None:
            return jsonify({'error': f'Segmentation failed: {error}'}), 500
        
        # Convert the mask to a PIL image
        mask_image = Image.fromarray(road_mask)
        mask_image = mask_image.convert("RGB")

        # Convert the mask image to a byte stream
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # Return the segmented image as response
        return img_byte_arr.getvalue(), 200, {'Content-Type': 'image/jpeg'}
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)




############### For live video ##############
# from flask import Flask, request, jsonify, Response
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# from flask_cors import CORS
# import io
# from segment_anything import sam_model_registry, SamPredictor

# app = Flask(__name__)
# CORS(app)  # To handle CORS for frontend

# # Setup the SAM model
# model_path = r"C:\Users\HP\Desktop\track3D\sam\sam_vit_b.pth"  # Path to SAM model checkpoint
# model_type = "vit_b"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# sam = sam_model_registry[model_type](checkpoint=model_path)
# sam.to(device=device)
# predictor = SamPredictor(sam)

# # Segment Road function for each frame
# def segment_road_frame(frame):
#     try:
#         # Convert the frame to RGB (OpenCV reads as BGR)
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(image_rgb)
        
#         # Set the image for the SAM model
#         predictor.set_image(image_rgb)

#         # Define the input point and label for segmentation (same as before)
#         input_point = np.array([[frame.shape[1] // 2, int(frame.shape[0] * 0.8)]])  # Point in the lower center
#         input_label = np.array([1])  # Label for road (assuming label 1)

#         # Get the mask prediction
#         masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

#         # Select the largest mask (road segmentation)
#         largest_mask = max(masks, key=lambda x: np.sum(x))
        
#         # Convert to binary mask
#         road_mask = (largest_mask * 255).astype(np.uint8)

#         # Overlay road mask on the original frame (green for road)
#         frame_with_mask = frame.copy()
#         frame_with_mask[road_mask > 0] = [0, 255, 0]  # Highlight roads in green

#         return frame_with_mask
#     except Exception as e:
#         print(f"Error during segmentation: {e}")
#         return frame

# # Stream video frames with segmentation applied
# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Use webcam (0) or video file path
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Segment the current frame
#         segmented_frame = segment_road_frame(frame)

#         # Encode the frame in JPEG format
#         _, buffer = cv2.imencode('.jpg', segmented_frame)
#         frame_bytes = buffer.tobytes()

#         # Yield the frame as a byte stream for real-time display
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True, threaded=True)


# ################################ For Video ################################
# from flask import Flask, request, jsonify, send_file
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# from flask_cors import CORS
# import io
# from segment_anything import sam_model_registry, SamPredictor

# app = Flask(__name__)
# CORS(app)  # To handle CORS for frontend

# # Setup the SAM model
# model_path = r"C:\Users\HP\Desktop\track3D\sam\sam_vit_b.pth"  # Path to SAM model checkpoint
# model_type = "vit_b"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# sam = sam_model_registry[model_type](checkpoint=model_path)
# sam.to(device=device)
# predictor = SamPredictor(sam)

# # Segment Road function for each frame
# def segment_road_frame(frame):
#     try:
#         # Convert the frame to RGB (OpenCV reads as BGR)
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(image_rgb)
        
#         # Set the image for the SAM model
#         predictor.set_image(image_rgb)

#         # Define the input point and label for segmentation (same as before)
#         input_point = np.array([[frame.shape[1] // 2, int(frame.shape[0] * 0.8)]])  # Point in the lower center
#         input_label = np.array([1])  # Label for road (assuming label 1)

#         # Get the mask prediction
#         masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)

#         # Select the largest mask (road segmentation)
#         largest_mask = max(masks, key=lambda x: np.sum(x))
        
#         # Convert to binary mask
#         road_mask = (largest_mask * 255).astype(np.uint8)

#         # Overlay road mask on the original frame (green for road)
#         frame_with_mask = frame.copy()
#         frame_with_mask[road_mask > 0] = [0, 255, 0]  # Highlight roads in green

#         return frame_with_mask
#     except Exception as e:
#         print(f"Error during segmentation: {e}")
#         return frame

# @app.route('/process_video', methods=['POST'])
# def process_video():
#     video_file = request.files['video']
#     if not video_file:
#         return jsonify({"error": "No video file found"}), 400

#     # Read the uploaded video
#     video_path = "uploaded_video.mp4"
#     video_file.save(video_path)
    
#     # Open video file
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         return jsonify({"error": "Failed to open video"}), 500

#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Output video setup
#     output_path = "segmented_video.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     # Process each frame of the video
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Apply road segmentation
#         segmented_frame = segment_road_frame(frame)

#         # Write the segmented frame to the output video
#         out.write(segmented_frame)

#     # Release resources
#     cap.release()
#     out.release()

#     # Return the processed video as a downloadable file
#     return send_file(output_path, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)
