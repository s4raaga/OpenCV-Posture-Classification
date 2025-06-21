# IMPORTS #
import cv2, time, queue, pickle
import pandas as pd
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# READABILITY VARS #
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# CALLBACK FUNCTION #
def on_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """
    Annotates landmarks onto frame & queues annotated frame and list of landmarks.
    """

    annotated_frame = output_image.numpy_view().copy()
    pose_landmarks_list = result.pose_landmarks

    #Get frame dimensions.
    h, w = annotated_frame.shape[:2]

    # ANNOTATE LANDMARKS #

    #Iterate over detected poses and annotate
    for pose in pose_landmarks_list:

        #Convert plain python list into a NormalizedLandmarkList protobuf.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose])

        #Draw the pose landmarks.
        mp.solutions.drawing_utils.draw_landmarks(annotated_frame, 
                                                    pose_landmarks_proto, 
                                                    mp.solutions.pose.POSE_CONNECTIONS, 
                                                    mp.solutions.drawing_styles.get_default_pose_landmarks_style())



    # ADD ANNOTATED FRAME AND LANDMARKS LIST TO QUEUE #
    # Replace old frame without blocking (returns immediately).
    try:
        frame_q.get_nowait()
    except queue.Empty:
        pass

    frame_q.put_nowait((annotated_frame, pose_landmarks_list))



# SET UP QUEUE #
frame_q = queue.Queue(maxsize=2)

# LANDMARKER OBJECT #
model_path = '/Users/saraarsenio/Documents/GitHub/Pose-Estimation/pose_landmarker_lite.task'
options = PoseLandmarkerOptions(base_options=BaseOptions(model_path), running_mode = VisionRunningMode.LIVE_STREAM, result_callback=on_callback)

with PoseLandmarker.create_from_options(options) as landmarker:
    
    # VIDEO CAPTURE #
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("ERROR: CAMERA NOT OPENED")

    # LOAD CLASSIFIER MODEL #
    with open('pose_detection.pkl', 'rb') as f:
        model = pickle.load(f)

    # READING & DISPLAYING FRAMES #
    while True:

        ret, frame_bgr = cap.read()
        if not ret:
            continue
        
        # FRAME DIMENSIONS #
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FRAME PROCESSING #
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #Convert frame to rgb.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) #Convert frame to a MediaPipe image object.
        timestamp_ms = int(time.time() * 1000)

        # LANDMARKER OBJECT #
        landmarker.detect_async(mp_image, timestamp_ms)


        # GET QUEUE ITEMS #
        
        try:
            annotated_frame, pose_landmarks = frame_q.get_nowait()

            try:

                pose_class = model.predict(X)[0] #Predict class.
                #pose_prob = model.predict_proba(X)[0] #Predict probability of particular class.

                # ANNOTATE CLASS ONTO FRAME #
                text_coords = (int(pose_landmarks[0][0].x * width), int(pose_landmarks[0][0].y * height))
                cv2.putText(annotated_frame, pose_class, text_coords, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
            
            except:
                pass

        #If queue is empty, display unannotated frame.
        except queue.Empty:
            annotated_frame = frame_rgb
        

        # DISPLAY ANNOTATED FRAME WITH OPENCV #
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR) #Convert back to bgr.
        cv2.imshow("Pose Estimation", annotated_frame_bgr)
    

        # EXIT #
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


    cap.release()
    cv2.destroyAllWindows()