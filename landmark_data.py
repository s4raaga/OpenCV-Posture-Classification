# IMPORTS #
import cv2, math, time, queue, csv
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# READABILITY VARS #
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

"""# CSV SETUP #
num_points = 33
data_header = ['class']

for val in range(1, num_points+1):
    data_header += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

with open('dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(data_header)
"""
# POSE CLASSES #
class_name = 'slouch'


# FUNCTIONS #
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
        
        #Get key points for analysis.
        get_key_points(pose, h, w)

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



def get_key_points(pose, frame_height: int, frame_width: int) -> None:
    """
    """
        
    left_shoulder = (int(pose[12].x * frame_width), int(pose[12].y * frame_height))
    right_shoulder = (int(pose[11].x * frame_width), int(pose[11].y * frame_height))
    left_mouth = (int(pose[10].x * frame_width), int(pose[10].y * frame_height))
    right_mouth = (int(pose[9].x * frame_width), int(pose[9].y * frame_height))
    left_outer_eye = (int(pose[8].x * frame_width), int(pose[8].y * frame_height))
    right_outer_eye = (int(pose[7].x * frame_width), int(pose[7].y * frame_height))

    # CALCULATE ANGLES #


def calculate_angle(p1: tuple[int], p2: tuple[int], midpoint: int) -> int:
    """
    Calculates the planar angle at the midpoint of 3 2D points.

    """
    
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    midpoint = np.array(midpoint, dtype=float)

    vec_1 = p1 - midpoint
    vec_2 = p2 = midpoint

    cosine = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)) # Angle between the 2 lines.

    # atan2(cross, dot) gives signed angle in (-π, π]
    cross = vec_1[0]*vec_2[1] - vec_1[1]*vec_2[0] # 2-D cross-product z-component
    dot = np.dot(vec_1, vec_2)

    angle_rad = math.atan2(cross, dot)
    angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    return angle_rad



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


    # READING & DISPLAYING FRAMES #
    while True:

        ret, frame_bgr = cap.read()
        if not ret:
            continue


        # FRAME PROCESSING #
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #Convert frame to rgb.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) #Convert frame to a MediaPipe image object.
        timestamp_ms = int(time.time() * 1000)
        

        # LANDMARKER OBJECT #
        landmarker.detect_async(mp_image, timestamp_ms)


        # GET QUEUE ITEMS #
        
        try:
            annotated_frame, pose_landmarks = frame_q.get_nowait()

            # EXPORT LANDMARK DATA TO CSV ON KEY PRESS (q) #
            
            row = [class_name]
            try:
                for landmark in pose_landmarks[0]:
                    row += [landmark.x, landmark.y, landmark.z, landmark.visibility]

                with open('dataset.csv', 'a', newline='') as f:
                    csv.writer(f).writerow(row) #Export only the first person's landmark data.
            except:
                pass

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