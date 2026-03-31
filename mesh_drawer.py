import mediapipe as mp
import cv2

indexes = [362, 263, 385, 380, 387, 373, 33, 133, 160, 144, 158, 153]  
drawing_points = []

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

style1 = mp_drawing.DrawingSpec(color=(0,255,0), circle_radius=1)
style2 = mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(refine_landmarks = True, max_num_faces=1) as face_mesh:
    while True:
        rec, frame = cap.read()
        if not rec:
            continue
        
        vid = cv2.flip(frame, 1)
        vid_rgb = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)

        vid_rgb.flags.writeable = False
        results = face_mesh.process(vid_rgb)
        
        if results.multi_face_landmarks:

            for landmark in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(vid, landmark, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=style1, connection_drawing_spec=style2)
        
        cv2.imshow("Face Mesh", vid)

        if cv2.waitKey(1) & 0xff == ord('x'):
            break

cv2.destroyWindow("Face Mesh")


with mp_face_mesh.FaceMesh(refine_landmarks = True, max_num_faces=1) as face_mesh:
    while True:
        rec, frame = cap.read()
        if not rec:
            continue
        
        vid = cv2.flip(frame, 1)
        vid_rgb = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)

        vid_rgb.flags.writeable = False
        results = face_mesh.process(vid_rgb)

        h, w, _ = vid.shape
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            for point_counter in indexes:
                
                point = landmarks.landmark[point_counter]
                
                cx_raw, cy_raw = int(point.x*w), int(point.y*h)
                
                cx, cy = max(0, min(cx_raw, w - 1)), max(0, min(cy_raw, h - 1))
                
                cv2.circle(vid, (cx,cy), 2, (255,255,0), 2)
                drawing_points.append((point.x, point.y))
                
                
        cv2.imshow("Face Mesh", vid)

        if cv2.waitKey(1) & 0xff == ord('x'):
            break

cap.release()

cv2.destroyAllWindows()


