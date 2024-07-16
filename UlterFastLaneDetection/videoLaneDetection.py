import cv2
import matplotlib.pyplot as plt
from UlterFastLaneDetection.utilities.ultrafastLaneDetector  import UltrafastLaneDetector, ModelType


def main(path_to_video):
    model_path = "E:\\git_hup_project\\UlterFastLaneDetection\\models\\mobilenet_large.pth"
    model_type = ModelType.TUSIMPLE
    use_gpu = False


    # Initialize video
    cap = cv2.VideoCapture(path_to_video)


    # Initialize lane detection model
    lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

    cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	
    num_frames = 0



    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Process the frame with your lane detector
        output = lane_detector.detect_lanes(frame)
        x,y = UltrafastLaneDetector.prosess_points(output[1], output[0])

        angle = UltrafastLaneDetector.fit_line_and_angle(x,y)
        
        frame_with_lanes = UltrafastLaneDetector.draw_lanes(frame, output[0], output[1], 1280, 720, draw_points=True)
        
        cv2.imshow('Frame', frame_with_lanes)
        #plt.imshow(frame_with_lanes)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

