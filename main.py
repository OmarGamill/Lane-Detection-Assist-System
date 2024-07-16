import argparse
from LaneDetection_openCV import LaneModule 
from  UlterFastLaneDetection import videoLaneDetection

def run_video_lane_detection(path_to_video):
    print("Running video lane detection...")

    videoLaneDetection.main(path_to_video)

    print("Video lane detection is complete.")

def run_lane_module(path_to_video):
    print("Running lane module...")
    LaneModule.main(path_to_video)
    print("Lane module is complete.")

def main():
    parser = argparse.ArgumentParser(description="Choose a file to run")
    parser.add_argument(
        'file',
        choices=['videoLaneDetection', 'LaneModule'],
        help="The file to run"
    )
    parser.add_argument(
        '--video',
        type=str,
        help="The path to the video file for video lane detection"
    )

    args = parser.parse_args()

    if args.file == 'videoLaneDetection':
        if not args.video:
            parser.error("--video argument is required for videoLaneDetection")
        run_video_lane_detection(args.video)
    elif args.file == 'LaneModule':
        if not args.video:
            parser.error("--video argument is required for LaneModule")
        run_lane_module(args.video)
        
if __name__ == "__main__":
    main()
