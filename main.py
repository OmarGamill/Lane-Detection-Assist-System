import argparse
from LaneDetection_openCV import LaneModule 
from  UlterFastLaneDetection import videoLaneDetection

def run_video_lane_detection():
    print("Running video lane detection...")

    videoLaneDetection.main()

    print("Video lane detection is complete.")

def run_lane_module():
    print("Running lane module...")
    LaneModule.main()
    print("Lane module is complete.")

def main():
    parser = argparse.ArgumentParser(description="Choose a file to run")
    parser.add_argument(
        'file',
        choices=['videoLaneDetection', 'LaneModule'],
        help="The file to run"
    )

    args = parser.parse_args()

    if args.file == 'videoLaneDetection':
        run_video_lane_detection()
    elif args.file == 'LaneModule':
        run_lane_module()

if __name__ == "__main__":
    main()
