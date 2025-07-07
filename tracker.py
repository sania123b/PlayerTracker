import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main():
    # Initialize models (YOUR EXACT PREFERENCES)
    model = YOLO("best.pt")  # Your custom model
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.4,
        nn_budget=100,
        embedder="mobilenet",
        half=False
    )

    cap = cv2.VideoCapture("15sec_input_720p.mp4")
    
    # VERIFICATION STEPS
    print("Starting player tracking...")
    if not cap.isOpened():
        print("Error: Could not open video file!")
        return
    
    # PROCESSING LOOP (YOUR STYLE)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # DETECTION (YOUR EXACT IMPLEMENTATION)
        results = model(frame, conf=0.5)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2-x1, y2-y1], float(box.conf[0]), 0))

        # TRACKING (YOUR PREFERRED STYLE)
        tracks = tracker.update_tracks(detections, frame=frame)

        # VISUALIZATION (GUARANTEED BOXES/IDs)
        boxes_shown = 0
        for track in tracks:
            if track.is_confirmed():
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                # YOUR GREEN BOXES AND IDs
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                boxes_shown += 1

        # DIAGNOSTIC FEEDBACK
        print(f"Frame processed - Detections: {len(detections)}, Boxes shown: {boxes_shown}")
        
        cv2.imshow("Player Tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking complete")

if __name__ == "__main__":
    main()