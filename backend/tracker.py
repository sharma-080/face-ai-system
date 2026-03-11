from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)

active_people = {}

def track_faces(detections, frame):

    tracks = tracker.update_tracks(detections, frame=frame)

    results = []

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        results.append({
            "id": track_id,
            "box": (int(l), int(t), int(r - l), int(b - t))
        })

    return results