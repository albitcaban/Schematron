#!/usr/bin/env python3
"""Interactive rotated-rectangle ROI editor for PMA videos.

Outputs a zone_rois.json next to the selected video (unless --output is provided).
The JSON contains native-video pixel-space polygons compatible with OpenCV
pointPolygonTest. No frame resizing is applied to stored coordinates.

Requires: pip install opencv-python numpy
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

NAMES = ('platform', 'reward')
COLORS = {'platform': (60, 215, 80), 'reward': (0, 185, 255)}  # BGR


def select_video_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title='Choose PMA session video',
            filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv'), ('All files', '*.*')]
        )
        root.destroy()
        return path
    except Exception as exc:
        raise RuntimeError('File dialog unavailable. Run with --video /path/to/video.mp4') from exc


def polygon(roi):
    rect = (tuple(roi['center']), tuple(roi['size']), float(roi['angle_deg']))
    return cv2.boxPoints(rect).astype(np.float64)


def inside(roi, xy):
    return cv2.pointPolygonTest(polygon(roi).astype(np.float32), xy, False) >= 0


def draw_overlay(frame, rois, active, frame_idx, frame_total, dragging=None):
    output = frame.copy()
    for name in NAMES:
        roi = rois.get(name)
        if not roi:
            continue
        pts = np.rint(polygon(roi)).astype(np.int32)
        cv2.polylines(output, [pts], True, COLORS[name], 3 if name == active else 2, cv2.LINE_AA)
        midpoint = tuple(np.rint(roi['center']).astype(int))
        cv2.circle(output, midpoint, 4, COLORS[name], -1, cv2.LINE_AA)
        cv2.putText(output, name.upper(), (int(pts[:, 0].min()), max(20, int(pts[:, 1].min()) - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[name], 2, cv2.LINE_AA)
    shade = output.copy()
    cv2.rectangle(shade, (0, 0), (output.shape[1], 104), (8, 8, 8), -1)
    cv2.addWeighted(shade, 0.63, output, 0.37, 0, output)
    lines = [
        f'Frame {frame_idx+1}/{frame_total} | Active: {active.upper()} | press P/R then drag to draw; drag inside to move',
        'Q/E rotate 1 deg | +/- resize 2% | ,/. frame step | S save | X delete active | ESC quit',
        'Frame trackbar below lets you jump through the video. Save polygons in original video pixels.'
    ]
    for i, line in enumerate(lines):
        cv2.putText(output, line, (12, 28 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.57, (250, 250, 250), 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, help='Input video (omit to choose via dialog)')
    parser.add_argument('--output', type=Path, help='Output JSON path, default video_dir/zone_rois.json')
    parser.add_argument('--max-width', type=int, default=1150, help='Display width; original coordinate system preserved')
    args = parser.parse_args()
    video = args.video or Path(select_video_dialog())
    if not str(video) or not video.is_file():
        raise SystemExit('No video selected or video does not exist.')
    video = video.resolve()
    output_path = args.output or (video.parent / 'zone_rois.json')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f'Could not open video: {video}')
    nframes = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1., args.max_width / max(width, 1))
    state = {'frame': 0, 'active': 'platform', 'rois': {}, 'drag': None,
             'ignore_trackbar': False, 'dirty': True}
    # Load existing saved ROIs only for a matching video geometry.
    if output_path.exists():
        try:
            stored = json.loads(output_path.read_text())
            if stored.get('video_width_px') == width and stored.get('video_height_px') == height:
                for name in NAMES:
                    if name in stored.get('zones', {}):
                        state['rois'][name] = stored['zones'][name]['rotated_rect']
                state['frame'] = min(nframes-1, max(0, int(stored.get('selected_frame', 0))))
                print(f'Loaded ROIs from {output_path}')
        except (ValueError, KeyError, TypeError) as exc:
            print(f'Could not load previous ROIs: {exc}')
    win = 'PMA ROI Editor'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def seek(new_idx):
        state['frame'] = int(np.clip(new_idx, 0, nframes-1))
        state['dirty'] = True

    def on_trackbar(value):
        if not state['ignore_trackbar']:
            seek(value)

    def on_mouse(event, x, y, flags, userdata):
        xy = np.array([x / scale, y / scale], dtype=np.float64)
        name = state['active']
        if event == cv2.EVENT_LBUTTONDOWN:
            roi = state['rois'].get(name)
            if roi and inside(roi, tuple(xy)):
                state['drag'] = ('move', xy.copy(), np.array(roi['center'], dtype=float))
            else:
                state['drag'] = ('draw', xy.copy())
        elif event == cv2.EVENT_MOUSEMOVE and state['drag'] is not None:
            drag = state['drag']
            if drag[0] == 'move':
                newcenter = drag[2] + xy - drag[1]
                state['rois'][name]['center'] = [float(newcenter[0]), float(newcenter[1])]
            elif drag[0] == 'draw':
                origin = drag[1]
                midpoint = (origin + xy)/2
                dimensions = np.maximum(abs(xy-origin), 2)
                state['rois'][name] = {
                    'center': midpoint.tolist(), 'size': dimensions.tolist(), 'angle_deg': 0.0
                }
            state['dirty'] = True
        elif event == cv2.EVENT_LBUTTONUP:
            state['drag'] = None

    cv2.createTrackbar('Frame', win, state['frame'], max(1, nframes-1), on_trackbar)
    cv2.setMouseCallback(win, on_mouse)

    def save():
        if any(name not in state['rois'] for name in NAMES):
            print('Both platform and reward rectangles must be drawn before saving.')
            return
        doc = {
            'version': 1,
            'source_video': video.name,
            'video_width_px': width, 'video_height_px': height,
            'video_fps_metadata': fps,
            'selected_frame': state['frame'],
            'chamber_dimensions_cm': {'side_a': 18.796, 'side_b': 20.574},
            'platform_point_threshold': 4,
            'platform_landmarks': ['nose', 'left_ear', 'right_ear', 'implant',
                                   'mid_back', 'left_hip', 'right_hip', 'tail_base'],
            'excluded_landmarks': ['tail_midpoint'],
            'zones': {}
        }
        for name in NAMES:
            roi = state['rois'][name]
            doc['zones'][name] = {
                'type': 'rotated_rectangle',
                'rotated_rect': roi,
                'polygon_px': np.round(polygon(roi), 3).tolist(),
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(doc, indent=2))
        print(f'Saved: {output_path}')

    try:
        while True:
            if state['dirty']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, state['frame'])
                okay, frame = cap.read()
                if not okay:
                    print(f'Could not read frame {state["frame"]}')
                    break
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                output = draw_overlay(frame, state['rois'], state['active'], state['frame'], nframes)
                if scale != 1:
                    output = cv2.resize(output, (round(width*scale), round(height*scale)))
                cv2.imshow(win, output)
                state['ignore_trackbar'] = True
                cv2.setTrackbarPos('Frame', win, state['frame'])
                state['ignore_trackbar'] = False
                state['dirty'] = False
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('z')):
                break
            if key in (ord('p'), ord('r')):
                state['active'] = 'platform' if key == ord('p') else 'reward'
                state['dirty'] = True
            elif key in (ord('q'), ord('e')):
                roi = state['rois'].get(state['active'])
                if roi:
                    roi['angle_deg'] += -1 if key == ord('q') else 1
                    state['dirty'] = True
            elif key in (ord('+'), ord('='), ord('-'), ord('_')):
                roi = state['rois'].get(state['active'])
                if roi:
                    factor = 1.02 if key in (ord('+'), ord('=')) else 1/1.02
                    roi['size'] = [max(2, float(n*factor)) for n in roi['size']]
                    state['dirty'] = True
            elif key in (ord('x'), ord('X')):
                state['rois'].pop(state['active'], None)
                state['dirty'] = True
            elif key in (ord(','), ord('<')):
                seek(state['frame']-1)
            elif key in (ord('.'), ord('>')):
                seek(state['frame']+1)
            elif key in (ord('s'), ord('S')):
                save()
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
