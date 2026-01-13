import cv2

def extraire_segment_video(input_path, output_path, t1, t2):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(t1 * fps)
    end_frame = int(t2 * fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

extraire_segment_video('./temp/Phillippe_amont_aout.27.mer.12h43_0.mp4', './temp/Phillippe_amont_aout.27.mer.12h43_0_cut_bis.m4v', 21.5*60, 28.5*60)