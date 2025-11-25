from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import subprocess
import time
import os
import uuid
import shutil
import logging
from pathlib import Path
import sys
import asyncio
import torch
from insightface_func.face_detect_crop_multi import Face_detect_crop
import cv2
import base64
import numpy as np
from typing import List, Optional
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SimSwap Face Swap API",
    description="Simple face swapping service",
    version="1.0.0"
)

# Configuration
CONFIG = {
    "upload_folder": Path("./uploads"),
    "output_folder": Path("./output"), 
    "temp_folder": Path("./temp_results"),
    "max_file_size": 500 * 1024 * 1024,  # 500MB
    "timeout": 2400,  # 5 minutes
    "arcface_path": "arcface_model/arcface_checkpoint.tar"
}

# Create directories
for folder in [CONFIG["upload_folder"], CONFIG["output_folder"], CONFIG["temp_folder"]]:
    folder.mkdir(parents=True, exist_ok=True)

def cleanup_files(*paths):
    """Background cleanup task"""
    for path in paths:
        try:
            if isinstance(path, str):
                path = Path(path)
            if path and path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Cleanup failed for {path}: {e}")

def get_optimal_batch_size():
    """Calculate optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 4
    
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        
        # Tối ưu cho RTX 3050 (4GB VRAM)
        if "3050" in gpu_name or gpu_memory_gb <= 4.5:
            optimal_batch = 10  # Safe cho 4GB
        elif gpu_memory_gb <= 6:
            optimal_batch = 16
        elif gpu_memory_gb <= 8:
            optimal_batch = 20
        else:
            optimal_batch = 24
        
        logger.info(f"✅ GPU: {gpu_name} ({gpu_memory_gb:.1f}GB) -> Batch size: {optimal_batch}")
        return optimal_batch
    except Exception as e:
        logger.warning(f"⚠️ Cannot detect GPU: {e}, using batch_size=8")
        return 8

async def save_uploaded_file(upload_file: UploadFile, save_path: Path):
    """Save uploaded file with size validation"""
    content = await upload_file.read()
    if len(content) > CONFIG["max_file_size"]:
        raise HTTPException(status_code=400, detail=f"File {upload_file.filename} is too large")
    
    with open(save_path, 'wb') as f:
        f.write(content)
    
    return len(content)

@app.get("/")
async def root():
    return {"message": "SimSwap Face Swap API", "status": "online"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_available": os.path.exists(CONFIG["arcface_path"])
    }

@app.post("/merge-video")
async def merge_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    music: UploadFile = File(...),
    lyrics: UploadFile = File(None)
):
    request_id = str(uuid.uuid4())
    video_path = CONFIG["upload_folder"] / f"{request_id}_video.mp4"
    music_path = CONFIG["upload_folder"] / f"{request_id}_music.mp3"
    lyrics_path = CONFIG["upload_folder"] / f"{request_id}_lyrics.mp3" if lyrics else None
    output_path = CONFIG["output_folder"] / f"{request_id}_merged.mp4"

    try:
        await save_uploaded_file(video, video_path)
        await save_uploaded_file(music, music_path)
        if lyrics:
            await save_uploaded_file(lyrics, lyrics_path)

        # Convert to absolute POSIX paths
        v_path = os.path.abspath(str(video_path)).replace("\\", "/")
        m_path = os.path.abspath(str(music_path)).replace("\\", "/")

        video_clip = VideoFileClip(v_path)
        music_clip = AudioFileClip(m_path)

        audio_clips = [music_clip]
        if lyrics and lyrics_path:
            l_path = os.path.abspath(str(lyrics_path)).replace("\\", "/")
            lyrics_clip = AudioFileClip(l_path)
            audio_clips.append(lyrics_clip)

        final_audio = CompositeAudioClip(audio_clips)
        audio_duration = final_audio.duration
        video_duration = video_clip.duration

        if video_duration < audio_duration:
            repeats = int(audio_duration / video_duration) + 1
            video_clip = concatenate_videoclips([video_clip] * repeats)
        video_clip = video_clip.subclip(0, audio_duration)

        final_video = video_clip.set_audio(final_audio)
        final_video.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=str(CONFIG["upload_folder"] / f"{request_id}_temp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None
        )

        video_clip.close()
        music_clip.close()
        if lyrics and lyrics_path:
            lyrics_clip.close()
        final_audio.close()
        final_video.close()

        background_tasks.add_task(cleanup_files, video_path, music_path, lyrics_path, output_path)

        return FileResponse(
            path=str(output_path),
            filename=f"merged_{request_id}.mp4",
            media_type="video/mp4"
        )

    except Exception as e:
        cleanup_files(video_path, music_path, lyrics_path, output_path)
        raise HTTPException(status_code=500, detail=f"Merge failed: {e}")

@app.post("/detect-faces-video")
async def detect_faces_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Target video"),
    max_people: int = 4
):
    """
    Detect unique people using simple image similarity (no embeddings)
    """
    request_id = str(uuid.uuid4())
    video_path = None
    total_start = time.time()
    
    logger.info(f"[{request_id}] ========== UNIQUE FACE DETECTION STARTED ==========")
    
    try:
        # Validate
        if not video.content_type or not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        # Save video
        video_path = CONFIG["upload_folder"] / f"{request_id}_video.mp4"
        video_size = await save_uploaded_file(video, video_path)
        logger.info(f"[{request_id}] ✓ Video saved: {video_size/1024:.0f}KB")
        
        # Initialize detector
        logger.info(f"[{request_id}] Loading face detector...")
        app_detector = Face_detect_crop(name='antelope', root='./insightface_func/models')
        app_detector.prepare(ctx_id=0, det_thresh=0.4, det_size=(640, 640))
        logger.info(f"[{request_id}] ✓ Detector loaded")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"[{request_id}] ✓ Video: {total_frames} frames @ {fps:.1f} fps")
        
        # Process frames with histogram-based clustering
        process_start = time.time()
        unique_people = []  # {histograms: [], images: [], frames: []}
        sample_rate = max(1, int(fps / 5)) if fps > 0 else 10
        frame_idx = 0
        total_faces = 0
        
        HISTOGRAM_THRESHOLD = 0.9  # Similarity threshold for histograms
        
        logger.info(f"[{request_id}] Processing with histogram clustering...")
        
        def compute_histogram(img):
            """Compute color histogram for face image"""
            # Convert to HSV for better color matching
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Compute histogram for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
            
            # Normalize
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            # Concatenate
            return np.concatenate([hist_h, hist_s, hist_v])
        
        def compare_histograms(hist1, hist2):
            """Compare two histograms using correlation"""
            return cv2.compareHist(
                hist1.astype(np.float32), 
                hist2.astype(np.float32), 
                cv2.HISTCMP_CORREL
            )

        def score_face(face_img, landmarks):
            # 1. Pose score
            if landmarks is not None:
                # landmarks: [[x1,y1], [x2,y2], ...]
                # dùng hai mắt + mũi để ước lượng độ lệch
                left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
                eye_dx = right_eye[0] - left_eye[0]
                eye_dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(eye_dy, eye_dx))
                pose_score = 1 / (1 + abs(angle))  # góc càng gần 0 càng tốt
            else:
                pose_score = 0.5

            # 2. Size score
            h, w = face_img.shape[:2]
            size_score = (w * h) / 10000.0

            # 3. Sharpness score
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var() / 100.0

            # 4. Tổng hợp
            return 0.5*pose_score + 0.3*size_score + 0.2*sharpness_score

        def expand_bbox(x1, y1, x2, y2, scale=1.2, img_shape=None):
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2

            new_w = int(w * scale)
            new_h = int(h * scale)

            new_x1 = max(0, cx - new_w // 2)
            new_y1 = max(0, cy - new_h // 2)
            new_x2 = min(img_shape[1], cx + new_w // 2)
            new_y2 = min(img_shape[0], cy + new_h // 2)

            return new_x1, new_y1, new_x2, new_y2

        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            bboxes, kpss = app_detector.det_model.detect(frame, max_num=0, metric='default')

            if bboxes is None or len(bboxes) == 0:
                logger.debug(f"[{request_id}] No faces detected at frame {frame_idx}")
                frame_idx += 1
                continue

            # Crop faces với bbox mở rộng
            align_img_list = []
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2, score = bbox
                
                # Mở rộng bbox (scale=1.5 = rộng hơn 50%, bạn có thể điều chỉnh)
                expanded_bbox = expand_bbox(
                    int(x1), int(y1), int(x2), int(y2), 
                    scale=2.0,  # <-- Điều chỉnh giá trị này: 1.2-2.0
                    img_shape=frame.shape
                )
                
                # Crop face từ frame gốc
                face_img = frame[expanded_bbox[1]:expanded_bbox[3], 
                                expanded_bbox[0]:expanded_bbox[2]]
                
                # Resize về kích thước chuẩn
                if face_img.size > 0:
                    # Giữ nguyên kích thước hoặc resize lên lớn hơn
                    h, w = face_img.shape[:2]
                    if h < 512 or w < 512:
                        # Chỉ resize nếu quá nhỏ
                        face_img = cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_CUBIC)
                    
                    align_img_list.append(face_img)
            
            if len(align_img_list) > 0:  # Thêm check này cho chắc
                total_faces += len(align_img_list)
                logger.debug(f"[{request_id}] Detected {len(align_img_list)} faces at frame {frame_idx}")

                for i, face_img in enumerate(align_img_list):
                    # Lấy landmarks tương ứng (nếu đã lưu từ bước 1)
                    landmarks = kpss[i] if kpss is not None and i < len(kpss) else None
                    
                    # Compute histogram for this face
                    face_hist = compute_histogram(face_img)
                    
                    # Tính score cho face này
                    face_quality_score = score_face(face_img, landmarks)
                    
                    # Match with existing people...
                    best_match_idx = -1
                    best_similarity = 0
                    
                    for person_idx, person in enumerate(unique_people):
                        similarities = [compare_histograms(face_hist, stored_hist) for stored_hist in person['histograms']]
                        max_similarity = np.max(similarities)
                        
                        if max_similarity > best_similarity:
                            best_similarity = max_similarity
                            best_match_idx = person_idx
                    
                    if best_similarity > HISTOGRAM_THRESHOLD:
                        # Existing person
                        person = unique_people[best_match_idx]
                        
                        # Check if different enough (use max sim)
                        max_sim_to_stored = max([compare_histograms(face_hist, stored_hist) for stored_hist in person['histograms']])
                        if max_sim_to_stored < 0.95:
                            person['histograms'].append(face_hist)
                            
                            _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
                            person['images'].append(base64.b64encode(buffer).decode('utf-8'))
                            person['frames'].append(frame_idx)
                            
                            logger.debug(f"[{request_id}] Added variant to Person {best_match_idx+1} (max_sim: {max_sim_to_stored:.3f})")
                    
                    else:
                        # New person
                        if len(unique_people) >= max_people:
                            continue
                        
                        _, buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        unique_people.append({
                            'histograms': [face_hist],
                            'images': [base64.b64encode(buffer).decode('utf-8')],
                            'frames': [frame_idx]
                        })
                        
                        logger.info(f"[{request_id}] ✨ New person #{len(unique_people)} at frame {frame_idx} "
                                f"(best_max_sim with others: {best_similarity:.3f})")
            
            frame_idx += 1
            
            if frame_idx % 150 == 0:
                logger.info(f"[{request_id}] Frame {frame_idx}/{total_frames}, "
                        f"people: {len(unique_people)}, faces: {total_faces}")
        
        cap.release()
        process_time = time.time() - process_start

        results = []
        for idx, person in enumerate(unique_people):
            scores = []
            for i, img_b64 in enumerate(person['images']):
                img_data = base64.b64decode(img_b64)
                img_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                # Tính điểm
                score = score_face(img, None)  # nếu có landmarks thì truyền vào
                scores.append(score)

            best_idx = int(np.argmax(scores))  # chọn ảnh tốt nhất

            results.append(person['images'][best_idx])
        
        # Cleanup
        background_tasks.add_task(cleanup_files, video_path)
        
        total_time = time.time() - total_start
        logger.info(f"[{request_id}] ========== SUMMARY ==========")
        logger.info(f"[{request_id}] Method: Histogram clustering")
        logger.info(f"[{request_id}] Unique people: {len(results)}")
        logger.info(f"[{request_id}] Total faces detected: {total_faces}")
        logger.info(f"[{request_id}] Processing: {process_time:.2f}s")
        logger.info(f"[{request_id}] Total: {total_time:.2f}s")
        logger.info(f"[{request_id}] ========== COMPLETED ==========")
        
        return {
            'results': results
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] ✗ Error: {str(e)}")
        import traceback
        logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")
        if video_path and video_path.exists():
            cleanup_files(video_path)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        try:
            if 'cap' in locals():
                cap.release()
                cv2.destroyAllWindows()
        except:
            pass


@app.post("/swap-single-face")
async def swap_face(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Source face image"),
    video: UploadFile = File(..., description="Target video")
):
    """Face swap endpoint with detailed timing logs"""
    request_id = str(uuid.uuid4())
    total_start_time = time.time()
    
    # Initialize paths for cleanup
    image_path = None
    video_path = None
    output_path = None
    temp_path = None
    
    logger.info(f"[{request_id}] ========== FACE SWAP REQUEST STARTED ==========")
    
    try:
        # Task 1: Validate file types
        validation_start = time.time()
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")
        if not video.content_type or not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Invalid video format")
        validation_time = time.time() - validation_start
        logger.info(f"[{request_id}] ✓ File validation completed in {validation_time:.3f}s")
        
        # Task 2: Setup paths
        setup_start = time.time()
        image_path = CONFIG["upload_folder"] / f"{request_id}_image.jpg"
        video_path = CONFIG["upload_folder"] / f"{request_id}_video.mp4"
        output_path = CONFIG["output_folder"] / f"{request_id}_output.mp4"
        temp_path = CONFIG["temp_folder"] / request_id
        temp_path.mkdir(exist_ok=True)
        setup_time = time.time() - setup_start
        logger.info(f"[{request_id}] ✓ Path setup completed in {setup_time:.3f}s")
        
        # Task 3: Save uploaded files
        file_save_start = time.time()
        logger.info(f"[{request_id}] Saving uploaded files...")
        
        # Save image
        image_save_start = time.time()
        image_size = await save_uploaded_file(image, image_path)
        image_save_time = time.time() - image_save_start
        logger.info(f"[{request_id}] ✓ Image saved in {image_save_time:.3f}s ({image_size/1024:.0f}KB)")
        
        # Save video
        video_save_start = time.time()
        video_size = await save_uploaded_file(video, video_path)
        video_save_time = time.time() - video_save_start
        logger.info(f"[{request_id}] ✓ Video saved in {video_save_time:.3f}s ({video_size/1024:.0f}KB)")
        
        total_file_save_time = time.time() - file_save_start
        logger.info(f"[{request_id}] ✓ Total file save time: {total_file_save_time:.3f}s")
        
        # Task 4: Build command and setup
        cmd_setup_start = time.time()
        batch_size = get_optimal_batch_size()
        cmd = [
            sys.executable, 'test_video_swapsingle.py',
            '--isTrain', 'false',
            '--use_mask',
            '--crop_size', '512',
            '--no_simswaplogo',
            '--name', 'people',
            '--Arc_path', CONFIG["arcface_path"],
            '--pic_a_path', str(image_path),
            '--video_path', str(video_path),
            '--output_path', str(output_path),
            '--temp_path', str(temp_path),
            '--batchSize', str(batch_size),
        ]
        
        # Setup environment
        env = os.environ.copy()
        if torch.cuda.is_available():
            env['CUDA_VISIBLE_DEVICES'] = '0'
        
        cmd_setup_time = time.time() - cmd_setup_start
        logger.info(f"[{request_id}] ✓ Command setup completed in {cmd_setup_time:.3f}s (batch_size: {batch_size})")
        
        # Task 5: Face swap processing
        logger.info(f"[{request_id}] Starting AI face swap processing...")
        ai_process_start = time.time()
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=CONFIG["timeout"]
            )
            ai_process_time = time.time() - ai_process_start
            logger.info(f"[{request_id}] ✓ AI processing completed in {ai_process_time:.3f}s")
            
        except asyncio.TimeoutError:
            ai_process_time = time.time() - ai_process_start
            logger.error(f"[{request_id}] ✗ AI processing timeout after {ai_process_time:.3f}s")
            process.kill()
            await process.wait()
            raise HTTPException(status_code=504, detail="Processing timeout")
        
        # Task 6: Validation and result check
        validation_start = time.time()
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            stdout_msg = stdout.decode() if stdout else ""
            logger.error(f"[{request_id}] ✗ Processing failed: {error_msg}")
            if stdout_msg:
                logger.error(f"[{request_id}] Stdout: {stdout_msg}")
            raise HTTPException(status_code=500, detail="Face swap processing failed")
        
        if not output_path.exists():
            logger.error(f"[{request_id}] ✗ Output file not generated")
            raise HTTPException(status_code=500, detail="Output file not generated")
        
        # Check output file size
        output_size = output_path.stat().st_size
        validation_time = time.time() - validation_start
        logger.info(f"[{request_id}] ✓ Result validation completed in {validation_time:.3f}s (output: {output_size/1024:.0f}KB)")
        
        # Task 7: Cleanup scheduling
        cleanup_start = time.time()
        background_tasks.add_task(cleanup_files, image_path, video_path, output_path, temp_path)
        cleanup_time = time.time() - cleanup_start
        logger.info(f"[{request_id}] ✓ Cleanup scheduled in {cleanup_time:.3f}s")
        
        # Final summary
        total_time = time.time() - total_start_time
        logger.info(f"[{request_id}] ========== PERFORMANCE SUMMARY ==========")
        logger.info(f"[{request_id}] File validation:  {validation_time:.3f}s ({validation_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] Path setup:       {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] File I/O:         {total_file_save_time:.3f}s ({total_file_save_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}]   - Image save:   {image_save_time:.3f}s")
        logger.info(f"[{request_id}]   - Video save:   {video_save_time:.3f}s")
        logger.info(f"[{request_id}] Command setup:    {cmd_setup_time:.3f}s ({cmd_setup_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] AI processing:    {ai_process_time:.3f}s ({ai_process_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] Result validation: {validation_time:.3f}s ({validation_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] Cleanup setup:    {cleanup_time:.3f}s ({cleanup_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] TOTAL TIME:       {total_time:.3f}s")
        logger.info(f"[{request_id}] ========== REQUEST COMPLETED ==========")
        
        return FileResponse(
            path=str(output_path),
            filename=f"swapped_{request_id}.mp4",
            media_type="video/mp4"
        )
        
    except HTTPException:
        total_time = time.time() - total_start_time
        logger.error(f"[{request_id}] ✗ Request failed after {total_time:.3f}s")
        cleanup_files(image_path, video_path, output_path, temp_path)
        raise
    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"[{request_id}] ✗ Unexpected error after {total_time:.3f}s: {str(e)}")
        cleanup_files(image_path, video_path, output_path, temp_path)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/swap-multi-faces")
async def swap_multi_faces(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Target video"),
    src_faces: List[UploadFile] = File(..., description="Source face images (people to replace in video)"),
    dst_faces: List[UploadFile] = File(..., description="Destination face images (new faces to swap in)")
):
    """
    Swap multiple specific faces in video
    - Each SRC face in video will be replaced by corresponding DST face
    - SRC_01 -> DST_01, SRC_02 -> DST_02, etc.
    """
    request_id = str(uuid.uuid4())
    total_start_time = time.time()
    
    # Initialize paths
    video_path = None
    output_path = None
    temp_path = None
    multispecific_dir = None
    
    logger.info(f"[{request_id}] ========== MULTI-FACE SWAP REQUEST STARTED ==========")
    
    try:
        # Task 1: Validate inputs
        validation_start = time.time()
        
        if not video.content_type or not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        if len(src_faces) == 0 or len(dst_faces) == 0:
            raise HTTPException(status_code=400, detail="At least one source and destination face required")
        
        if len(src_faces) != len(dst_faces):
            raise HTTPException(
                status_code=400, 
                detail=f"Number of source faces ({len(src_faces)}) must match destination faces ({len(dst_faces)})"
            )
        
        # Validate all face images
        for idx, face in enumerate(src_faces + dst_faces):
            if not face.content_type or not face.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Invalid image format for face #{idx+1}")
        
        validation_time = time.time() - validation_start
        logger.info(f"[{request_id}] ✓ Validation completed in {validation_time:.3f}s")
        
        # Task 2: Setup paths
        setup_start = time.time()
        video_path = CONFIG["upload_folder"] / f"{request_id}_video.mp4"
        output_path = CONFIG["output_folder"] / f"{request_id}_multi_output.mp4"
        temp_path = CONFIG["temp_folder"] / request_id
        multispecific_dir = CONFIG["upload_folder"] / f"{request_id}_multispecific"
        
        # Create directories
        temp_path.mkdir(parents=True, exist_ok=True)
        multispecific_dir.mkdir(parents=True, exist_ok=True)
        
        setup_time = time.time() - setup_start
        logger.info(f"[{request_id}] ✓ Path setup completed in {setup_time:.3f}s")
        
        # Task 3: Save files
        file_save_start = time.time()
        logger.info(f"[{request_id}] Saving uploaded files...")
        
        # Save video
        video_save_start = time.time()
        video_size = await save_uploaded_file(video, video_path)
        video_save_time = time.time() - video_save_start
        logger.info(f"[{request_id}] ✓ Video saved in {video_save_time:.3f}s ({video_size/1024:.0f}KB)")
        
        # Save face images
        faces_save_start = time.time()
        
        # Save SRC faces
        for idx, src_face in enumerate(src_faces):
            face_path = multispecific_dir / f"SRC_{idx+1:02d}.jpg"
            await save_uploaded_file(src_face, face_path)
        
        # Save DST faces
        for idx, dst_face in enumerate(dst_faces):
            face_path = multispecific_dir / f"DST_{idx+1:02d}.jpg"
            await save_uploaded_file(dst_face, face_path)
        
        faces_save_time = time.time() - faces_save_start
        logger.info(f"[{request_id}] ✓ {len(src_faces)} face pairs saved in {faces_save_time:.3f}s")
        
        total_file_save_time = time.time() - file_save_start
        logger.info(f"[{request_id}] ✓ Total file save time: {total_file_save_time:.3f}s")
        
        # Task 4: Build command and setup
        cmd_setup_start = time.time()
        batch_size = get_optimal_batch_size()
        
        cmd = [
            sys.executable, 'test_video_swap_multispecific.py',
            '--isTrain', 'false',
            '--no_simswaplogo',
            '--crop_size', '224',
            '--use_mask',
            '--name', 'people',
            '--Arc_path', CONFIG["arcface_path"],
            '--video_path', str(video_path),
            '--output_path', str(output_path),
            '--temp_path', str(temp_path),
            '--multisepcific_dir', str(multispecific_dir),
            '--batchSize', str(batch_size),
        ]
        
        # Setup environment
        env = os.environ.copy()
        if torch.cuda.is_available():
            env['CUDA_VISIBLE_DEVICES'] = '0'
        
        cmd_setup_time = time.time() - cmd_setup_start
        logger.info(f"[{request_id}] ✓ Command setup completed in {cmd_setup_time:.3f}s (batch_size: {batch_size})")
        
        # Task 5: Multi-face swap processing
        logger.info(f"[{request_id}] Starting AI multi-face swap processing...")
        ai_process_start = time.time()
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=CONFIG["timeout"]
            )
            ai_process_time = time.time() - ai_process_start
            logger.info(f"[{request_id}] ✓ AI processing completed in {ai_process_time:.3f}s")
            
        except asyncio.TimeoutError:
            ai_process_time = time.time() - ai_process_start
            logger.error(f"[{request_id}] ✗ AI processing timeout after {ai_process_time:.3f}s")
            process.kill()
            await process.wait()
            raise HTTPException(status_code=504, detail="Processing timeout")
        
        # Task 6: Validation and result check
        validation_start = time.time()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            stdout_msg = stdout.decode() if stdout else ""
            logger.error(f"[{request_id}] ✗ Processing failed: {error_msg}")
            if stdout_msg:
                logger.error(f"[{request_id}] Stdout: {stdout_msg}")
            raise HTTPException(status_code=500, detail="Multi-face swap processing failed")
        
        if not output_path.exists():
            logger.error(f"[{request_id}] ✗ Output file not generated")
            raise HTTPException(status_code=500, detail="Output file not generated")
        
        # Check output file size
        output_size = output_path.stat().st_size
        validation_time = time.time() - validation_start
        logger.info(f"[{request_id}] ✓ Result validation completed in {validation_time:.3f}s (output: {output_size/1024:.0f}KB)")
        
        # Task 7: Cleanup scheduling
        cleanup_start = time.time()
        background_tasks.add_task(cleanup_files, video_path, output_path, temp_path, multispecific_dir)
        cleanup_time = time.time() - cleanup_start
        logger.info(f"[{request_id}] ✓ Cleanup scheduled in {cleanup_time:.3f}s")
        
        # Final summary
        total_time = time.time() - total_start_time
        logger.info(f"[{request_id}] ========== PERFORMANCE SUMMARY ==========")
        logger.info(f"[{request_id}] Validation:       {validation_time:.3f}s ({validation_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] Path setup:       {setup_time:.3f}s ({setup_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] File I/O:         {total_file_save_time:.3f}s ({total_file_save_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}]   - Video save:   {video_save_time:.3f}s")
        logger.info(f"[{request_id}]   - Faces save:   {faces_save_time:.3f}s")
        logger.info(f"[{request_id}] Command setup:    {cmd_setup_time:.3f}s ({cmd_setup_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] AI processing:    {ai_process_time:.3f}s ({ai_process_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] Result validation: {validation_time:.3f}s ({validation_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] Cleanup setup:    {cleanup_time:.3f}s ({cleanup_time/total_time*100:.1f}%)")
        logger.info(f"[{request_id}] TOTAL TIME:       {total_time:.3f}s")
        logger.info(f"[{request_id}] ========== REQUEST COMPLETED ==========")
        
        return FileResponse(
            path=str(output_path),
            filename=f"multi_swapped_{request_id}.mp4",
            media_type="video/mp4"
        )
        
    except HTTPException:
        total_time = time.time() - total_start_time
        logger.error(f"[{request_id}] ✗ Request failed after {total_time:.3f}s")
        cleanup_files(video_path, output_path, temp_path, multispecific_dir)
        raise
    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"[{request_id}] ✗ Unexpected error after {total_time:.3f}s: {str(e)}")
        cleanup_files(video_path, output_path, temp_path, multispecific_dir)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_v2:app", host="0.0.0.0", port=6006, reload=False)