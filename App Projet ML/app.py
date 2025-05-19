import os, cv2, numpy as np, threading, uuid
from flask import (
    Flask, request, render_template, jsonify,
    redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename

# services
from service.unet_service import process_video as unet_video_srv, process_image as unet_image_srv
from service.yolo_service import process_video as yolo_video_srv, process_image as yolo_image_srv

app = Flask(__name__)
app.config.update(
    SECRET_KEY='segmentation-semantic-app',
    UPLOAD_FOLDER='static/uploads',
    PROCESSED_FOLDER='static/uploads/processed',
    MAX_CONTENT_LENGTH=500*1024*1024,
    ALLOWED_EXTENSIONS={'mp4','avi','mov','mkv','webm','jpg','jpeg','png','bmp'}
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# In‐memory status store
processing_status = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_index():
    f = request.files.get('video')
    if not f or not allowed_file(f.filename):
        return jsonify(error='Invalid file'), 400

    sid = str(uuid.uuid4())
    fname = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(in_path)

    # mark as UNet video
    processing_status[sid] = {
        'type': 'unet-video',
        'status': 'uploading',
        'message': 'Uploaded',
        'progress': 0
    }

    threading.Thread(
        target=lambda: unet_video_srv(
            in_path, app.config['PROCESSED_FOLDER'], sid, processing_status
        ),
        daemon=True
    ).start()

    return jsonify(success=True, session_id=sid, message='Processing video...')


@app.route('/unet')
def unet():
    return render_template('unet.html')


@app.route('/unet/video', methods=['POST'])
def unet_video():
    f = request.files.get('video')
    if not f or not allowed_file(f.filename):
        return jsonify(error='Invalid file'), 400

    sid = str(uuid.uuid4())
    fname = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(in_path)

    # mark as UNet video
    processing_status[sid] = {
        'type': 'unet-video',
        'status': 'uploading',
        'message': 'Uploaded',
        'progress': 0
    }

    threading.Thread(
        target=lambda: unet_video_srv(
            in_path, app.config['PROCESSED_FOLDER'], sid, processing_status
        ),
        daemon=True
    ).start()

    return jsonify(success=True, session_id=sid, message='Processing video...')


@app.route('/unet/image', methods=['POST'])
def unet_image():
    f = request.files.get('image')
    if not f or not allowed_file(f.filename):
        return jsonify(error='Invalid file'), 400

    fname   = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(in_path)

    out = unet_image_srv(in_path, app.config['PROCESSED_FOLDER'])
    # out is dict: { original, segmented, grayscale, overlay }
    return render_template('unet_result_image.html', **out)


@app.route('/yolo')
def yolo():
    return render_template('yolo.html')


@app.route('/yolo/video', methods=['POST'])
def yolo_video():
    f = request.files.get('video')
    if not f or not allowed_file(f.filename):
        return jsonify(error='Invalid file'), 400

    sid = str(uuid.uuid4())
    fname = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(in_path)

    # mark as YOLO video
    processing_status[sid] = {
        'type': 'yolo-video',
        'status': 'uploading',
        'message': 'Uploaded',
        'progress': 0
    }

    threading.Thread(
        target=lambda: yolo_video_srv(
            in_path, app.config['PROCESSED_FOLDER'], sid, processing_status
        ),
        daemon=True
    ).start()

    return jsonify(success=True, session_id=sid, message='Processing video...')


@app.route('/yolo/image', methods=['POST'])
def yolo_image():
    f = request.files.get('image')
    if not f or not allowed_file(f.filename):
        return jsonify(error='Invalid file'), 400

    fname   = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(in_path)

    out = yolo_image_srv(in_path, app.config['PROCESSED_FOLDER'])
    return render_template('yolo_result_image.html', **out)


@app.route('/status/<sid>')
def status(sid):
    # return the status dict for this session
    return jsonify(processing_status.get(
        sid,
        {'status': 'unknown', 'message': 'No session', 'progress': 0}
    ))


@app.route('/results/<sid>')
def results(sid):
    data = processing_status.get(sid)
    if not data or data.get('status') != 'completed':
        return redirect(url_for('index'))

    # dispatch to the correct video-results template
    t = data.get('type')
    if t == 'unet-video':
        return render_template('unet_result_video.html', data=data)
    elif t == 'yolo-video':
        return render_template('yolo_result_video.html', data=data)

    # fallback
    return redirect(url_for('index'))


@app.route('/download/<fn>')
def download(fn):
    return send_from_directory(app.config['PROCESSED_FOLDER'], fn, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
