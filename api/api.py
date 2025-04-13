from flask import Flask, request, jsonify, send_file
from predict import process
import os
import tempfile

app = Flask(__name__)


@app.route('/process_image', methods=['POST'])
def process_image():
    # 接收上传的图片
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 保存上传的图片
        input_path = os.path.join(temp_dir, 'input.jpg')
        file.save(input_path)

        # 处理图片
        output_path = os.path.join(temp_dir, 'output.png')
        process(input_path, temp_dir)

        # 返回处理后的图片
        return send_file(output_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)