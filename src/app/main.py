#!/usr/bin/env python3
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from pdf_reader import PDFReader
import ai_inference

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can talk to backend
logging.basicConfig(level=logging.INFO)

def clean_table(ai_output):
    """
    Converts AI table output into list of lists of strings
    """
    cleaned = []
    for row in ai_output:
        cleaned_row = []
        for cell in row:
            if isinstance(cell, dict) and 'text' in cell:
                cleaned_row.append(cell['text'])
            else:
                cleaned_row.append(str(cell))
        cleaned.append(cleaned_row)
    return cleaned


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/process', methods=['POST'])
def process():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400

    pdf_file = request.files['pdf']
    temp_path = Path("/tmp") / pdf_file.filename
    pdf_file.save(temp_path)

    try:
        reader = PDFReader(str(temp_path))
        images = reader.convert_to_images()
        logging.info(f"Converted '{temp_path}' into {len(images)} images.")
        tables = ai_inference.extract_tables_from_images([str(p) for p in images])
        logging.info(f"Extracted tables: {tables}")

        # Robustly extract first table
        if isinstance(tables, dict) and 'data' in tables:
            data_list = tables['data']
            if isinstance(data_list, list) and len(data_list) > 0 and 'table' in data_list[0]:
                table_data = data_list[0]['table']
                cleaned_table = clean_table(table_data)
                logging.info(f"Cleaned table: {cleaned_table}")
                return jsonify(cleaned_table)
            else:
                logging.error(f"No table found in 'data': {tables}")
                return jsonify({
                    'error': 'No table found in AI output',
                    'data': tables
                }), 500
        elif isinstance(tables, dict) and 'table' in tables:
            table_data = tables['table']
            cleaned_table = clean_table(table_data)
            logging.info(f"Cleaned table: {cleaned_table}")
            return jsonify(cleaned_table)
        else:
            logging.error(f"AI returned unexpected output: {tables}")
            return jsonify({
                'error': 'AI returned unexpected output',
                'data': tables
            }), 500

    except Exception as e:
        logging.exception("Error during processing")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)